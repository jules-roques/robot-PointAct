"""RoboCasa365 evaluation client for the PointAct policy server.

Adapted from experiments/2_libero/run_libero_client.py. Structurally different from Libero:
RoboCasa365 is a *single task* per env (e.g. OpenDrawer) evaluated over N trials with the
simulator re-randomising the scene each reset, rather than a task suite with recorded initial
states.

Runs in the robocasa365 environment (needs MuJoCo/EGL + a GPU); the policy server runs in the
pointact environment. See eval_robocasa365.sh.

STATUS: scaffold. The server round-trip mirrors the Libero client, but the action/state
plumbing for the 13-D PandaOmron action space is only partially validated — every spot that
depends on RoboCasa365 action semantics is marked TODO(verify). Confirm these against a short
run before trusting success numbers.
"""

import dataclasses
import datetime as dt
import json
import logging
import pathlib
import sys
from pathlib import Path

import imageio
import numpy as np
import tqdm
import tyro

from pointact.robot_envs.robocasa365_utils.environments import (
    POINT_LABEL_TARGET_DOOR,
    POINT_LABEL_TARGET_HANDLE,
    RoboCasa365Env,
)
from pointact.roi_sampling.live_anchor import LiveMolmoAnchor
from pointact.utils.rotation import convert_rotation
from pointact.utils.server_client import PolicyClient
from pointact.utils.torch_utils import set_seed


@dataclasses.dataclass
class ClientArgs:
    seed: int = 7
    env_name: str = "OpenDrawer"  # any RoboCasa365 task in TASK_SET_REGISTRY["all_tasks"]
    split: str = "target"  # pretrain, target
    num_trials: int = 50  # rollouts (each with a freshly randomised scene)
    image_size: int = 256
    max_steps: int = 0  # 0 -> use the task horizon from the env

    repo_id: str | None = "OpenDrawer"  # must match the training data config repo_id
    post_process_action: bool = True
    replan_steps: int = 8
    pred_rot_type: str = "rot6d"  # euler, rot6d — must match the trained model
    use_depth: bool = True  # PointAct needs point clouds
    # Send the simulator's ground-truth handle position as the sampling anchor, for policies
    # trained with oracle sampling. This is privileged information: it makes the evaluation an
    # upper bound, not a deployable system. Must be paired with --args.point_sampling anchor on
    # the server, and must be off for the uniform and eef-density policies.
    oracle_anchor: bool = False

    # Produce the sampling anchor live with MolmoPoint, for policies trained with
    # molmo_sampling. Unlike --args.oracle_anchor this uses no privileged information: the
    # pointer sees the same camera images the policy does. It needs a second server
    # (scripts/run_molmo_server.py) on --args.molmo_port, and like the oracle it must be
    # paired with --args.point_sampling anchor on the policy server.
    molmo_anchor: bool = False
    # Which of the task's pointing queries become Gaussian centres. MUST match the training
    # data config's molmo_anchor_ids: (0,) is the manipulated object alone, (0, 1) adds the
    # destination. Getting this wrong is a silent train/eval mismatch, which is the exact
    # failure that voided the first round of stage-3 numbers.
    molmo_anchor_ids: tuple[int, ...] = (0,)
    # Render segmentation and log the live anchor's distance to the ground-truth target. Only
    # OpenDrawer exposes those labels. Off by default: it costs an extra render pass per step,
    # and it is a diagnostic, not part of the arm.
    molmo_audit_gt: bool = False

    # Capture two rollouts as point-cloud animations: one success and one failure if both
    # occur, otherwise two of whichever did. Shows what the policy actually saw, which a
    # success rate cannot. Buffers only the current episode plus the two kept ones.
    viz_rollouts: bool = False
    # How to colour the rollout figures: must match how the checkpoint was trained, exactly as
    # --args.point_sampling does for the server. eval_robocasa365.sh passes the derived value.
    point_sampling_for_viz: str = "uniform"   # uniform | eef | anchor
    viz_sigma: float = 0.08
    viz_floor: float = 0.05

    save_dir: str = ""
    save_video: bool = False
    verbose: bool = False

    host: str = "localhost"
    port: int = 5555
    molmo_host: str = "localhost"
    molmo_port: int = 5556


def setup_logging(filename=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    if filename is not None:
        fh = logging.FileHandler(filename)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def build_fused_point_cloud(obs: dict) -> np.ndarray:
    """Fuse the left/right/wrist camera clouds into one [N, 6] (xyz + rgb) array.

    Mirrors data_prep/robocasa365_to_lerobot/replay.py:make_point_cloud, which built the
    3-view training clouds: per-camera xyz (already in the robot-base frame) concatenated with
    rgb/255, stacked across cameras. Workspace crop, 1cm voxel downsample and the 4096-point
    subsample are applied server-side by the processor, so they are intentionally omitted here.
    """
    clouds = []
    for cam in ("left", "right", "wrist"):
        xyz = np.asarray(obs[f"observation.points.{cam}"], dtype=np.float32)
        rgb = np.asarray(obs[f"observation.images.{cam}_image"], dtype=np.float32) / 255.0
        clouds.append(np.concatenate([xyz, rgb], axis=-1).reshape(-1, 6))
    return np.concatenate(clouds, axis=0)


def ground_truth_anchor(obs: dict) -> np.ndarray | None:
    """Centroid of the ground-truth handle points, in the point-cloud (robot-base) frame.

    Mirrors LeRobotPointCloudDataset.oracle_anchor: prefer the handle, fall back to the
    drawer/door panel it sits on, and give up (leaving the sampler uniform) if neither is
    visible in any camera this frame.
    """
    xyz, labels = [], []
    for cam in ("left", "right", "wrist"):
        key = f"observation.point_labels.{cam}"
        if key not in obs:
            return None
        xyz.append(np.asarray(obs[f"observation.points.{cam}"], dtype=np.float32).reshape(-1, 3))
        labels.append(np.asarray(obs[key]).reshape(-1))
    xyz = np.concatenate(xyz, axis=0)
    labels = np.concatenate(labels, axis=0)

    for wanted in ((POINT_LABEL_TARGET_HANDLE,), (POINT_LABEL_TARGET_DOOR,)):
        mask = np.isin(labels, wanted)
        if mask.any():
            return xyz[mask].mean(axis=0).astype(np.float32)
    return None


def prepare_state(raw_state: np.ndarray, pred_rot_type: str) -> np.ndarray:
    """Convert the eef rotation quat in observation.state to the model's rotation type.

    RoboCasa365 observation.state layout (16-D):
      [eef_pos_rel(3), eef_quat_rel_xyzw(4), gripper_qpos(2), base_pos(3), base_quat_xyzw(4)]

    TODO(verify): only the eef rotation at [3:7] is converted here, matching the single
    --state_rotation_slice used to build the norm stats. The base rotation quat at [12:16] is
    passed through unchanged. If training normalises the base rotation differently, mirror it.
    """
    rot = raw_state[3:7]
    if pred_rot_type == "euler":
        rot = convert_rotation(rot, "quat", "euler", quat_order_src="xyzw", euler_order_dst="xyz")
    elif pred_rot_type == "rot6d":
        rot = convert_rotation(rot, "quat", "rot6d", quat_order_src="xyzw")
    return np.concatenate([raw_state[:3], rot, raw_state[7:]])


def want_more_rollouts(kept: dict) -> bool:
    """Stop buffering once two of each outcome are held.

    Without this every episode's clouds accumulate: ~26MB per rollout x 150 trials would be
    several GB for two figures.
    """
    return any(len(kept.get(k, [])) < 2 for k in ("success", "failure"))


def choose_rollouts(kept: dict) -> list[tuple[str, int, list]]:
    """One success and one failure if both occurred, else two of whichever did."""
    successes, failures = kept.get("success", []), kept.get("failure", [])
    if successes and failures:
        picked = [("success", *successes[0]), ("failure", *failures[0])]
    else:
        available = successes or failures
        label = "success" if successes else "failure"
        picked = [(label, *entry) for entry in available[:2]]
    return picked


def save_rollout_npz(kept: dict, save_dir: pathlib.Path, sampling: str) -> list:
    """Dump the chosen rollouts as .npz for later rendering.

    Rendering happens in a separate step, not here: this client runs in the robocasa365 env,
    which has neither plotly nor lmdb (the viz module imports both), while the pointact env
    has all three. Writing raw arrays keeps this side numpy-only.
    """
    written = []
    for outcome, trial, frames in choose_rollouts(kept):
        if not frames:
            continue
        payload = {"sampling": np.array(sampling), "trial": np.array(trial),
                   "outcome": np.array(outcome), "n_frames": np.array(len(frames))}
        for i, fd in enumerate(frames):
            payload[f"points_{i}"] = fd["points"].astype(np.float32)
            payload[f"eef_{i}"] = fd["eef"].astype(np.float32)
            payload[f"step_{i}"] = np.array(fd["step"])
            # Absent on the uniform/eef arms, and on anchor frames that fell back to uniform.
            # The renderer treats a missing key as "no anchor this frame".
            if fd.get("anchor") is not None:
                payload[f"anchor_{i}"] = fd["anchor"].astype(np.float32)
        out = save_dir / f"rollout_{outcome}_{trial}.npz"
        np.savez_compressed(out, **payload)
        written.append(out)
    return written


def main(args: ClientArgs) -> None:
    assert args.pred_rot_type in ["rot6d", "euler"]
    # Both fill observation.sampling_anchor, so enabling both would mean the last writer wins
    # -- an evaluation whose sampling prior is decided by statement order.
    assert not (args.oracle_anchor and args.molmo_anchor), \
        "--args.oracle_anchor and --args.molmo_anchor both fill the sampling anchor; pick one"

    policy_client = PolicyClient(args.host, args.port)
    while not policy_client.ping():
        pass
    print(f"Server is running on host {args.host} port {args.port}")

    molmo_anchor = None
    if args.molmo_anchor:
        molmo_client = PolicyClient(args.molmo_host, args.molmo_port)
        while not molmo_client.ping():
            pass
        print(f"Molmo pointer is running on host {args.molmo_host} port {args.molmo_port}")
        molmo_anchor = LiveMolmoAnchor(args.env_name, args.molmo_anchor_ids, molmo_client,
                                       verbose=args.verbose)

    video_out_dir = None
    log_filename = None
    if args.save_dir:
        base = Path(args.save_dir, f"{args.env_name}-{dt.datetime.now():%Y-%m-%d-%H-%M-%S}")
        base.parent.mkdir(parents=True, exist_ok=True)
        log_filename = f"{base}.log"
        if args.save_video:
            video_out_dir = f"{base}+videos"
            pathlib.Path(video_out_dir).mkdir(parents=True, exist_ok=True)
    setup_logging(log_filename)
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    set_seed(args.seed)

    env = RoboCasa365Env(
        env_name=args.env_name,
        split=args.split,
        seed=args.seed,
        image_resolution=args.image_size,
        use_depth=args.use_depth,
        use_point_cloud=args.use_depth,
        # Segmentation is what ground_truth_anchor reads. The molmo arm does not need it to
        # run -- only to be audited against the GT target it is trying to find.
        use_segmentation=args.oracle_anchor or args.molmo_audit_gt,
        enable_render=True,
        terminate_on_success=True,  # so `done` marks success, as in the Libero client
    )
    max_steps = args.max_steps or env.max_episode_steps

    total_episodes, total_successes = 0, 0
    # Per-trial outcomes. Every arm is evaluated on the SAME seeded scene stream, so trial i is
    # the same kitchen for every policy: the arms are naturally PAIRED. Recording the per-trial
    # outcome (rather than only the aggregate rate) is what makes a paired test -- McNemar on
    # the discordant trials -- possible, which is markedly more powerful than comparing two
    # independent proportions at these sample sizes.
    per_trial: list[dict] = []
    kept_rollouts: dict[str, list] = {}
    # Episodes the simulator could not present a usable observation for (see the guard in the
    # rollout loop). Skipped rather than counted as policy failures, and reported so the
    # denominator is never silently smaller than it looks.
    skipped_episodes: list[dict] = []

    def dump_results(final: bool = False):
        """Write the per-trial record. Called after EVERY episode, not just at the end.

        A single bad scene used to abort the job and lose every completed trial with it -- one
        150-trial run died after 1h18 with nothing to show. Rewriting the file each episode
        costs nothing at this size and means a crash costs one episode.
        """
        if not args.save_dir:
            return None
        # The in-progress file must NOT match the per_trial_seed*_n*.json glob that pooling
        # uses: naming it by episode count would leave one file per episode and every trial
        # would be counted many times over.
        out = Path(args.save_dir) / (
            f"per_trial_seed{args.seed}_n{total_episodes}.json" if final
            else f"per_trial_seed{args.seed}_partial.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({
                "seed": int(args.seed),
                "env_name": args.env_name,
                "num_trials": int(total_episodes),
                "successes": int(total_successes),
                "success_rate": total_successes / max(total_episodes, 1),
                "oracle_anchor": bool(args.oracle_anchor),
                "molmo_anchor": bool(args.molmo_anchor),
                # The molmo arm's failure mode is quiet: a pointer that answers nothing turns
                # every frame into a uniform draw and the run still reports a plausible
                # success rate. Writing the counters next to the rate is what tells "did not
                # help" apart from "did not run".
                "molmo_stats": molmo_anchor.stats.summary() if molmo_anchor else None,
                "skipped": skipped_episodes,
                "trials": per_trial,
            }, f, indent=2)
        if final:
            (Path(args.save_dir) / f"per_trial_seed{args.seed}_partial.json").unlink(missing_ok=True)
        return out
    import collections

    for episode_idx in tqdm.tqdm(range(args.num_trials)):
        # Each reset re-randomises the scene. TODO(verify): reseeding per trial for
        # reproducible-yet-varied scenes may need explicit env support.
        obs, _ = env.reset()
        action_plan = collections.deque()
        replay_images = []
        rollout_frames = []
        success = False

        episode_failed = None
        try:
          for _t in range(max_steps):
            replay_images.append(obs["observation.images.left_image"])

            state = prepare_state(np.asarray(obs["observation.state"]), args.pred_rot_type)

            if not action_plan:
                batch = {
                    # left + right agentviews feed the VLM (matches the training data config's
                    # select_video_keys / video_key_ids_for_vlm = [0,1]; wrist is not a VLM view).
                    # The server picks whichever keys the checkpoint's baked config names, so
                    # sending all three here is harmless — wrist is simply ignored by the VLM.
                    "observation.images.left_image": [obs["observation.images.left_image"]],
                    "observation.images.right_image": [obs["observation.images.right_image"]],
                    "observation.images.wrist_image": [obs["observation.images.wrist_image"]],
                    "observation.state": [state],
                    "task": [obs["task"]],
                    "repo_id": [args.repo_id],
                }
                # Needed by the molmo lift (it crops candidate pixels to the same box the
                # server crops the cloud to) as well as by the server, so it is derived
                # before the anchor block rather than just before the request.
                points_workspace = env.get_points_workspace(obs)

                anchor = None
                if args.oracle_anchor:
                    anchor = ground_truth_anchor(obs)
                elif molmo_anchor is not None:
                    # Once per replan, matching the stride the training cache was built at:
                    # both the cache and this hold one anchor for the following 8 steps.
                    anchor = molmo_anchor(obs, points_workspace)
                    if args.molmo_audit_gt:
                        molmo_anchor.record_gt_error(anchor, ground_truth_anchor(obs))
                if anchor is not None:
                    batch["observation.sampling_anchor"] = [anchor]

                if args.use_depth:
                    # Send a single fused 3-view point cloud (matches training data). This uses the
                    # server's existing-points branch directly, so we don't depend on the server
                    # re-deriving cameras from select_video_keys — the fused cloud is what training
                    # saw (points_3views: left+right+wrist), avoiding any view mismatch.
                    fused = build_fused_point_cloud(obs)
                    batch["observation.points"] = [fused]
                    if args.viz_rollouts and want_more_rollouts(kept_rollouts):
                        # The cloud as sent, with the eef so the weighting can be recomputed
                        # offline exactly as the server did it. For the anchor arms the eef is
                        # NOT the centre, so the anchor actually used is recorded too --
                        # without it the renderer drew the eef bump for every arm and the
                        # oracle/molmo figures showed a density the server never applied.
                        rollout_frames.append(
                            {"points": fused.astype(np.float32),
                             "eef": np.asarray(obs["observation.state"][:3], dtype=np.float32),
                             "anchor": (None if anchor is None
                                        else np.atleast_2d(np.asarray(anchor, np.float32))),
                             "step": _t}
                        )

                ov_out = policy_client.get_action(
                    batch,
                    options={
                        "pred_rot_type": args.pred_rot_type,
                        "points_workspace": points_workspace,
                    },
                )
                action_chunk = ov_out.action[0].copy()

                # No Libero-style gripper remap here: for RoboCasa365 gripper_close sits mid-vector
                # (index 7 of the 13-D action) and the env's unmap_panda_omron_action thresholds it
                # at 0.5 internally, so the raw server action is stepped as-is.
                assert len(action_chunk) >= args.replan_steps
                action_plan.extend(action_chunk[: args.replan_steps])

            # The server returns a 13-D env-ready action:
            #   [eef_pos(3), eef_quat_xyzw(4), gripper_close(1), base_motion(4), control_mode(1)]
            # pred_rot_type (rot6d->quat) and the absolute-position offset are applied server-side
            # in _build_action_output, so the action goes straight to env.step (no reconstruction).
            obs, reward, done, info = env.step(action_plan.popleft())
            # Count success from the explicit flag, not `done`: robosuite also returns
            # done=True at the horizon timeout, which would otherwise inflate the rate.
            if info.get("success", False):
                success = True
                break
            if done:
                break

        except RuntimeError as exc:
            # Chiefly "Point cloud is empty after workspace filtering and downsampling": a
            # degenerate randomly-generated scene, not a policy failure. Counting it as one
            # would bias the success rate down, so the episode is dropped from the denominator
            # and recorded. Killed three otherwise-complete 150-trial runs before this guard.
            episode_failed = str(exc)
            logging.warning(f"[{episode_idx}] skipped: {episode_failed}")

        if episode_failed is not None:
            skipped_episodes.append({"trial": int(episode_idx), "error": episode_failed})
            dump_results()
            continue

        total_episodes += 1
        total_successes += int(success)
        per_trial.append({"trial": int(episode_idx), "success": bool(success), "task": str(obs.get("task", ""))})
        if args.viz_rollouts and rollout_frames:
            # Keep the FIRST of each outcome and stop buffering that class; two of one class
            # only if the other never occurs, decided once the trials are done.
            bucket = kept_rollouts.setdefault("success" if success else "failure", [])
            if len(bucket) < 2:
                bucket.append((int(episode_idx), rollout_frames))

        dump_results()
        suffix = "success" if success else "failure"
        if video_out_dir is not None:
            imageio.mimwrite(
                pathlib.Path(video_out_dir) / f"{args.env_name}_rollout{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
        if args.verbose:
            logging.info(
                f"[{total_episodes}] success={success} "
                f"running={total_successes}/{total_episodes} "
                f"({100 * total_successes / total_episodes:.1f}%)"
            )

    # A run where nothing was evaluated is not a 0% policy, it is a broken eval -- but both
    # write success_rate 0.0 and exit 0, and the four stage-2 arms did exactly that for a whole
    # sweep (repo_id mismatch -> server KeyError -> every scene "skipped"). Fail loudly instead:
    # a wrong number that looks plausible is the one outcome this campaign cannot afford.
    if total_episodes == 0 and skipped_episodes:
        errors = sorted({str(s["error"]) for s in skipped_episodes})
        logging.error(
            f"all {len(skipped_episodes)} trial(s) skipped, nothing evaluated. "
            f"Distinct errors: {errors[:5]}"
        )
        if args.save_dir:
            dump_results(final=True)
        raise SystemExit(2)

    logging.info(f"Total success rate: {total_successes / max(total_episodes, 1):.4f}")

    if molmo_anchor is not None:
        summary = molmo_anchor.stats.summary()
        logging.info(f"MolmoPoint anchors: {json.dumps(summary)}")
        # A frame with no anchor is a uniform draw against a non-uniformly-trained policy. A
        # few are expected (the target leaves frame); a majority means the arm did not really
        # run and the success rate above is measuring something else.
        if summary["frame_cover"] < 0.5:
            logging.warning(
                f"MolmoPoint produced an anchor for only {100 * summary['frame_cover']:.1f}% "
                f"of replans -- most frames fell back to UNIFORM sampling, which does not "
                f"match how this checkpoint was trained. Treat the rate above as suspect."
            )

    # The record is rewritten after every episode (dump_results); this is the final one.
    # Keyed by seed so two runs at different seeds pool without double-counting scenes.
    if args.save_dir:
        out = dump_results(final=True)
        logging.info(f"wrote per-trial outcomes to {out}")
        if skipped_episodes:
            logging.warning(f"{len(skipped_episodes)} episode(s) skipped as unusable scenes; "
                            f"denominator is {total_episodes}, not {args.num_trials}")

        if args.viz_rollouts and kept_rollouts:
            try:
                for path in save_rollout_npz(kept_rollouts, Path(args.save_dir),
                                             args.point_sampling_for_viz):
                    logging.info(f"wrote rollout data {path} "
                                 f"({path.stat().st_size / 1e6:.1f} MB)")
            except Exception as exc:  # noqa: BLE001 - a figure must never lose the eval result
                logging.warning(f"rollout dump failed: {type(exc).__name__}: {exc}")
    logging.info(f"Total episodes: {total_episodes}")
    env.close()


if __name__ == "__main__":
    tyro.cli(main)
