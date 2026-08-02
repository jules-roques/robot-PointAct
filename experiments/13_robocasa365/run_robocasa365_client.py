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


def render_rollouts(kept: dict, save_dir: pathlib.Path, sampling: str, sigma: float, floor: float):
    """Write the chosen rollouts as point-cloud animations, matching the training figures.

    Reuses viz_sampling_episode.build_figure so eval and training artefacts render identically
    (same weight colour scale, same controls). Weights are recomputed here from the buffered
    cloud and eef rather than returned by the server, which keeps the wire protocol unchanged.
    """
    import types

    from data_prep.roi_sampling.viz_sampling_episode import (
        CLOG_MAX, CLOG_MIN, build_figure, oversampling_factor,
    )
    from pointact.roi_sampling.geometry import eef_density_weights

    style = types.SimpleNamespace(
        color_by="weight", dark=True, frame_ms=120, point_size=2.0,
        near_radius=0.15, roi_radius=0.15, roi_radius_scale=1.0,
    )
    written = []
    for outcome, trial, frames in choose_rollouts(kept):
        frames_data = []
        for fd in frames:
            pts, eef = fd["points"], fd["eef"]
            weights = (eef_density_weights(pts[:, :3], eef, sigma, floor)
                       if sampling in ("eef", "anchor") else None)
            logw = np.log2(np.clip(oversampling_factor(weights, len(pts)), 1e-6, None))
            frames_data.append({
                "pts": pts[:, :3], "colors": logw.astype(np.float32), "n_sel": len(pts),
                "anchor": eef if weights is not None else None, "frame": fd["step"],
                "near_frac": None, "n_handle_total": 0, "n_handle_sel": 0,
                "handle_recall": None, "n_cloud": len(pts), "n_frames": len(frames),
                "phase": outcome,
            })
        if not frames_data:
            continue
        fig = build_figure(sampling, frames_data, trial, f"rollout {outcome}", style)
        out = save_dir / f"rollout_{outcome}_{trial}.html"
        # CDN plotly: inlining it would add ~3.5MB to every figure, and W&B renders these in
        # the viewer's browser, which can fetch it.
        fig.write_html(str(out), include_plotlyjs="cdn", auto_play=False)
        written.append(out)
    return written


def main(args: ClientArgs) -> None:
    assert args.pred_rot_type in ["rot6d", "euler"]

    policy_client = PolicyClient(args.host, args.port)
    while not policy_client.ping():
        pass
    print(f"Server is running on host {args.host} port {args.port}")

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
        use_segmentation=args.oracle_anchor,
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
    import collections

    for episode_idx in tqdm.tqdm(range(args.num_trials)):
        # Each reset re-randomises the scene. TODO(verify): reseeding per trial for
        # reproducible-yet-varied scenes may need explicit env support.
        obs, _ = env.reset()
        action_plan = collections.deque()
        replay_images = []
        rollout_frames = []
        success = False

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
                if args.use_depth:
                    # Send a single fused 3-view point cloud (matches training data). This uses the
                    # server's existing-points branch directly, so we don't depend on the server
                    # re-deriving cameras from select_video_keys — the fused cloud is what training
                    # saw (points_3views: left+right+wrist), avoiding any view mismatch.
                    fused = build_fused_point_cloud(obs)
                    batch["observation.points"] = [fused]
                    if args.viz_rollouts and want_more_rollouts(kept_rollouts):
                        # The cloud as sent, with the eef so the weighting can be recomputed
                        # offline exactly as the server did it.
                        rollout_frames.append(
                            {"points": fused.astype(np.float32),
                             "eef": np.asarray(obs["observation.state"][:3], dtype=np.float32),
                             "step": _t}
                        )

                if args.oracle_anchor:
                    anchor = ground_truth_anchor(obs)
                    if anchor is not None:
                        batch["observation.sampling_anchor"] = [anchor]

                points_workspace = env.get_points_workspace(obs)
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

        total_episodes += 1
        total_successes += int(success)
        per_trial.append({"trial": int(episode_idx), "success": bool(success), "task": str(obs.get("task", ""))})
        if args.viz_rollouts and rollout_frames:
            # Keep the FIRST of each outcome and stop buffering that class; two of one class
            # only if the other never occurs, decided once the trials are done.
            bucket = kept_rollouts.setdefault("success" if success else "failure", [])
            if len(bucket) < 2:
                bucket.append((int(episode_idx), rollout_frames))
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

    logging.info(f"Total success rate: {total_successes / max(total_episodes, 1):.4f}")

    # Dump the paired record next to the run's other results, keyed by seed so two runs at
    # different seeds can be pooled without double-counting scenes.
    if args.save_dir:
        out = Path(args.save_dir) / f"per_trial_seed{args.seed}_n{total_episodes}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(
                {
                    "seed": int(args.seed),
                    "env_name": args.env_name,
                    "num_trials": int(total_episodes),
                    "successes": int(total_successes),
                    "success_rate": total_successes / max(total_episodes, 1),
                    "oracle_anchor": bool(args.oracle_anchor),
                    "trials": per_trial,
                },
                f,
                indent=2,
            )
        logging.info(f"wrote per-trial outcomes to {out}")

        if args.viz_rollouts and kept_rollouts:
            try:
                paths = render_rollouts(
                    kept_rollouts, Path(args.save_dir), args.point_sampling_for_viz,
                    args.viz_sigma, args.viz_floor,
                )
                for path in paths:
                    logging.info(f"wrote rollout animation {path} "
                                 f"({path.stat().st_size / 1e6:.1f} MB)")
            except Exception as exc:  # noqa: BLE001 - a figure must never lose the eval result
                logging.warning(f"rollout rendering failed: {type(exc).__name__}: {exc}")
    logging.info(f"Total episodes: {total_episodes}")
    env.close()


if __name__ == "__main__":
    tyro.cli(main)
