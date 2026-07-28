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
    import collections

    for episode_idx in tqdm.tqdm(range(args.num_trials)):
        # Each reset re-randomises the scene. TODO(verify): reseeding per trial for
        # reproducible-yet-varied scenes may need explicit env support.
        obs, _ = env.reset()
        action_plan = collections.deque()
        replay_images = []
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
                    batch["observation.points"] = [build_fused_point_cloud(obs)]

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
    logging.info(f"Total episodes: {total_episodes}")
    env.close()


if __name__ == "__main__":
    tyro.cli(main)
