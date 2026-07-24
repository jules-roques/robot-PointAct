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

from pointact.robot_envs.robocasa365_utils.environments import RoboCasa365Env
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


def reconstruct_env_action(action: np.ndarray, pred_rot_type: str) -> np.ndarray:
    """Turn one predicted action step into the 13-D flat PandaOmron action expected by step().

    Env action layout (see convert_flat_action_to_panda_omron_action):
      [eef_pos(3), eef_quat_xyzw(4), gripper_close(1), base_motion(4), control_mode(1)]

    The model emits rotation as pred_rot_type, so the trailing fields shift accordingly:
      rot6d -> action = [pos(3), rot6d(6), gripper(1), base(4), mode(1)]  (15-D)
      euler -> action = [pos(3), euler(3), gripper(1), base(4), mode(1)]  (12-D)

    TODO(verify): index math and the gripper/control_mode handling below against a real
    checkpoint's output. These are the fields most likely to need adjustment.
    """
    if pred_rot_type == "rot6d":
        rot_len = 6
    elif pred_rot_type == "euler":
        rot_len = 3
    else:
        raise ValueError(pred_rot_type)

    pos = action[:3]
    rot = action[3:3 + rot_len]
    tail = action[3 + rot_len:]  # [gripper(1), base_motion(4), control_mode(1)]

    if pred_rot_type == "rot6d":
        quat = convert_rotation(rot, "rot6d", "quat", quat_order_dst="xyzw")
    else:
        quat = convert_rotation(rot, "euler", "quat", euler_order_src="xyz", quat_order_dst="xyzw")

    return np.concatenate([pos, quat, tail])  # 13-D flat action for RoboCasa365Env.step


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
        enable_render=True,
        terminate_on_success=True,  # so `done` marks success, as in the Libero client
    )
    max_steps = args.max_steps or env.max_episode_steps

    total_episodes, total_successes = 0, 0
    import collections

    for episode_idx in tqdm.tqdm(range(args.num_trials)):
        # Each reset re-randomises the scene. TODO(verify): reseeding per trial for
        # reproducible-yet-varied scenes may need explicit env support.
        obs = env.reset()
        action_plan = collections.deque()
        replay_images = []
        done = False

        for _t in range(max_steps):
            replay_images.append(obs["observation.images.left_image"])

            state = prepare_state(np.asarray(obs["observation.state"]), args.pred_rot_type)

            if not action_plan:
                batch = {
                    # left agentview is the VLM view (matches the training data config)
                    "observation.images.left_image": [obs["observation.images.left_image"]],
                    "observation.images.right_image": [obs["observation.images.right_image"]],
                    "observation.images.wrist_image": [obs["observation.images.wrist_image"]],
                    "observation.state": [state],
                    "task": [obs["task"]],
                    "repo_id": [args.repo_id],
                }
                if args.use_depth:
                    batch.update({
                        "observation.points.left": [obs["observation.points.left"]],
                        "observation.points.right": [obs["observation.points.right"]],
                        "observation.points.wrist": [obs["observation.points.wrist"]],
                    })

                points_workspace = env.get_points_workspace(obs)
                ov_out = policy_client.get_action(
                    batch,
                    options={
                        "pred_rot_type": args.pred_rot_type,
                        "points_workspace": points_workspace,
                    },
                )
                action_chunk = ov_out.action[0].copy()

                # TODO(verify): Libero remaps the last dim (gripper) as
                #   action[..., -1] = 2*(1-action[..., -1]) - 1
                # For RoboCasa365 the last dim is control_mode and gripper_close sits mid-vector,
                # so that remap does NOT apply as-is. Decide the correct gripper handling.
                assert len(action_chunk) >= args.replan_steps
                action_plan.extend(action_chunk[: args.replan_steps])

            step_action = reconstruct_env_action(action_plan.popleft(), args.pred_rot_type)
            obs, reward, done, info = env.step(step_action)
            if done:
                total_successes += 1
                break

        total_episodes += 1
        suffix = "success" if done else "failure"
        if video_out_dir is not None:
            imageio.mimwrite(
                pathlib.Path(video_out_dir) / f"{args.env_name}_rollout{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
        if args.verbose:
            logging.info(
                f"[{total_episodes}] success={done} "
                f"running={total_successes}/{total_episodes} "
                f"({100 * total_successes / total_episodes:.1f}%)"
            )

    logging.info(f"Total success rate: {total_successes / max(total_episodes, 1):.4f}")
    logging.info(f"Total episodes: {total_episodes}")
    env.close()


if __name__ == "__main__":
    tyro.cli(main)
