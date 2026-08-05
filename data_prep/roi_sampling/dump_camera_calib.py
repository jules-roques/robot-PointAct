"""Stage A: dump RoboCASA left/right camera calibration for ROI point sampling.

Runs in the **robocasa365 env** (MuJoCo/EGL, V100). The stored point clouds live in
the robot-base frame; to reproject them into a camera we need the intrinsics and the
base->camera transform. Because the agentview cameras are rigidly mounted to the
robot, base->cam and the intrinsics are constant across OpenDrawer episodes, so we
compute them from a handful of source episodes, verify they agree, and save a single
global calibration reused for the whole dataset.

Output npz (default: <OpenDrawer>/roi_meta/camera_calib.npz):
    left_K (3,3), left_base2cam (4,4), right_K (3,3), right_base2cam (4,4),
    image_hw (2,), plus per-episode stacks for auditing.

Usage (see roi_calib.slurm):
    python -m data_prep.roi_sampling.dump_camera_calib \
        --source-dir  $SCRATCH/.../OpenDrawer/20250816/lerobot \
        --out         $SCRATCH/.../lerobot_point_lmdb/OpenDrawer/roi_meta/camera_calib.npz \
        --num-episodes 8
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

from pointact.robot_envs.robocasa365_utils.environments import RoboCasa365Env
from pointact.roi_sampling.geometry import base2cam_from_extrinsic

LEFT_CAM = "robot0_agentview_left"
RIGHT_CAM = "robot0_agentview_right"
WRIST_CAM = "robot0_eye_in_hand"


def pose_matrix(pos, quat_xyzw) -> np.ndarray:
    """4x4 from a position and an xyzw quaternion."""
    from scipy.spatial.transform import Rotation as R

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat(np.asarray(quat_xyzw, dtype=np.float64).reshape(4)).as_matrix()
    T[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return T


def infer_env_name(source_dir: Path) -> str:
    meta_path = source_dir / "extras" / "dataset_meta.json"
    assert meta_path.exists(), f"dataset_meta.json not found at {meta_path}"
    meta = json.loads(meta_path.read_text())
    env_name = (
        meta.get("env")
        or meta.get("env_info", {}).get("env_name")
        or meta.get("env_args", {}).get("env_name")
    )
    assert env_name, f"Could not infer env_name from {meta_path}"
    return str(env_name)


def calib_for_episode(env: RoboCasa365Env, ep_dir: str) -> dict:
    obs, _ = env.reset(initial_state_dir=ep_dir, step_after_reset=False)
    base_pos = np.asarray(obs["state.base_position"], dtype=np.float64).reshape(3)
    base_quat = np.asarray(obs["state.base_rotation"], dtype=np.float64).reshape(4)  # xyzw
    out = {"base_pos": base_pos, "base_quat": base_quat}
    for tag, cam in (("left", LEFT_CAM), ("right", RIGHT_CAM), ("wrist", WRIST_CAM)):
        intrinsic, extrinsic = env._get_camera_matrices(cam)  # extrinsic = cam2world
        out[f"{tag}_K"] = np.asarray(intrinsic, dtype=np.float64)
        out[f"{tag}_base2cam"] = base2cam_from_extrinsic(extrinsic, base_pos, base_quat)

    # The wrist camera rides on the end-effector, so its base->cam is NOT constant: it moves
    # with the arm. What IS constant is its pose relative to the eef, and the eef pose in the
    # base frame is in the recorded proprio (observation.state[0:7]). So store eef2cam once
    # and reconstruct base2cam per frame as  eef2cam @ inv(T_base_eef(t))  -- no simulator
    # replay needed to point in the wrist view.
    eef_pos = np.asarray(obs["state.end_effector_position_relative"], dtype=np.float64)
    eef_quat = np.asarray(obs["state.end_effector_rotation_relative"], dtype=np.float64)
    T_base_eef = pose_matrix(eef_pos, eef_quat)
    out["eef_pose"] = T_base_eef
    out["wrist_eef2cam"] = out["wrist_base2cam"] @ T_base_eef
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True, type=Path,
                    help="The '<...>/lerobot' dir holding extras/episode_XXXXXX.")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--num-episodes", type=int, default=8)
    ap.add_argument("--image-resolution", type=int, default=256)
    ap.add_argument("--split", type=str, default="target")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    source_dir = args.source_dir.expanduser().resolve()
    ep_dirs = sorted(glob.glob(str(source_dir / "extras" / "episode_*")))
    assert ep_dirs, f"No episode dirs under {source_dir}/extras"
    # Spread the probe episodes across the dataset.
    step = max(1, len(ep_dirs) // args.num_episodes)
    probe = ep_dirs[::step][: args.num_episodes]

    env_name = infer_env_name(source_dir)
    print(f"env_name={env_name}  probing {len(probe)} episodes for calibration")

    env = RoboCasa365Env(
        env_name=env_name,
        split=args.split,
        seed=args.seed,
        image_resolution=args.image_resolution,
        use_depth=False,
        use_point_cloud=False,
        enable_render=True,
        terminate_on_success=False,
    )
    try:
        calibs = []
        for ep_dir in probe:
            c = calib_for_episode(env, ep_dir)
            calibs.append(c)
            print(f"  {os.path.basename(ep_dir)}: base_pos={c['base_pos'].round(4)}")
    finally:
        env.close()

    # Verify base->cam is constant across probe episodes.
    def stack(key):
        return np.stack([c[key] for c in calibs], axis=0)

    report = {}
    for tag in ("left", "right"):
        b2c = stack(f"{tag}_base2cam")
        K = stack(f"{tag}_K")
        b2c_dev = float(np.max(np.std(b2c, axis=0)))
        K_dev = float(np.max(np.std(K, axis=0)))
        report[tag] = (b2c_dev, K_dev)
        print(f"[{tag}] base2cam max-std={b2c_dev:.6g}  K max-std={K_dev:.6g}")
        if b2c_dev > 1e-3 or K_dev > 1e-2:
            print(f"  WARNING: {tag} calibration is NOT constant across episodes "
                  f"(dev above threshold). A global calibration may be inaccurate; "
                  f"per-episode calibration would be required.")

    # The wrist camera's constant is eef2cam, not base2cam. Probe episodes reset to similar
    # arm poses, so a small spread here is weak evidence -- the real check is the reprojection
    # test in validate_wrist_calib.py, which uses frames with genuinely different arm poses.
    e2c = stack("wrist_eef2cam")
    e2c_dev = float(np.max(np.std(e2c, axis=0)))
    print(f"[wrist] eef2cam max-std={e2c_dev:.6g} (across probe episodes; "
          f"validate against moving frames before trusting it)")
    if e2c_dev > 1e-3:
        print("  WARNING: wrist eef2cam is not constant -- the camera may not be rigidly "
              "mounted to the eef frame this state reports.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        left_K=np.median(stack("left_K"), axis=0),
        left_base2cam=np.median(stack("left_base2cam"), axis=0),
        right_K=np.median(stack("right_K"), axis=0),
        right_base2cam=np.median(stack("right_base2cam"), axis=0),
        image_hw=np.array([args.image_resolution, args.image_resolution], dtype=np.int64),
        env_name=env_name,
        wrist_K=np.median(stack("wrist_K"), axis=0),
        wrist_eef2cam=np.median(e2c, axis=0),
        probe_base_pos=stack("base_pos"),
        probe_left_base2cam=stack("left_base2cam"),
        probe_right_base2cam=stack("right_base2cam"),
        probe_wrist_eef2cam=e2c,
        probe_eef_pose=stack("eef_pose"),
    )
    print(f"saved global calibration -> {args.out}")


if __name__ == "__main__":
    main()
