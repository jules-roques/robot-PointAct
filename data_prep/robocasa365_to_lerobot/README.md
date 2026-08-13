# RoboCasa365 to LeRobot Point Dataset

This converter uses action rollout replay for RoboCasa365 PandaOmron demos and writes a LeRobot dataset plus a fused 3-view point-cloud LMDB.

Note: `robocasa365` simulator cannot be properly ran in H100 GPU.

Run it in two stages, in **two different environments**:

- `envs/robocasa365`: read the source parquet actions, replay them in simulation, and cache
  replay outputs (needs RoboCasa / robosuite / MuJoCo).
- the root `pointact` env: write the cached replay into LeRobot format.

Both are uv projects, selected with `--project`; no activation step is needed. See
[`envs/robocasa365/README.md`](../../envs/robocasa365/README.md) for how to obtain the source
demonstrations and the kitchen assets first — replay cannot run without them.

All commands below are run from the repository root.

## 1. Replay Actions in Simulation

```bash
taskname=OpenDrawer
# $ROBOCASA_DATASET_ROOT must match DATASET_BASE_PATH in robocasa/macros_private.py.
# The trailing date is per task and comes from robocasa's dataset registry.
input_dir=$ROBOCASA_DATASET_ROOT/v1.0/target/atomic/$taskname/20250816

export MUJOCO_GL=egl   # headless GPU rendering

uv run --project envs/robocasa365 data_prep/robocasa365_to_lerobot/replay.py \
  --input-dir $input_dir/lerobot \
  --cache-dir $input_dir/replay_cache \
  --resume
```

Add `--episodes 0` for a small debug run.
Add `--resume` to skip already-completed episode caches and continue the rest.

This writes per-episode replay caches under `replay_cache/episodes`. Each episode
has one metadata/image/action `.npz` and one fused point-cloud `_points.npy`.

Not every episode replays successfully — `replay_summary.jsonl` records the outcome per
episode. On OpenDrawer, 496 of 514 episodes replayed successfully and only those were
converted. As a SLURM job, use `replay.slurm` (Jean Zay, supports job-array chunking) or
`experiments/13_robocasa365/data_prep_replay.slurm` (CLEPS).

## 2. Write LeRobot Dataset

```bash
export SVT_LOG=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=TRUE
export HDF5_USE_FILE_LOCKING=FALSE

uv run data_prep/robocasa365_to_lerobot/convert.py \
  --cache-dir $input_dir/replay_cache \
  --output-dir $ROBOCASA_DATASET_ROOT/lerobot_point_lmdb/${taskname} \
  --repo-id ${taskname} \
  --image-writer-threads 12
```

`--episodes` restricts conversion to a subset (e.g. only the successful replays).

**Note on episode numbering:** `convert.py` numbers its output episodes densely by position
among the replay cache files, silently dropping failed replays. When any episode fails, the
converted episode index therefore does *not* equal the source episode index. Anything joining
converted data back to the source demonstrations needs that mapping — do not assume identity.

`convert.py` writes one LMDB key per timestep and commits point-cloud writes every
50 frames by default. Use `--lmdb-commit-every N` to tune this, or
`--lmdb-commit-every 0` for one transaction per episode. It also enables LeRobot
async image writing with `--image-writer-threads 12` by default.

## 3. Normalization statistics

Training additionally needs a state/action normalization file, generated with
`data_prep/prepare_robot_state_action_stats.py`. Two flags are mandatory for RoboCasa365 —
see the "State / action statistics" section of
[`experiments/13_robocasa365/README.md`](../../experiments/13_robocasa365/README.md).

The final dataset is written to:

```text
lerobot_point_lmdb/OpenDrawer/
  data/
  videos/
  meta/
  points_3views/
  cache_meta.json
  replay_summary.jsonl
```

## Conventions

- Source action order: `base_motion, control_mode, eef_position, eef_rotation_axisangle, gripper_close`.
- Simulator action order: `eef_position, eef_rotation_quat_xyzw, gripper_close, base_motion, control_mode`.
- Saved LeRobot action order: `eef_position, eef_rotation_quat_xyzw, gripper_close, base_motion, control_mode`.
- Saved image keys are `observation.images.left_image`, `observation.images.right_image`,
  and `observation.images.wrist_image`, matching `RoboCasa365Env.get_observation`.
- Task text is stored through the LeRobot episode task field; no extra
  `annotation.human.*` features are written.
- Saved `observation.state` follows `RoboCasa365Env.get_observation`:
  `eef_position_relative, eef_rotation_relative_quat, gripper_qpos, base_position, base_rotation_quat`.
- When replay reaches `done`, the cache appends the terminal observation with a no-op
  action whose quaternion is xyzw identity `[0, 0, 0, 1]`, keeping image, point cloud,
  action, reward, done, and state lengths identical.
- Replay point clouds are fused from left, right, and wrist cameras, transformed to
  robot base frame, cropped to the workspace, voxelized, and saved per episode as
  one `(sum_N, 6)` float32 `xyzrgb` `.npy`, with frame offsets in the `.npz`.
- `convert.py` writes those replay point clouds into the final `points_3views/` LMDB.


## Troubleshooting

Some robocasa tasks contain transparent objects, such as the bottle in `PickPlaceCounterToCabinet`. The depth is not accurate.
