# RoboCasa365 Benchmark

Training and evaluation of PointAct (frozen Qwen2.5-VL + trainable point-action expert) on
RoboCasa365 tasks, starting with **OpenDrawer**. Scaffolded from `experiments/2_libero`.

> Status: the training path mirrors Libero and should work once the state/action stats exist.
> The evaluation client (`run_robocasa365_client.py`) is a scaffold — its action/state
> plumbing for the 13-D PandaOmron action space carries `TODO(verify)` markers that must be
> checked against a real checkpoint before the success numbers are trustworthy.

## Prerequisites

1. **Dataset** — produced by `data_prep/robocasa365_to_lerobot` (download → replay → convert):
   `$SCRATCH/datasets/robot_data/robocasa365/lerobot_point_lmdb/OpenDrawer` (514 episodes,
   `points_3views` LMDB). See `envs/robocasa365/README.md`.

2. **`robot_data` symlink** — the data config uses a repo-root-relative path (Libero
   convention). Create it once:
   ```bash
   ln -s $SCRATCH/datasets/robot_data robot_data
   ```

3. **Pretrained backbones** — the train scripts point at:
   - VLM: `$DSDIR/HuggingFace_Models/Qwen/Qwen2.5-VL-3B-Instruct` (IDRIS shared models).
   - PTv3: `$SCRATCH/models/Pointcept-Concerto/concerto_large.pth` (Concerto). The Utonia
     variant expects `$SCRATCH/models/Pointcept-Utonia/utonia.pth` — download it (see
     `INSTALLATION.md`) only if you use `train_pointact_utonia.sh`.

   Adjust these paths in the train scripts if your copies live elsewhere.

## State / action statistics

Training needs a normalization file (referenced by the data config). The eef rotation quat is
at index `[3:7]` in both state and action. Two flags differ from the Libero command and are
**required** here:

- `--point_cloud_dir points_3views` — mandatory whenever `--*_xyz_slice` is given (position
  stats are computed in the point-cloud frame, as for RLBench).
- `--replace_zero_std` — RoboCasa365 tasks like OpenDrawer are fixed-base, so the state's base
  pose (and the action's base-motion) dims are constant → zero std. PointAct's state
  normalization (`processor_base.py`, `(state - mean) / std`) has no zero-std guard, so without
  this flag those dims produce NaN and training diverges immediately.

```bash
python data_prep/prepare_robot_state_action_stats.py \
    --dataset_dirs robot_data/robocasa365/lerobot_point_lmdb/OpenDrawer \
    --output_file  robot_data/robocasa365/lerobot_point_lmdb/OpenDrawer/robot_state_action_stats/rot6d.json \
    --point_cloud_dir points_3views \
    --state_xyz_slice 0 3 --action_xyz_slice 0 3 \
    --state_rotation_slice 3 7 --action_rotation_slice 3 7 \
    --rotation_type quat --target_rotation_type rot6d \
    --replace_zero_std
```

Note: only the eef rotation quat at `[3:7]` is converted to rot6d; the base rotation quat at
`[12:16]` passes through unchanged. Fine for fixed-base tasks like OpenDrawer — revisit for
navigation tasks where the base actually turns.

## Training

Same architecture as Libero (frozen vision tower + LLM + merger; trainable PTv3 point-action
expert). Effective batch size 128 on 1–2 H100. RoboCasa365 training has no simulator
dependency, so it runs in the `pointact` (root) env and can use H100.

```bash
# PTv3 = Concerto
bash experiments/13_robocasa365/train_pointact_concerto.sh
# PTv3 = Utonia
bash experiments/13_robocasa365/train_pointact_utonia.sh
```

Both take an optional data-config path as `$1` (default: the OpenDrawer config). Outputs land
in `$SCRATCH/datasets/PointAct_exprs/robocasa365/pointact/...`.

The 13-D PandaOmron action becomes 15-D after quat→rot6d, well under `max_action_dim=32`, so
the model architecture is unchanged from Libero — only the data differs.

## Evaluation

Server (pointact env, model) + client (robocasa365 env, MuJoCo/EGL) on the **same V100** GPU
(RoboCasa365 does not run on H100):

```bash
srun -A rgx@v100 -C v100-32g --gres=gpu:1 --cpus-per-task=10 --hint=nomultithread \
     --qos=qos_gpu-dev --time=02:00:00 --pty \
     experiments/13_robocasa365/eval_robocasa365.sh \
     OpenDrawer <ckpt_dir> <ckpt_step> rot6d " --args.save_video --args.verbose"
```

## Open items (what a scaffold cannot decide for you)

- **`is_delta_action`** in the data config — RoboCasa365 eef actions absolute vs delta. Set to
  match the source action definition and keep the client consistent.
- **VLM camera view** — the config feeds `left` (agentview_left) to the VLM. RoboCasa365 has
  two external views (left/right) plus wrist; revisit if both externals should go to the VLM.
- **State base-rotation normalization** — see the stats note above.
- **Client action reconstruction** (`reconstruct_env_action`) and **gripper handling** — the
  `TODO(verify)` blocks in `run_robocasa365_client.py`. Libero's last-dim gripper remap does
  not transfer (RoboCasa365's last dim is `control_mode`; `gripper_close` is mid-vector).
- **Per-trial scene seeding** — how `RoboCasa365Env.reset()` should vary/reproduce scenes
  across eval trials.
- **Success filtering** — the converted dataset keeps all 514 episodes; 18 replays did not
  reach success. Decide whether to train on successes only.
