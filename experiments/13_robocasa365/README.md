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

3. **Pretrained backbones** — Qwen2.5-VL-3B and the PTv3 checkpoints (Concerto / Utonia), per
   `INSTALLATION.md`.

## State / action statistics

Training needs a normalization file (referenced by the data config). Generate it like Libero;
the eef rotation quat is at index `[3:7]` in both state and action:

```bash
python data_prep/prepare_robot_state_action_stats.py \
    --dataset_dirs robot_data/robocasa365/lerobot_point_lmdb/OpenDrawer \
    --output_file  robot_data/robocasa365/lerobot_point_lmdb/OpenDrawer/robot_state_action_stats/rot6d.json \
    --state_rotation_slice 3 7 \
    --action_rotation_slice 3 7 \
    --rotation_type quat \
    --target_rotation_type rot6d
```

TODO(verify): the RoboCasa365 state also carries a **base** rotation quat at `[12:16]` that a
single `--state_rotation_slice` does not convert. Confirm this matches how you want the state
normalized (Libero has a single, fixed-base arm and no base rotation).

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
