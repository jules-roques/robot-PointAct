# RoboCasa365 Benchmark

Training and evaluation of PointAct (frozen Qwen2.5-VL + trainable point-action expert) on
RoboCasa365 tasks, starting with **OpenDrawer**. Scaffolded from `experiments/2_libero`.

The full path is verified end to end: the concerto OpenDrawer checkpoint scores **60% (30/50)**
on the 50-trial evaluation. See [`docs/robocasa365.md`](../../docs/robocasa365.md) for the
end-to-end walkthrough from a fresh checkout; this file is the reference for the training and
evaluation stages specifically.

## Prerequisites

1. **Dataset** — produced by `data_prep/robocasa365_to_lerobot` (download → replay → convert):
   `$SCRATCH/datasets/robot_data/robocasa365/lerobot_point_lmdb/OpenDrawer` (496 converted
   episodes, `points_3views` LMDB). See `envs/robocasa365/README.md` for the download step and
   `data_prep/robocasa365_to_lerobot/README.md` for replay/convert.

2. **`robot_data` symlink** — the data config uses a repo-root-relative path (Libero
   convention). Create it once:
   ```bash
   ln -s $SCRATCH/datasets/robot_data robot_data
   ```

3. **Pretrained backbones** — the train script points at:
   - VLM: `$SCRATCH/models/Qwen2.5-VL-3B-Instruct`.
   - PTv3: `$SCRATCH/models/Pointcept-Concerto/concerto_large.pth` (Concerto).

   Adjust these paths in the train script if your copies live elsewhere. (On Jean Zay the VLM
   is picked up from `$DSDIR/HuggingFace_Models/Qwen/Qwen2.5-VL-3B-Instruct` automatically;
   CLEPS has no `$DSDIR`, so the `$SCRATCH` fallback applies.) See `INSTALLATION.md` for where
   to download them.

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
```

On CLEPS, submit via `sbatch experiments/13_robocasa365/train.slurm`
(`--account=willow --partition=gpu --gres=gpu:h100:4`; see `train.slurm` for details — CLEPS has
no `module load cuda`/`ffmpeg`, so `LD_LIBRARY_PATH` points at a dedicated
`conda create -n ffmpeg-libs -c conda-forge ffmpeg=6.1` env instead, via `$FFMPEG_LIB_DIR`). On
Jean Zay, use `train_jeanzay.slurm` — same payload script, but Jean Zay and CLEPS are separate
SLURM controllers so the `#SBATCH` account/partition/module directives can't be shared between
them. **The account/partition/qos in both files are specific to our projects; override them on
the `sbatch` command line for your own allocation.**

The train script takes an optional data-config path as `$1` (default: the OpenDrawer config).
Outputs land in `$SCRATCH/PointAct_exprs/robocasa365/pointact/...` (see "Storing results").

The 13-D PandaOmron action becomes 15-D after quat→rot6d, well under `max_action_dim=32`, so
the model architecture is unchanged from Libero — only the data differs.

## Evaluation

Policy server (pointact env, model) + sim client (robocasa365 env, MuJoCo/EGL) on the **same
A100** GPU, driven by `eval.slurm` (CLEPS) / `eval_jeanzay.slurm` (Jean Zay) — same payload
(`eval_robocasa365.sh`), different `#SBATCH` directives per cluster. A100 (not V100) is required
because the model uses FlashAttention in both the Qwen VLM and the PTv3 backbone (Ampere+ only);
H100 is avoided because RoboCasa365 does not run correctly there.

```bash
# Full 50-trial success rate (default checkpoint = the OpenDrawer concerto run):
sbatch --export=ALL,CKPT_STEP=final-48750 experiments/13_robocasa365/eval.slurm

# Smoke (3 trials + videos, short walltime):
sbatch --time=01:00:00 \
       --export=ALL,CKPT_STEP=1000,NUM_TRIALS=3,OPTS="--args.save_video --args.verbose" \
       experiments/13_robocasa365/eval.slurm
```

Override the checkpoint via `--export=ALL,CKPT_DIR=...,CKPT_STEP=...`. Results (per-trial log,
success rate, optional videos) are written under `<run_dir>/results/checkpoint-<step>/`.
Baseline: the concerto OpenDrawer checkpoint scores **~60% (30/50)** on this eval.

The client sends the model a **fused 3-view point cloud** (left+right+wrist, matching the
`points_3views` training data): the server otherwise builds the cloud from `select_video_keys`
alone (the single `left` VLM view), which starves the PTv3 backbone and collapses the policy.
The server already applies `pred_rot_type` and the absolute-position offset, so the client
steps the returned 13-D action directly.

## Storing results

Run outputs live on `$SCRATCH` (fast, large) but SCRATCH is **purged after ~30 days of no
access**, so anything worth keeping must be copied off it:

- **Training curves** — logged to Weights & Biases (`WANDB_MODE=offline` in the slurm scripts).
  `wandb sync $SCRATCH/wandb/wandb/offline-run-*` from the login node uploads them to the
  `diffusion4robots` cloud project (durable).
- **Final checkpoint + eval results + configs** — archive to `$HOME` (durable, not purged; CLEPS
  has no `$STORE`):
  ```bash
  bash experiments/13_robocasa365/archive_run.sh <run_dir>
  ```
  This tars the *final* checkpoint, the `results/` tree and the run config into
  `$HOME/archives/PointAct/robocasa365/<run>.tar`. `$HOME` has a 100GB space quota, so keep it
  to a few large tars, not loose files.
- **Intermediate checkpoints** stay on SCRATCH; they exist for resume and are regenerable, so
  let the purge reclaim them.

## Decisions (resolved)

- **`is_delta_action` = False (absolute eef)** — baked into the checkpoint
  (`is_action_eef: true`); the client and stats are consistent with it.
- **Client action / gripper plumbing** — resolved. The server's `_build_action_output` applies
  `pred_rot_type` (rot6d→quat) and re-adds the absolute-position offset, returning a 13-D
  env-ready action; the client steps it directly (no reconstruction, no Libero gripper remap —
  the env thresholds `gripper_close`/`control_mode` at 0.5 internally).
- **Point-cloud input** — the client fuses the 3 camera views into `observation.points`; see
  the Evaluation section for why a single-view cloud collapses the policy.
- **Success counting** — from the sim's `info["success"]`, not `done` (robosuite also sets
  `done` at the horizon timeout, which would inflate the rate).
- **Success filtering** — the training set was filtered to the 496 successful replays.
- **State base-rotation normalization** — zero-std dims guarded via `--replace_zero_std` when
  generating the stats (otherwise the base-quat dims divide by zero → NaN).

## Remaining

- **VLM camera view** — the config feeds `left` (agentview_left) to the VLM. RoboCasa365 has
  two external views (left/right) plus wrist; revisit if both externals should go to the VLM.
- **Per-trial scene seeding** — `RoboCasa365Env.reset()` re-randomises each trial from the
  env RNG (seeded once); revisit if you need reproducible per-trial scenes.
