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
   - VLM: `$SCRATCH/models/Qwen2.5-VL-3B-Instruct`.
   - PTv3: `$SCRATCH/models/Pointcept-Concerto/concerto_large.pth` (Concerto). The Utonia
     variant expects `$SCRATCH/models/Pointcept-Utonia/utonia.pth` — download it (see
     `INSTALLATION.md`) only if you use `train_pointact_utonia.sh`.

   Adjust these paths in the train scripts if your copies live elsewhere. (On Jean Zay the VLM
   path was `$DSDIR/HuggingFace_Models/Qwen/Qwen2.5-VL-3B-Instruct`; CLEPS has no `$DSDIR`.)

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

## Point-count / task ablation (the current grid)

Three tasks x point counts x sampling arms, trained **without the VLM**: the context the
point-action expert cross-attends to is a cached text-only embedding per instruction rather
than a live Qwen forward. With `--ptv3_apply_point_ca False` (every run here) that context
was the VLM's only contribution, so dropping the 3B forward and the images leaves the point
branch untouched while cutting the grid from ~2,160 to ~380 H100-hours. Language
conditioning survives, which OpenDrawer needs — its instruction carries left/right and the
target drawer is resampled per episode.

**One yaml per run** (`runs/`), holding the ablation coordinates, the data config and the
training args together; `runs/_base.yaml` carries everything the arms share.

```bash
# 0. Build the text-context cache once per task (needs the Qwen weights, not a GPU-heavy job)
python data_prep/cache_text_context.py \
    --dataset-dir $SCRATCH/datasets/robot_data/robocasa365/lerobot_point_lmdb/OpenDrawer \
    --vlm-path $SCRATCH/models/Qwen2.5-VL-3B-Instruct

# 1. One run
sbatch --export=ALL,RUN_CONFIG=experiments/13_robocasa365/runs/od-eef-n4096-s0.yaml \
       experiments/13_robocasa365/train.slurm

# 2. Or the whole of stage A (9 runs + their eval arrays + the gate)
bash experiments/13_robocasa365/submit_stage_a.sh

# 3. Evaluate one run at 20/30/40/50K (array; skips checkpoints not yet written)
sbatch --export=ALL,RUN=od-eef-n4096-s0 experiments/13_robocasa365/eval_grid.slurm

# 4. Push its results into W&B as a success-vs-checkpoint curve
python experiments/13_robocasa365/log_eval_to_wandb.py \
    --run-dir $SCRATCH/PointAct_exprs/robocasa365/ablation/od-eef-n4096-s0
```

Stage B (the two new tasks) **does not auto-launch**. When stage A finishes, a gate job mails
the point-count x sampling table; pick a point count, then
`python experiments/13_robocasa365/runs/generate_stage_b.py --npoints <N>`.

### Steps, not epochs — and what the datasets actually measure

Budget is denominated in **gradient steps**. Measured after conversion (2026-08-01):

| task | episodes | frames | frames/ep | steps/epoch @128 | epochs at 50K |
|---|---|---|---|---|---|
| OpenDrawer | 496 | 124,800 | 252 | 975 | 51.3 |
| PickPlaceCounterToStove | 501 | 122,274 | 244 | 955 | 52.3 |
| TurnOnMicrowave | 543 | 72,335 | 133 | 565 | **88.5** |

Note the planning estimate was wrong for TurnOnMicrowave. `docs/atomic_tasks/
atomic_episode_lengths.js` gives it `mean_seconds: 23`, but the target split averages 133
frames = 6.65 s at 20 fps — the doc figures are pretrain-split and run up to ~3.5x high. **Use
converted frame counts, not the docs, for any steps-per-epoch reasoning.**

So the three tasks are not equally exposed: two get ~52 epochs, TurnOnMicrowave gets 88.5, a
1.7x spread. Fixing steps remains right — it is compute-matched and it is what makes one
checkpoint grid (20/30/40/50K) comparable across tasks — and 1.7x is well inside the ~3-4x
threshold at which the dataset itself should be equalised instead. But **TurnOnMicrowave is
the over-exposure risk**: watch for its duration curve peaking before 50K and declining. That
is a finding to report, not a bug to patch.

### Measured throughput (pilot, 2026-08-01)

`pilot_throughput.py` on 4x H100 (gpu_p6), effective batch 128, marginal rate over a 40->160
step window so startup and CUDA warm-up are excluded. Projected to the 50K-step budget:

| points | cached-context s/step | live-VLM s/step | speedup | cached GPU-h | VLM GPU-h |
|---|---|---|---|---|---|
| 2048 | 0.259 | 1.346 | **5.2x** | 14.4 | 74.8 |
| 4096 | 0.469 | 1.537 | **3.3x** | 26.1 | 85.4 |
| 8192 | 0.550 | 1.820 | **3.3x** | 30.6 | 101.1 |

Two things this settles:

- **Dropping the VLM buys 3.3x** at the point counts that matter, so the plan's assumed 3x was
  about right (and 5.2x at 2048, where the point branch is cheapest and the VLM dominates most).
- **Point-count scaling is strongly sublinear: 0.55x / 1.00x / 1.17x**, not the 0.5x / 1x / 2x
  assumed. Doubling 4096 -> 8192 costs only 17% more, because fixed per-step costs dominate
  per-point compute at these sizes. If 8192 wins on success rate it is nearly free to adopt --
  the opposite of what the budget implied.

Resulting budget: **stage A 213 H100-h** (plan: 272), stage B **104 h at 4096** or **122 h at
8192**, so the whole grid lands at **317-335 H100-h** against the 380 planned. For reference,
the naive 27-run grid with a live VLM would be ~2,350 H100-h.

Sanity check on the whole exercise: live-VLM at 4096 measures 21.4 h for 50K steps on 4 GPUs,
which is the ~20 h/run figure the budget was originally built on.

### W&B conventions

Run names are short (`od-eef-n4096-s0`); identity lives in config columns. Group the runs
table by `exp_task` > `exp_sampling` > `exp_npoints` to get the grid as nested rows, and save
one workspace view per figure. `WANDB_RUN_ID` is pinned from the output dir so a requeued job
resumes one run instead of creating a second. Training runs are `job_type=train`, eval runs
`job_type=eval`, and both share a `group` per arm.

## Training (pre-ablation baseline runs)

Same architecture as Libero (frozen vision tower + LLM + merger; trainable PTv3 point-action
expert). Effective batch size 128 on 1–2 H100. RoboCasa365 training has no simulator
dependency, so it runs in the `pointact` (root) env and can use H100.

```bash
# PTv3 = Concerto
bash experiments/13_robocasa365/train_pointact_concerto.sh
# PTv3 = Utonia
bash experiments/13_robocasa365/train_pointact_utonia.sh
# PTv3 = Concerto, EEF-density point sampling ablation (see below)
bash experiments/13_robocasa365/train_pointact_concerto_eefdensity.sh
```

On CLEPS, submit via `sbatch experiments/13_robocasa365/train.slurm [concerto|concerto_eefdensity]`
(`--account=willow --partition=gpu --gres=gpu:h100:4`; see `train.slurm` for details — CLEPS has
no `module load cuda`/`ffmpeg`, so `LD_LIBRARY_PATH` points at a dedicated
`conda create -n ffmpeg-libs -c conda-forge ffmpeg=6.1` env instead). On Jean Zay, use
`train_jeanzay.slurm` instead — same payload scripts, but Jean Zay and CLEPS are separate SLURM
controllers so the `#SBATCH` account/partition/module directives can't be shared between them.
Check GPU availability/queue depth on both clusters (`squeue`) before deciding where to submit.

Both take an optional data-config path as `$1` (default: the OpenDrawer config). Outputs land
in `$SCRATCH/PointAct_exprs/robocasa365/pointact/...` (see "Storing results" below).

The 13-D PandaOmron action becomes 15-D after quat→rot6d, well under `max_action_dim=32`, so
the model architecture is unchanged from Libero — only the data differs.

### EEF-density point sampling (ablation)

Simpler alternative to the (parked) ROI-guided detector pipeline: instead of a uniform random
subsample, points are drawn with probability proportional to
`floor + (1 - floor) * exp(-d^2 / (2*sigma^2))`, `d` = distance to the frame's end-effector
position. No preprocessing/cache needed — the anchor is `observation.state[:3]`, already in the
point-cloud frame every frame. See `pointact/roi_sampling/geometry.py:eef_density_weights` and
`pointact/data/schema.py` (`eef_sampling`, `eef_sampling_sigma`, `eef_sampling_floor`). Config:
`data_configs/data-robocasa365-opendrawer-point-eefdensity.yaml` (`sigma=0.08`, `floor=0.05`,
both easy to sweep — no rebuild required, unlike the ROI halo cache).

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
