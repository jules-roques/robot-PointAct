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

# 2. Or the whole of stage 1 (9 runs + their eval arrays + the gate)
bash experiments/13_robocasa365/submit_stage_a.sh

# 3. Evaluate one run at 10/20/30/40/50K (array; skips checkpoints not yet written)
sbatch --export=ALL,RUN=od-eef-n4096-s0 experiments/13_robocasa365/eval_grid.slurm

# 4. Push its results into W&B as a success-vs-checkpoint curve
python experiments/13_robocasa365/log_eval_to_wandb.py \
    --run-dir $SCRATCH/PointAct_exprs/robocasa365/ablation/od-eef-n4096-s0
```

Stage 2 (the two new tasks) **does not auto-launch**. When stage 1 finishes, a gate job mails
the point-count x sampling table; pick a point count, then
`python experiments/13_robocasa365/runs/generate_stage_b.py --npoints <N>`.

### The three stages

`meta.stage` in each run yaml becomes a W&B tag and the `exp_stage` config column, so the runs
table filters straight to one stage.

| stage | tag | runs | what it asks |
|---|---|---|---|
| 0 | `Stage 0: Two camera views` | `od-{uniform,eef,anchor}-n4096-vlm-s0` | does image conditioning help, at matched budget? |
| 1 | `Stage 1: Num points & Train steps` | the nine `od-*-n{2048,4096,8192}-s0` | how many points, and how long to train? |
| 2 | `Stage 2: Task transfer` | `ppcs-*`, `tom-*` | does the sampling result hold on other tasks? |
| 3 | `Stage 3: MolmoPoint anchor` | `{od,tom}-molmo-*`, `ppcs-molmo-{obj,objpan}-*` | how much of the uniform→oracle gap does a frozen pointing model recover, with no privileged information? |
| 5 | `Stage 5: Five tasks x four samplers` | the twenty `s5-*` | does the sampling result hold across five tasks at one budget? |
| 6 | `Stage 6: Warp space` | `s6-od-nosampler-s0` | what does the whole cloud, with no input sampler at all, score? |
| 7 | `Stage 7: Point budget x sampler (Utonia)` | the eighteen `s7-od-*` | **the point-budget axis, re-measured on an encoder whose pretraining granularity matches ours** |

**Stage 3 evals are not interchangeable with the other arms'.** The anchor has to be produced
live by a MolmoPoint server running beside the policy, so a stage-3 eval needs roughly twice
the walltime — submit `eval_grid_jeanzay.slurm` with `--time=05:00:00`. `eval_robocasa365.sh`
starts the pointer itself; see `data_prep/roi_sampling/README.md`. The first round of these
numbers was thrown away because eval had no molmo branch and silently sampled uniformly, so
check `molmo_stats.frame_cover` in the per-trial json before reading any stage-3 rate.

**Stage 3 answer, at 50K (100 trials, seed 7).** OpenDrawer 71.0%, against 44.0% uniform, 69.3%
eef and 77.3% for the GT-handle oracle: +27.0 pp on uniform (p<1e-4) and statistically
indistinguishable from *both* eef (p=0.78) and the oracle (p=0.26). It does not transfer —
TurnOnMicrowave lands on 59.0%, exactly its uniform baseline, and PickPlaceCounterToStove
reaches 32.0% (object) / 26.0% (object + pan) against 4% uniform but 51% eef. The ordering is
the detector's, not the sampler's: `eval_molmo_accuracy.py` measures the point landing within
one sigma of the GT target 63% of the time on OpenDrawer and 33% on the microwave, whose start
button sits among near-identical neighbours. Adding the destination as a second Gaussian centre
does not pay for splitting the budget (26% vs 32%, p=0.35).

**Stage 0** is the with-VLM control, and the only reason it is a full retrain rather than a
comparison against the old 20-epoch runs: those stopped at 19,500 steps under a cosine schedule
annealed to *that* horizon, so their checkpoints are not a prefix of a 50K run and cannot be
read as a point on its duration curve. At 4096 points the image-free arms beat them, and stage
0 asks whether images simply converge more slowly. Everything except `context_source` matches
`_base.yaml` — see `runs/_base_2views.yaml` for the two settings that cannot match
(`gradient_checkpointing`, which needs a VLM to exist, and `image_aug`, which needs images).

Stage 0 costs ~3x stage 1 per step (1.54 vs 0.51 s/step at 4096), so ~21.4 h per run — past the
20 h `qos_gpu_h100-t3` cap. Submit them under `t4`:

```bash
RUNS="od-uniform-n4096-vlm-s0 od-eef-n4096-vlm-s0 od-anchor-n4096-vlm-s0" \
SUBMIT_GATE=0 TRAIN_EXTRA="--qos=qos_gpu_h100-t4 --time=30:00:00 --constraint=h100" \
bash experiments/13_robocasa365/submit_stage_a.sh
```

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

### Stage 7 — the point-budget axis, on Utonia

**Why it is being re-run.** Stage 1 answered "how many points does a point policy need?" with
PTv3 initialised from **Concerto**, which pretrains on a 2 cm grid. This pipeline voxelises at
1 cm and the encoder hardcodes `grid_size=0.01`, so every stage-1 number sits on a
factor-of-two mismatch between the granularity the backbone was pretrained at and the one it
is fed. **Utonia** pretrains at `grid_size=0.01` — its per-domain `RandomScale` exists
precisely to move every dataset onto that one canonical unit — which is the unit we feed.
`runs/_base.yaml` has set `ptv3_backend: utonia` since 2026-08-17 (commit 2b70004); stage 1
predates it. Stage 7 is that axis, measured again on the right encoder.

**The grid.** OpenDrawer × {uniform, eef, oracle} × {512, 1024, 2048, 4096, 8192, 16384} = 18
arms, 50K steps, checkpoints every 5K. Generated by `runs/generate_stage7.py`, submitted by
`submit_stage7.sh`. One task on purpose: stage 5 showed the sampler ordering is task-dependent
(eef wins on four tasks and loses 8.4 pp on CoffeeSetupMug), so a curve pooled across tasks
would average two different shapes.

**The budget list is weighted low, deliberately.** At 1 cm the whole OpenDrawer cloud is only
~18–22K points and the eef draw already takes 97% of everything within one σ of the gripper at
4096 (the voxel-probe table above), so 4096 → 8192 buys about eleven ROI points and the axis is
expected to be *flat* up there. 512 is where a uniform draw should visibly break while a
targeted one holds — the knee that makes this a curve rather than a line. At the other end,
16384 keeps ~73% of the median cloud, so the three samplers can differ only in which ~27% they
discard and **must** converge there by construction. That is the point of including it: it
brackets the axis between "sampling decides everything" and "sampling cannot matter".

**The no-sampler end is reused, not retrained.** `od-none-s0` (stage 6) is already at 50K on
this exact recipe — Utonia, cached text context, effective batch 128, the full 1 cm cloud —
with checkpoints every 2500, so every 5K step this stage evaluates exists. It differs only in
`per_device_train_batch_size` (16, which gradient accumulation absorbs at a held effective
batch of 128). It lives under `.../robocasa365/ablation`, not `.../stage7`, so its eval needs
its own submission with `EXPRS_DIR` pointing there — see `submit_stage7.sh`.

**Evaluation: 100 at every checkpoint, 500 at the end.** One seed × 100 trials at each of
5K…50K gives the duration curve; four more seeds at 50K pool with it to n=500 for the headline
budget-vs-success table. At n=100 the Wilson half-width is ~±10 pp near 50% and the stage-5
sampler gaps were 3–8 pp, so the headline comparison needs the pooling and the curve does not.
Never re-run a seed to raise n — the same seed replays the same episode stream; run a new one
and pool (`pool_eval_results.py`).

**Cost — measured 2026-09-08, job 1880350, and it is ~2x what the stage-1 table implies.**
~795 H100-h of training, plus ~17 node-hours of eval for ~26,600 rollouts at the measured
~1600 trials/h/node. Eval is a rounding error against training, which is why the trial counts
are generous.

Marginal s/step on 4×H100 at effective batch 128 (`pilot_throughput.py`, 40→200 step window),
and what each arm costs over 50K steps:

| budget | s/step | 50K wall | GPU-h |
|---|---|---|---|
| 1024 | 0.725 | 10.1 h | 40.3 |
| 2048 | 0.704 | 9.8 h | 39.1 |
| 4096 | 0.835 | 11.6 h | 46.4 |
| 8192 | 0.798 | 11.1 h | 44.3 |
| 16384 | 0.985 | 13.7 h | 54.7 |

Every arm fits the 20 h `qos_gpu_h100-t3` cap, so none needs t4. **The rate is ~2.7x the
stage-1 pilot's** (0.259/0.469/0.550 s/step at 2048/4096/8192) and the reason is the backbone,
not the point count: `_base.yaml` runs Utonia at `enc_depths [3,3,3,12,3]` — 24 blocks —
against the `[2,2,2,6,2]` (14 blocks) the pipeline defaults to. Utonia's *widths* are narrower
than Concerto's, so the cost is in the depth. **Do not budget a Utonia grid off a stage-1
number.** Note also that this measurement pins per-device batch at 16 with accumulation 2,
while the ≤8192 arms actually train at 32 with accumulation 1, which should run slightly
faster than the table.

The pilot column is **not monotonic** (2048 < 1024, 8192 < 4096). That is the differencing
method's noise, ~±0.05 s/step here, not a real inversion — read it for absolute wall-clock and
read the profiler below for the shape.

#### What a step is actually spent on (`profile_point_cost.py`, 1 GPU, batch 16, 30 reps)

| budget | points fed | dataload | forward | backward | step total | peak GPU |
|---|---|---|---|---|---|---|
| 1,024 | 1,024 | 1.67 ± 0.68 ms | — | — | 316.5 ± 8.1 ms | — |
| 2,048 | 2,048 | 1.61 ± 0.81 ms | 171.4 ± 3.6 ms | 178.7 ± 4.2 ms | 366.2 ± 6.5 ms | 8.4 GB |
| 4,096 | 4,096 | 1.95 ± 0.87 ms | 231.8 ± 20.4 ms | 228.1 ± 19.6 ms | 479.3 ± 35.0 ms | 10.0 GB |
| 8,192 | 8,190 | 1.99 ± 0.89 ms | 237.6 ± 23.4 ms | 253.3 ± 17.6 ms | 510.2 ± 39.6 ms | 11.9 GB |
| 16,384 | 16,058 | 2.23 ± 0.99 ms | 278.9 ± 52.4 ms | 310.2 ± 23.1 ms | 612.5 ± 55.9 ms | 14.5 GB |

Three things this settles, and they are the actual content of "point policies are expensive":

- **16x the points costs 1.94x the step.** Strongly sublinear, for the reason the voxel-probe
  table gives: only PTv3's first stages see more tokens, the deep stages are limited by
  occupied 1 cm *voxels* rather than by points, and attention is windowed at `patch_size`.
- **Data loading is 0.3–0.5% of a step and essentially flat** — 1.67 ms at 1024 against 2.23 ms
  at 16384, for 16x the points. The budget is drawn *after* the full cloud is read from LMDB
  and cropped, so the arms differ only in the tail of `__getitem__`. Whatever makes a point
  policy expensive, it is not the dataloader.
- **The expense is the network, split evenly between forward and backward**, with the optimiser
  a flat ~13–18 ms. Memory is not a constraint at these budgets: 14.5 GB of 80 at the top.

At 16384 the points actually fed are 16,058, not 16,384 — the `U(0.8, 1.0)` dropout binds
before the budget does, which is the same measurement that says 16384 is the convergence end
of the axis rather than a larger budget.

#### Cost profiling: `profile_point_cost.py`

The other half of what stage 7 reports is *how expensive* the points are, decomposed. Use
`profile_point_cost.py` for the breakdown — data loading, host-to-device, forward, backward,
optimiser — with a mean **and a standard deviation** over N timed iterations, each bracketed
by an explicit `torch.cuda.synchronize()`:

```bash
python experiments/13_robocasa365/profile_point_cost.py \
    experiments/13_robocasa365/runs/s7-od-uniform-n4096-s0.yaml \
    --npoints 1024 2048 4096 8192 16384 --reps 30 --out cost.json
```

It is the microscope; `pilot_throughput.py` is the clock. Quote **both**: pilot_throughput
reports the end-to-end marginal s/step a real 4-GPU `accelerate` run sustains (the number that
predicts wall-clock), while this reports where that time goes and how much it varies. Expect
the profiler's `step_total` to sit below pilot_throughput's s/step — it has no gradient
accumulation, no DDP all-reduce and no checkpoint writes. Ten repetitions is enough for a mean
and thin for a standard deviation, which is half of what is being reported; 30 is the default.

### W&B conventions

Run names are short (`od-eef-n4096-s0`); identity lives in config columns. Group the runs
table by `exp_task` > `exp_sampling` > `exp_npoints` to get the grid as nested rows, and save
one workspace view per figure. `WANDB_RUN_ID` is pinned from the output dir so a requeued job
resumes one run instead of creating a second. Training runs are `job_type=train`, eval runs
`job_type=eval`, and both share a `group` per arm.

## Training (pre-ablation baseline runs)

Same architecture as Libero (frozen vision tower + LLM + merger; trainable PTv3 point-action
expert). Effective batch size 128 on 1–2 H100. RoboCasa365 training has no simulator
dependency, so it runs in the `pointact` (root) env.

The three 20-epoch with-VLM runs that produced the first sampling ablation (uniform 30.7% /
eef 51.3% / oracle 66.7%) live in `runs/legacy/`, which reproduces the shell scripts they were
launched from:

```bash
sbatch --export=ALL,RUN_CONFIG=experiments/13_robocasa365/runs/legacy/concerto-uniform.yaml \
       experiments/13_robocasa365/train_jeanzay.slurm
```

**Their checkpoints have been deleted** (2026-08-02, 123 GB) — superseded by the stage-0 arms
above, which train the same three arms to 50K on the current recipe. Each run directory keeps
its `results/`, `trainer_state.json` and `training_args.json`, so the published numbers stay
auditable; only the weights are gone.

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

#### The 1 cm voxel grid, not the budget, is what limits these samplers

Measured 2026-08-03 on OpenDrawer episodes 0-4, 300 frames, by
`voxel_probe_jeanzay.slurm` + `voxel_probe_report.py`. Medians per frame:

| grid | cloud | ≤4 cm of eef | ≤8 cm | on the handle | ROI drawn @4096 | ROI saturation @4096 |
|---|---|---|---|---|---|---|
| 1 cm (current) | 18,214 | 9 | 146 | 71 | 141 | **97%** |
| 5 mm | 50,695 | 19 | 388 | 216 | 252 | 65% |
| 2 mm | 77,804 | 50 | 1,156 | 730 | 526 | 45% |

At 1 cm the eef draw already selects 97% of every point within 1σ of the gripper, and 100% at
8192 — so neither a larger point budget nor a sharper `sigma`/lower `floor` can concentrate
further. That, not the sampler, is why the point-count axis of stage 1 is flat: 4096 → 8192
buys about eleven ROI points.

The finer grids resolve **disproportionately more in the ROI than in the scene**: at 2 mm the
whole cloud grows 4.3× but the handle grows 10.3×. That is the wrist camera — it sits ~20 cm
from the gripper, where its pixel pitch is ~1 mm, so the 1 cm merge was discarding real
detail exactly where the samplers need it. The left/right cameras (256² over a ~1.5 m scene,
~6 mm pitch) are near their render ceiling at 1 cm, which is why the *global* gain saturates.

Not a knob, though — see the caveats in `voxel_probe_jeanzay.slurm`. Voxel size is set
independently in `replay.py --voxel-size`, `processor_base.py` (eval, server-side) and
`ptv3_backbone.py`'s `grid_size`, and PTv3 truncates coords onto `grid_size` to build its
z-order codes and sparse indices, with `stride=(2,2,2,2)` putting the coarsest encoder level
at 16× the voxel (1 cm → 16 cm today, 5 mm → 8 cm, 2 mm → 3.2 cm). Going finer without
adding an encoder stage trades ROI resolution for receptive field. Storage scales with the
cloud: OpenDrawer's `points_3views` is 60 GB at 1 cm, ~170 GB at 5 mm, ~260 GB at 2 mm.
Re-replay is cheap (~1 h/task on the 8-way array) and training cost at a fixed `max_npoints`
is unchanged.

## Evaluation

Policy server (pointact env, model) + sim client (robocasa365 env, MuJoCo/EGL) on the **same
A100** GPU, driven by `eval.slurm` (CLEPS) / `eval_jeanzay.slurm` (Jean Zay) — same payload
(`eval_robocasa365.sh`), different `#SBATCH` directives per cluster. Ampere+ (not V100) is
required because the model uses FlashAttention in both the Qwen VLM and the PTv3 backbone. A100
is the default out of availability, not necessity — H100 works equally well.

**100 trials per checkpoint, at 10/20/30/40/50K** — the `eval_grid*.slurm` default, and the
convention for everything from here on. The first sweep used 50 on the intermediate points and
150 at 50K; that turned out backwards, because the 10K column carried the most interesting
result (anchor already at 76% while uniform sat at 22%) and was the least precise. A single
trial count also means a duration curve does not change resolution halfway along. At n=100 the
Wilson half-width is ~±10 pp near 50%, ~±8 pp near 80%.

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

### One node-job per task, not one job per (checkpoint, seed)

`eval_grid_jeanzay.slurm` (array over checkpoints) and `eval_seeds_jeanzay.slurm` (array over
seeds) each run **one** eval pair per 1-GPU job, so a 5-checkpoint x 5-seed grid is 25 jobs
that queue independently for ~1 h of work each. Eval is simulator-bound rather than GPU-bound,
so that leaves almost a whole node idle per job. Measured on a full node
(`packing_probe_jeanzay.slurm`, 2026-09-02):

| K concurrent pairs | A100 node (8 GPU, 64 cores) | H100 node (4 GPU, 96 cores) |
|---|---|---|
| 1 | 28 s/trial, 1.00x | 29 s/trial, 1.00x |
| 4 | 30 s, 3.73x | 30 s, 3.87x |
| 8 | 31 s, 7.22x | 33 s, 7.03x |
| 16 | 36 s, **12.44x**, 1600 trials/h | 35 s, **13.26x**, 1646 trials/h |

Nothing was saturated at K=16 (11-23 GB of 80 per GPU, 14-36% utilisation); the ceiling is
above 16 and was never found. Node throughput is the same on both tiers because the bottleneck
is CPU/MuJoCo/EGL, so **pick the tier on queue depth, not throughput** — which is why the packed
harness defaults to `gpu_p6`. It also means `--cpus-per-task=8` per eval, as the array scripts
ask for, is over-provisioned 2-4x.

`eval_task_jeanzay.slurm` packs one task's whole grid into a single node-job:

```bash
# One task, every arm x every checkpoint x every seed, 16 pairs at a time:
sbatch --export=ALL,RUNS="od-uniform-n8192-s0 od-eef-n8192-s0" \
       experiments/13_robocasa365/eval_task_jeanzay.slurm

# Smoke it first (2 trials, 2 pairs, dev QoS) -- always do this for a new grid:
sbatch --qos=qos_gpu_h100-dev --time=00:30:00 \
       --export=ALL,RUNS=od-eef-n8192-s0,EVAL_STEPS=50000,EVAL_SEEDS=7,NUM_TRIALS=2,CONCURRENCY=1 \
       experiments/13_robocasa365/eval_task_jeanzay.slurm
```

**One job per task, enforced.** All `RUNS` must train on the same task or the job refuses to
start. Wall-clock is dominated by episode *length* (OpenDrawer ~1h00-1h12 per 100 trials,
TurnOnMicrowave ~42 min), so only a per-task job has a runtime anyone can size; a mixed job also
puts the whole campaign in one walltime's blast radius.

Packing does not mix up the outputs, because none of them are keyed by job: ports self-negotiate,
results stay content-addressed at `<ckpt_dir>/results/checkpoint-<step>/per_trial_seed<S>_n<N>.json`
(the glob `summarize_stage_a.py` and `log_eval_to_wandb.py` already pool over), and each pair's
stdout goes to its own file under `$SCRATCH/logs/robocasa365/evaltask-<jobid>/`.

What it has to replace, versus an array:

- **Per-element exit codes.** The end-of-job report judges each pair by whether its final JSON
  *exists*, not by its exit code — a pair can exit 0 having written nothing, which is exactly the
  failure an array's exit codes hide.
- **All-or-nothing walltime.** A pair that cannot finish in the time left is not started (a
  killed client's partial dump is deliberately named outside the pooling glob, so it would
  contribute nothing), and any pair that already has its final JSON is skipped. The job is
  therefore **idempotent**: resubmit the identical command to finish a grid that ran out of time.

MolmoPoint arms are rejected up front — the arm was dropped, and its third process is not sized
for here.

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

## Analysis and pre-flight helpers

Small scripts that belong to the campaign rather than to any one stage. The two pre-flight
checks exist because both failures they guard against actually happened, and both cost a
node-day before anyone noticed.

| Script | What it does |
|---|---|
| `pool_eval_results.py` | Pools `per_trial_seed*_n*.json` per arm under a root, with Wilson CIs, as JSON. Raises on a duplicate seed or a mixed `env_name` inside one arm. |
| `summarize_stage_a.py` | Renders the stage-A point-count × sampling table for the gate mail. Name-parsed, fixed task list — use `pool_eval_results.py` for anything else. |
| `verify_arm_derivation.sh` | **Pre-flight.** Replays the eval's arm-identity derivation over archived `data_config.yaml`s offline, so an arm about to be scored as the *wrong* arm is caught in a second. |
| `check_oracle_gt.py` | **Pre-flight.** Confirms each oracle arm's geom key exists in `target_positions.npz`. A missing key otherwise fails per-sample inside the dataloader pool, with a traceback naming the pickle machinery. |
| `submit_stage1_reeval.sh` | Re-scores existing checkpoints with a given seed set (no training), setting legacy result files aside so protocols cannot silently pool. |

Never re-run a seed an arm already has — the scene stream is seed-deterministic, so it replays
the same kitchens and buys no information. To raise `n`, add *new* seeds and pool.

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
