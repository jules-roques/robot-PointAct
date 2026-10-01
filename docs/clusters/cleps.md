# CLEPS (Inria)

Primary development cluster as of 2026-07-27. Chosen over Jean Zay for day-to-day work
because dev jobs there can have internet access; Jean Zay is kept for large runs.

> **This page is a stub.** Nothing below the checklist has been verified on CLEPS yet.
> Do not copy SLURM flags from `jean-zay.md` — the account and partition names there are
> IDRIS-specific and will not work here. Fill each item in as you confirm it, and delete
> this banner once the page is real.

## To determine

- [ ] Account / partition / QoS flags for each GPU tier available.
- [ ] Which GPU generations are offered. The model needs **Ampere or newer** for
      FlashAttention; the simulator runs the same on any of them.
- [ ] Do compute nodes have internet? If yes, the model-weight pre-baking dance required
      on Jean Zay is unnecessary here.
- [ ] Scratch storage path, quota, and purge policy — and therefore whether
      `archive_run.sh` needs a CLEPS-specific destination.
- [ ] Where the RoboCasa365 datasets and LMDB point caches will live.
- [ ] Module system: are `module load` names the same? Is there a CUDA module to load at
      all, or does the uv-managed torch bring its own?
- [ ] Max walltime per partition, and whether requeue/checkpointing is needed for a full
      training run.

## Verified 2026-10-01 (placing stage 8/9 trainings)

- **Ampere+ GPU nodes:** `gpu015`/`gpu016`/`gpu017` = 4x H100 (128 cpus, feature `h100`),
  `gpu012` = 4x A100 (64 cpus, feature `a100`); `gpu013` (A100) was down. All are shared with
  other groups' partitions (`willow`, `astra`, `almanach`), so they are often fully allocated.
- **Submit with** `--account=willow --partition=gpu` (what `train.slurm` does). QoS `normal`
  caps a user at **8 GPUs** (two 4-GPU trainings at once); partition max walltime **2 days**.
  `train.slurm` pins `--gres=gpu:h100:4`; to accept either tier, override on the command line
  with `--gres=gpu:4 --constraint="h100|a100"` (plain `gpu:4` would also match pre-Ampere nodes).
- **`/scratch` has no per-user quota**, but the shared filesystem was **96% full (1.4 TB
  free)** -- check `df -h /scratch` before pulling hundreds of GB.
- **Jean Zay -> CLEPS** over `ssh jean-zay`: ~55 MB/s per stream, ~100 MB/s with four rsync
  streams in parallel. Run transfers as a `cpu_devel` batch job (the login node kills long
  transfers); compute nodes can ssh to Jean Zay.
- **Training needs no `--export` of model paths:** `train.sh` falls back to
  `$SCRATCH/models/Qwen2.5-VL-3B-Instruct`, and `_base.yaml` reads
  `$SCRATCH/models/Pointcept-Utonia/utonia.pth`. Both present.
- **The CLEPS OpenDrawer dataset predated the 2026-09-21 Jean Zay rebuild** (496 renumbered
  episodes vs 514). Every post-stage-7 artifact (frame cache, oracle targets, baselines) uses the
  514-episode numbering, so the Jean Zay copy replaced it; the old one is at
  `lerobot_point_lmdb/OpenDrawer.stale-496ep`.

## Environment setup

The three uv environments in `docs/envs.md` must be rebuilt here — they are not portable
from Jean Zay.

## RoboCasa365 across GPU generations

**The simulator produces identical output on H100 and V100** (measured 2026-08-17), so
GPU generation is not a constraint on sim work — pick on availability and cost.

The test replayed OpenDrawer episodes 0–2 at `--seed 7` on an H100 NVL (`gpu016`,
job 5189524) and on a V100 as control (`gpu001`, job 5189525), same `envs/robocasa365`
env and `MUJOCO_GL=egl`, then diffed the caches:

| Quantity | H100 vs V100 |
|---|---|
| `observation_state`, `action`, `next_reward` | bit-identical (max abs diff 0) |
| Episode lengths, success flags | identical (319/220/211, all success) |
| Point-cloud `xyz`, all 14.2M points | bit-identical (max abs diff 0) |
| Point-cloud `rgb` | ≤ 1/255 on a few points |
| Camera images | ≤ 1 grey level on ≤ 0.012% of pixels |

The residual is 8-bit rasterisation rounding, not a simulation difference. H100 was also
**24% faster** (2m33s vs 3m22s for the three episodes).

One thing that looks alarming and is not: MuJoCo's EGL context throws
`OpenGL.raw.EGL._errors.EGLError` from `Renderer.__del__` at interpreter shutdown. It is
noisy but harmless and appears on every GPU type — a PyOpenGL teardown quirk, not a
rendering failure.

Scope: the sim/rendering path (replay and, by the same code, rollouts), on CLEPS with the
current env.

## RoboCasa365 across GPU generations

**The simulator produces identical output on H100 and V100** (measured 2026-08-17), so
GPU generation is not a constraint on sim work — pick on availability and cost.

The test replayed OpenDrawer episodes 0–2 at `--seed 7` on an H100 NVL (`gpu016`,
job 5189524) and on a V100 as control (`gpu001`, job 5189525), same `envs/robocasa365`
env and `MUJOCO_GL=egl`, then diffed the caches:

| Quantity | H100 vs V100 |
|---|---|
| `observation_state`, `action`, `next_reward` | bit-identical (max abs diff 0) |
| Episode lengths, success flags | identical (319/220/211, all success) |
| Point-cloud `xyz`, all 14.2M points | bit-identical (max abs diff 0) |
| Point-cloud `rgb` | ≤ 1/255 on a few points |
| Camera images | ≤ 1 grey level on ≤ 0.012% of pixels |

The residual is 8-bit rasterisation rounding, not a simulation difference. H100 was also
**24% faster** (2m33s vs 3m22s for the three episodes).

One thing that looks alarming and is not: MuJoCo's EGL context throws
`OpenGL.raw.EGL._errors.EGLError` from `Renderer.__del__` at interpreter shutdown. It is
noisy but harmless and appears on every GPU type — a PyOpenGL teardown quirk, not a
rendering failure.

Scope: the sim/rendering path (replay and, by the same code, rollouts), on CLEPS with the
current env.

## Migrating from Jean Zay

Code arrives via `git clone`. Datasets, LMDB caches and model weights do not — they live
on Jean Zay scratch and must be re-derived or copied deliberately.
