# Jean Zay (IDRIS)

Notes accumulated while building the RoboCasa365 / ROI-sampling pipeline. Verified as of
2026-07-27.

## Accounts and partitions

| Target | Flags | Status |
|---|---|---|
| V100 | `--account=rgx@v100` | Default. Works. |
| H100 | `--account=rgx@h100 --partition=gpu_p6 --constraint=h100`, plus `module load arch/h100` | Works. |
| A100 | `--account=rgx@a100 --partition=gpu_p5` | **Rejected**: `IDRIS: Account rgx@a100 ----- Job type v100 / Invalid account/partition`. |

The A100 rejection is an allocation issue, not a syntax one — if a job needs A100
specifically, confirm the allocation before assuming a script is broken.

## Which GPU for which job

The constraints pull in opposite directions, so this is worth stating explicitly:

- **FlashAttention needs Ampere or newer.** Both the Qwen VLM and the PTv3 point backbone
  use it, so anything running the full model **fails on V100**.
- **RoboCasa365 simulation does not run correctly on H100.** Avoid H100 for anything doing
  MuJoCo rollouts.
- Pure-torch inference (e.g. the YOLO-World ROI detection pass) has no simulator
  dependency and runs happily on H100.

That leaves rollout-plus-model workloads (i.e. policy evaluation) wanting A100 — the one
tier whose account currently rejects jobs. Resolve the allocation before planning an
evaluation campaign here.

## Compute nodes have no internet

Login nodes have network access; compute nodes do not. Anything that would download at
runtime must be baked beforehand on a login node into `$SCRATCH/models/...`.

Specifically for YOLO-World: `set_classes` must be baked into the saved checkpoint, so
that no CLIP text-encoder download is attempted at inference time.

## Storage lifetime

`$SCRATCH` is purged after roughly 30 days without access. Training is the expensive,
hard-to-reproduce half, so final checkpoints (weights + config + normalization stats)
get tarred to `$STORE` by `experiments/13_robocasa365/archive_run.sh`. Evaluation
results are deliberately not archived — they are cheap to regenerate.

## Filesystems: inodes are the binding constraint, not bytes

Measured 2026-08-12 with `idr_quota_user`:

| | storage | inodes |
|---|---|---|
| `$HOME` | 3 GiB — tiny | 150k |
| `$WORK` | 5 TiB, ~7% used | **500k, and it fills long before the bytes do** |
| `$SCRATCH` | effectively unlimited, purged | no quota |
| `$STORE` | 50 TiB | 100k — tarballs only |

`$WORK` reached 98.9% of its inode quota at 7% of its storage quota. Anything that costs
*files* rather than *bytes* — package caches, unpacked wheels, asset trees, compiled
kernel caches — does not belong there. The rule that follows:

- **Regenerable and file-heavy → `$SCRATCH`.** `UV_CACHE_DIR` alone was 206,665 files
  (41% of the whole `$WORK` budget) for 37 G. Also `PIP_CACHE_DIR`, `TRITON_CACHE_DIR`,
  `WANDB_CACHE_DIR`, `HF_DATASETS_CACHE`.
- **Unrebuildable from a compute node → `$WORK`.** Model weights (`HF_HOME`, `TORCH_HOME`)
  are a handful of files for many gigabytes, and a compute node has no internet to
  refetch them. Same for `UV_PYTHON_INSTALL_DIR`: every venv's `pyvenv.cfg` points into
  it, so a purge there breaks every environment with a confusing error.
- **Per-job scratch space → `$JOBSCRATCH`**, node-local and deleted at job end. The right
  home for a compile cache in a multi-node run, at the price of a cold compile per job.

These live in `~/.config/jz_env.sh`, sourced from **both** `~/.bashrc` and `~/.zshrc`.
Putting them in `.zshrc` alone silently fails: a `#!/bin/bash` batch script reads no rc
file at all and inherits its environment from whatever shell ran `sbatch`, so anything
submitted over `ssh <host> '...'` got no cache variables and fell back to `$HOME/.cache`
— which then hit the 3 GiB quota and killed jobs with `Disk quota exceeded` from `uv`.

## Venvs: hardlinks, and how to move one

`uv` populates a venv by **hardlinking out of its cache when both sit on the same
filesystem** — 45k of the PointAct venv's 52k files had `nlink=2`. Quota counts inodes,
not directory entries, so a venv next to its cache is nearly free. Two consequences:

- Moving `UV_CACHE_DIR` alone, leaving the venv behind, is a pessimisation: uv falls back
  to copying and the venv grows its own tens of thousands of inodes. Move both or neither.
- Deleting a cache does not free inodes still hardlinked from a venv elsewhere. Retire the
  venv first, then the cache. Miss one and its files simply drop to `nlink=1` and stay
  charged — the space is not lost, but the deletion buys nothing until that venv moves too.

Find *every* venv before touching the cache. This repo has two, and the second one sits
four levels down, so the obvious `find -maxdepth 3 -name pyvenv.cfg` misses it:

```sh
find $WORK -name pyvenv.cfg -not -path '*/lib/*'
```

To relocate a venv, **copy it and leave a symlink behind** rather than rebuilding it:

```sh
cp -a $WORK/code/robot-PointAct/.venv $SCRATCH/venvs/robot-PointAct
rm -rf $WORK/code/robot-PointAct/.venv
ln -s $SCRATCH/venvs/robot-PointAct $WORK/code/robot-PointAct/.venv
```

The absolute shebangs baked into `bin/` still resolve through the symlink, and `uv run
--project` accepts a symlinked `.venv` and leaves it in place. Rebuilding with `uv sync`
instead is the risky option: the PointAct env carries `spconv-cu120` without the rest of
the `cuda` extra, so a clean sync does *not* reproduce it.

A venv on `$SCRATCH` is purgeable, so tar it to `$STORE` — one inode, and restoring it
needs no internet.

The same copy-and-symlink works for the conda base at `$WORK/miniforge3` (27,899 entries,
over half of it the `pkgs/` package cache). Note that its `envs/` is **empty on Jean Zay**:
the `$HOME/miniforge3/envs/ffmpeg-libs/lib` that `train.slurm`, `data_prep_*.slurm` and
`eval_robocasa365.sh` prepend to `LD_LIBRARY_PATH` is a CLEPS-ism (see `docs/clusters/cleps.md`
— no ffmpeg module there). A missing `LD_LIBRARY_PATH` entry is silently ignored, so those
lines are dead here rather than broken.

## Camera calibration (RoboCasa datasets)

Intrinsics and extrinsics are **not stored in the dataset**. Recover them from a
simulator env reset (V100, `MUJOCO_GL=egl`).

Two facts that save time:

- The `base -> cam` transform is **constant across episodes** for the robot-mounted
  agentview cameras, even though world-frame `base_pos` varies a lot. One global
  calibration is valid for every frame.
- Stored RGB and depth are both `[::-1]`-flipped consistently, so a bounding box on a
  stored frame shares row indexing with the reprojected points.
