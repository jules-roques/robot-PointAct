# Jean Zay (IDRIS)

Notes accumulated while building the RoboCasa365 / ROI-sampling pipeline. Verified as of
2026-07-27.

## Accounts and partitions

| Target | Flags | Status |
|---|---|---|
| V100 | `--account=rgx@v100` | Default. Works. |
| H100 | `--account=rgx@h100 --partition=gpu_p6 --constraint=h100`, plus `module load arch/h100` | Works. |
| A100 | `--account=rgx@a100 --partition=gpu_p5 --qos=qos_gpu_a100-t3` | Works. |

**The A100 allocation exists** (corrected 2026-08-01). `sacctmgr -n show assoc user=$USER`
lists `rgx@a100` with `qos_gpu_a100-dev` and `qos_gpu_a100-t3`:

```
rgx@a100    qos_gpu_a100-dev,qos_gpu_a100-t3
rgx@cpu     qos_cpu-dev,qos_cpu-t3,qos_cpu-t4
rgx@h100    qos_gpu_h100-dev,qos_gpu_h100-t3,qos_gpu_h100-t4
rgx@v100    qos_gpu-dev,qos_gpu-t3,qos_gpu-t4
```

Note there is **no `t4` QoS on a100**, unlike h100 — asking for one is the likeliest cause
of the earlier `Invalid account/partition` rejection recorded here. Always pass an explicit
`--qos` from the list above.

## Which GPU for which job

The constraints pull in opposite directions, so this is worth stating explicitly:

- **FlashAttention needs Ampere or newer.** Both the Qwen VLM and the PTv3 point backbone
  use it, so anything running the full model **fails on V100**.
- **GPU generation does not constrain the simulator.** RoboCasa365 rollouts produce
  identical output on H100 and V100 (measured on CLEPS 2026-08-17, see
  "RoboCasa365 across GPU generations" in `cleps.md`).
- Pure-torch inference (e.g. the MolmoPoint anchor-cache pass) has no simulator
  dependency and runs happily on H100.

So rollout-plus-model workloads (i.e. policy evaluation) need Ampere+, which here means
A100 (`rgx@a100 --partition=gpu_p5 --qos=qos_gpu_a100-t3`) or H100 — pick on availability
rather than on a sim constraint. Jean Zay and CLEPS both contribute eval capacity.

## Submit from a login shell, not `ssh host 'sbatch ...'`

The slurm scripts here call `module load arch/h100` and read `$DSDIR`. Both come from the
IDRIS **login** profile, and `sbatch --export=ALL` propagates them only if the submitting
shell had them. `ssh jean-zay 'sbatch ...'` is non-interactive and does not, so the job dies
in one second — `module: command not found` (exit 127), then `DSDIR: unbound variable` under
`set -u`, one line at a time. Submit like this instead:

```bash
ssh jean-zay 'bash -lc "cd <repo> && module load arch/h100 && sbatch --constraint=h100 ..."'
```

Chained jobs only need this for the first slice: successors are submitted from inside a job
that already has the environment.

Two dead ends, so nobody retries them:

- **`/etc/profile.d/modules.sh` is a stub** — its entire content is *"Replacement file to
  avoid loading environment modules installed at system level."* Sourcing it defines `module`
  against an empty `MODULEPATH`, so `module` appears to work and every load fails with
  "Unable to locate a modulefile".
- **`/etc/profile.d/z_modules.sh` is the real one, but exists on the login nodes only.** The
  compute nodes do not have it. Verifying a fix on the login node is exactly where this
  difference is invisible.

To make a script self-sufficient anyway, bootstrap from the shared lustre install that
`z_modules.sh` itself points at (guarded, so a module-loaded submission is unaffected):

```bash
if ! type module >/dev/null 2>&1; then
    . /lustre/fshomisc/sup/spack_soft/environment-modules/current/init/bash
    export MODULEPATH=/lustre/fshomisc/sup/hpe/pub/module-rh/modulefiles:/lustre/fshomisc/sup/hpe/pub/modules-idris-env4/modulefiles/linux-rhel9-skylake_avx512
fi
: "${DSDIR:=/lustre/fsmisc/dataset}"
```

`$WORK` and `$SCRATCH` survive a bare ssh (they come from `~/.bashrc`), which makes the
problem look narrower than it is — the script gets several lines in before failing.

**What an unset `$DSDIR` cost, concretely** (2026-08-24). `train.sh` reads the VLM from
`$DSDIR/HuggingFace_Models/Qwen/Qwen2.5-VL-3B-Instruct` and falls back to
`$SCRATCH/models/Qwen2.5-VL-3B-Instruct` when `$DSDIR` is empty. `--export=ALL` propagates the
emptiness faithfully, the fallback had since been eaten by the 30-day `$SCRATCH` purge, and
the job died ~40 s in on

```
OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/lustre/.../models/Qwen2.5-VL-3B-Instruct'
```

which reads as a bad Hub identifier rather than as a directory that is not there — transformers
treats a non-existent path as a repo name. Two arms were lost to it. `train_jeanzay.slurm` and
`train_jeanzay_dev.slurm` now default `$DSDIR` themselves, so the submission path no longer
decides which weights a run gets. Prefer `$DSDIR` over a private `$SCRATCH` copy for anything
IDRIS already mirrors: it is read-only, shared, and exempt from the purge.

## Compute nodes have no internet

Login nodes have network access; compute nodes do not. Anything that would download at
runtime must be baked beforehand on a login node into `$SCRATCH/models/...`.

MolmoPoint-8B lives at `$SCRATCH/models/MolmoPoint-8B`; fetch it with
`hf download allenai/MolmoPoint-8B --local-dir ...` from a login node. Its modelling code
is loaded with `trust_remote_code=True` from that directory, so nothing is fetched at
run time — but only if the download completed, which `du -sh` on the directory confirms
faster than a failed job does.

## $WORK is out of inodes, not out of space

`idr_quota_project` reports the number that matters: **~96% of 500,000 inodes used, against
7% of the 5 TiB**. A single torch install is roughly 40,000 files, so building any new uv
environment inside the repo fails with `Disk quota exceeded (os error 122)` mid-extraction
while `df` still shows terabytes free.

Put both the cache and the environment on `$SCRATCH`:

```bash
export UV_CACHE_DIR=$SCRATCH/.cache/uv
export UV_PROJECT_ENVIRONMENT=$SCRATCH/venvs/<name>
uv sync --project envs/<name>
```

`envs/molmo` is built this way; its jobs reference `$SCRATCH/venvs/molmo/bin/python`
directly rather than going through `uv run`, which would look for a `.venv` in the project
directory. Pin heavyweight dependencies (torch) to the root env's exact version too — left
to float, uv resolves the newest CUDA-13 build, which is both untested here and larger.

## A fresh worktree has no `robot_data`, and eval needs it

Dataset roots in a run's `data_config.yaml` are repo-relative
(`robot_data/robocasa365/lerobot_point_lmdb`), and resolve through a `robot_data` symlink at
the top of the checkout:

```
$WORK/code/robot-PointAct/robot_data -> $SCRATCH/datasets/robot_data
```

That symlink is gitignored, so `git worktree add` does not reproduce it. Any eval run from a
fresh worktree — `eval_grid_jeanzay.slurm`, `eval_seeds_jeanzay.slurm`,
`packing_probe_jeanzay.slurm` — resolves the text-context cache to a path that does not exist
and dies in `torch.load` with a bare `FileNotFoundError`, minutes into the job and well after
the checkpoint has loaded. Create it once per worktree:

```bash
ln -s $SCRATCH/datasets/robot_data <worktree>/robot_data
```

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
