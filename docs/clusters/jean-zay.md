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
- **RoboCasa365 simulation does not run correctly on H100.** Avoid H100 for anything doing
  MuJoCo rollouts.
- Pure-torch inference (e.g. the MolmoPoint anchor-cache pass) has no simulator
  dependency and runs happily on H100.

That leaves rollout-plus-model workloads (i.e. policy evaluation) wanting A100, which is
available here via `rgx@a100 --partition=gpu_p5 --qos=qos_gpu_a100-t3`. Jean Zay and CLEPS
therefore both contribute eval capacity; training stays on H100.

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

## Storage lifetime

`$SCRATCH` is purged after roughly 30 days without access. Training is the expensive,
hard-to-reproduce half, so final checkpoints (weights + config + normalization stats)
get tarred to `$STORE` by `experiments/13_robocasa365/archive_run.sh`. Evaluation
results are deliberately not archived — they are cheap to regenerate.

## Camera calibration (RoboCasa datasets)

Intrinsics and extrinsics are **not stored in the dataset**. Recover them from a
simulator env reset (V100, `MUJOCO_GL=egl`).

Two facts that save time:

- The `base -> cam` transform is **constant across episodes** for the robot-mounted
  agentview cameras, even though world-frame `base_pos` varies a lot. One global
  calibration is valid for every frame.
- Stored RGB and depth are both `[::-1]`-flipped consistently, so a bounding box on a
  stored frame shares row indexing with the reprojected points.
