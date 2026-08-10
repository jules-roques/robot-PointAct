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

### The `-dev` QoS caps wall clock at 2h, and exceeding it fails silently

`*-dev` is the short-turnaround QoS: quick scheduling, but **2 hours maximum**. Ask for more
and `sbatch` **accepts the job and prints a job id** — then it sits `PENDING` forever with:

```
$ squeue -u $USER -o "%.10i %.9T %.10l %.30R"
    812656   PENDING    3:00:00   (QOSMaxWallDurationPerJobLimit)
```

There is no error, no mail, and no failed state. In a busy queue this is indistinguishable
from waiting for a node, so a job can be presumed-queued for hours before anyone notices it
was never schedulable. **Always read the reason field in `squeue`, not just the state**, and
be wary of overriding a `--time` that a committed slurm script already set correctly — a CLI
`sbatch --time=...` silently wins over the `#SBATCH` line in the file.

If the work genuinely needs more than 2h, split it or move to `-t3`/`-t4`; do not raise
`--time` on a `-dev` job.

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

## Compute nodes have no internet

Login nodes have network access; compute nodes do not. Anything that would download at
runtime must be baked beforehand on a login node into `$SCRATCH/models/...`.

MolmoPoint-8B lives at `$SCRATCH/models/MolmoPoint-8B`; fetch it with
`hf download allenai/MolmoPoint-8B --local-dir ...` from a login node. Its modelling code
is loaded with `trust_remote_code=True` from that directory, so nothing is fetched at
run time — but only if the download completed, which `du -sh` on the directory confirms
faster than a failed job does.

## $WORK is out of inodes, not out of space

`idr_quota_project` reports the number that matters: **98.8% of 500,000 inodes used, against
7% of the 5 TiB** (measured 2026-08-08; it was ~96% when this page was written). A single
torch install is roughly 40,000 files, so building any new uv environment inside the repo
fails with `Disk quota exceeded (os error 122)` mid-extraction while `df` still shows
terabytes free. At that occupancy a `git worktree add` fails too, part-way through checkout.

Note that `idr_quota_project` is refreshed daily, so it will keep reporting the pre-cleanup
number for hours after a cleanup that really did work. Trust `du --inodes -d 1` for the
before/after, and the enforcement itself for whether you have room.

**Check your shell actually agrees with this page.** As of 2026-08-08 `~/.zshrc` exported
all three caches to `$WORK`, which is how `.cache/uv` reached 237,000 inodes — 47% of the
whole project quota — on its own:

```bash
export PIP_CACHE_DIR=$WORK/.cache/pip     # wrong, use $SCRATCH
export UV_CACHE_DIR=$WORK/.cache/uv       # wrong, use $SCRATCH
export HF_HOME=$WORK/.cache/huggingface   # wrong, use $SCRATCH
```

The advice below is only advice until the profile stops overriding it, and every job script
here that sets `HF_HOME=$SCRATCH/.cache/huggingface` is working around that line rather than
being belt-and-braces.

`uv cache prune` is the cheap recovery: it removes only entries no longer referenced and
freed ~45,000 inodes (11 GiB) here. It is not sufficient on its own — most of what remains
is hardlinked into live venvs, where deleting the cache copy frees a link and not an inode.

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
