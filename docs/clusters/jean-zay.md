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
- **GPU generation does not constrain the simulator.** RoboCasa365 rollouts produce
  identical output on H100 and V100 (measured on CLEPS 2026-08-17, see
  "RoboCasa365 across GPU generations" in `cleps.md`).
- Pure-torch inference (e.g. the YOLO-World ROI detection pass) has no simulator
  dependency and runs happily on H100.

So rollout-plus-model workloads (i.e. policy evaluation) need Ampere+, which here means
A100 or H100 — pick on availability rather than on a sim constraint.

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

## Camera calibration (RoboCasa datasets)

Intrinsics and extrinsics are **not stored in the dataset**. Recover them from a
simulator env reset (V100, `MUJOCO_GL=egl`).

Two facts that save time:

- The `base -> cam` transform is **constant across episodes** for the robot-mounted
  agentview cameras, even though world-frame `base_pos` varies a lot. One global
  calibration is valid for every frame.
- Stored RGB and depth are both `[::-1]`-flipped consistently, so a bounding box on a
  stored frame shares row indexing with the reprojected points.
