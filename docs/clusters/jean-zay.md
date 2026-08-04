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

## Compute nodes have no internet

Login nodes have network access; compute nodes do not. Anything that would download at
runtime must be baked beforehand on a login node into `$SCRATCH/models/...`.

MolmoPoint-8B lives at `$SCRATCH/models/MolmoPoint-8B`; fetch it with
`hf download allenai/MolmoPoint-8B --local-dir ...` from a login node. Its modelling code
is loaded with `trust_remote_code=True` from that directory, so nothing is fetched at
run time — but only if the download completed, which `du -sh` on the directory confirms
faster than a failed job does.

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
