# ROI-guided point sampling

Concentrates the fixed point budget where the task is, instead of sampling the cloud
uniformly. Every guided arm uses the **same** Gaussian-with-floor density
(`pointact/roi_sampling/geometry.py:eef_density_weights`) and differs only in where the
bump is centred:

| arm | centre | needs |
|---|---|---|
| `uniform` | — (baseline) | nothing |
| `eef` | the frame's own end-effector, `observation.state[:3]` | nothing |
| `oracle` | centroid of the points MuJoCo labels as the handle | `points_3views_labels` LMDB |
| `molmo` | where a frozen MolmoPoint-8B points, given the episode's instruction | this pipeline |

`oracle` is privileged information and exists to upper-bound what any detector could buy.
`molmo` is the honest version of it: same Gaussian, but the centre comes from an
off-the-shelf model that only sees what a policy could see.

## Pipeline

```
Stage A  calib_jeanzay.slurm       # robocasa365 env, V100: dump left/right camera calibration
Stage B  build_molmo_cache.py      # envs/molmo, A100/H100: per-frame anchor LMDB
Gate     viz_molmo_gate.py         # videos + point-cloud animations -> W&B, eyeball before training
Train    experiments/13_robocasa365/runs/*-molmo-*.yaml
```

### Stage A — camera calibration
```bash
sbatch --export=ALL,TASK=PickPlaceCounterToStove \
       experiments/13_robocasa365/calib_jeanzay.slurm
```
Writes `<task>/roi_meta/camera_calib.npz` (intrinsics + base→cam for the left and right
agentviews). The cameras are robot-mounted, so one calibration serves the whole dataset;
the script verifies that across episodes and **warns if it does not hold** — that warning
invalidates the reprojection this pipeline depends on, so stop and investigate.

### Stage B — anchor cache
```bash
sbatch --export=ALL,TASK=OpenDrawer experiments/13_robocasa365/molmo_cache_jeanzay.slurm
```
Writes `<task>/points_3views_molmo`, one record per frame keyed `{episode}-{frame}` like the
point LMDB, plus `roi_meta/molmo_build_summary.json` (per-view point rate, cross-view
agreement, fallback rate).

How a pixel becomes a 3D anchor: MolmoPoint names a pixel, that pixel is padded into a
small window, the stored base-frame cloud is projected into the camera, and the anchor is
the **median of the points landing in the window**. Lifting this way rather than reading
the depth map at the pixel means invalid-depth pixels reduce support instead of corrupting
the anchor.

Pointing runs at the policy's replan cadence (every 8 frames), not every frame, because
that is what eval can afford — the same anchor is then held across the intervening frames
so training sees exactly the signal eval will produce.

The cache stores **every** pointing query the task defines, so choosing which become
Gaussian centres is a dataloader knob (`molmo_anchor_ids`) rather than a rebuild:
`[0]` is the manipulated object alone, `[0, 1]` adds the destination.

### Gate — visualisation
```bash
python -m data_prep.roi_sampling.viz_molmo_gate --tasks OpenDrawer PickPlaceCounterToStove TurnOnMicrowave
```
Left- and right-camera videos with the predicted point drawn on each frame, plus the
interactive point-cloud animation from `viz_sampling_episode.py --method molmo`. Logged to
a W&B run tagged `Stage 3.0: Visu MolmoPoint`. Look for: does the point land on the named
object, does it pick the **correct** drawer on left/right episodes, does the lifted anchor
sit on the object in 3D, and does it jitter between replans?

## Notes

- An earlier version of this pipeline used YOLO-World. It was removed: an open-vocabulary
  box detector cannot tell which drawer an instruction names, which forced the
  demonstration's grasp point in as a disambiguator — privileged information in what was
  supposed to be the unprivileged arm. MolmoPoint reads "left"/"right" from the instruction
  itself.
- The 1 cm voxel grid, not the point budget, is what currently caps these samplers: at
  `eef` roughly 97% of the points within 1σ are already taken. A better-aimed Gaussian
  changes *which* region saturates, not how finely it is resolved.
