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
| `molmo_motion` | where a frozen MolmoMotion-4B forecasts the **gripper** will be | this pipeline (Stage 4) |

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
a W&B run tagged `Stage 3: MolmoPoint anchor`, the same stage tag the trained arms carry, so
one filter returns the gate and the runs it gated; `gate` tells them apart. Look for: does
the point land on the named
object, does it pick the **correct** drawer on left/right episodes, does the lifted anchor
sit on the object in 3D, and does it jitter between replans?

### Scoring the detector on its own

```bash
# TurnOnMicrowave needs ground truth dumped from the simulator first (V100, ~40 min)
sbatch --export=ALL,TASK=TurnOnMicrowave,RESET_ONLY=1 \
       experiments/13_robocasa365/target_positions_jeanzay.slurm
# then merge the shards, and build the episode map the join needs
python -m data_prep.roi_sampling.dump_target_positions --task TurnOnMicrowave \
       --out <task>/roi_meta/target_positions.npz --merge <task>/roi_meta/target_positions_shard*.npz
python -m data_prep.robocasa365_to_lerobot.episode_index_map \
       --source-dir <src>/lerobot --dataset-dir <task>

sbatch --export=ALL,TASK=OpenDrawer experiments/13_robocasa365/molmo_accuracy_jeanzay.slurm
```

`eval_molmo_accuracy.py` measures the pointer with no policy in the loop, from the cache
alone — no GPU, since the 8B forwards were paid once at build time. It reports the outcome
split (anchored / pointed-but-unliftable / pointed nowhere, which `frame_cover` merges), the
3D error distribution against the sampling sigma, per-camera pixel error from the stored
detections against the reprojected ground truth, distractor margins, and replan jitter.

Ground truth differs by task and the script picks neither for you. **OpenDrawer** uses
`--gt labels`, the same handle centroid the GT-oracle arm centres its Gaussian on, so the
error is the gap between this arm and the oracle arm's input. **TurnOnMicrowave** has no
labels and cannot cheaply get them — the labeller finds a task's target by probing
`env.drawer/.door/.cab/.obj/…` and the only match there is `.obj`, the food item *inside*
the microwave — so it uses `--gt geom --target start_button` against
`dump_target_positions.py`. That join goes through `source_episode_map.json`, because
converted episode indices are not source episode indices (OpenDrawer drops 18 failed replays
and renumbers); `--gt geom` fails rather than assuming the identity.

## Stage 4 — MolmoMotion, a forecast instead of a detection

`molmo` centres the Gaussian on a *detected object*. `molmo_motion` centres it on where the
**gripper is predicted to go**, which is a different bet with a different failure mode.

`allenai/MolmoMotion-4B-H3-F30` takes three history frames from one camera, a query point as
a pixel plus its 3D history in camera-frame-at-t₀, and the episode's instruction; it returns
that point's future 3D track, 30 steps of absolute camera-frame metres. The query point is
the **end-effector**, which proprio already gives us exactly — so unlike the MolmoPoint arm,
nothing here can point at the wrong object. The worst case is a bad forecast of a correctly
identified point.

Two structural differences from Stage 3 are worth stating before any numbers exist:

- **A forecast track starts at the current gripper.** A Gaussian tube along it therefore
  contains the `eef` arm's bump and extends it forward, where a static object anchor
  abandoned the gripper region entirely. That is the specific Stage 3 failure this is meant
  to avoid — though "meant to" is a hypothesis until measured.
- **It is single-camera.** MolmoPoint took left+right+wrist in one forward and each returned
  point named its image. MolmoMotion's image list axis is time, not view, so a second view is
  a second forward. Do not assume the three-view trick from `build_molmo_cache.py` carries
  over.

### Gate — is the forecast true?

```bash
sbatch experiments/13_robocasa365/gate_molmo_motion_jeanzay.slurm
```

This gate is stronger than Stage 3's, which is why it runs before anything else. MolmoPoint
had to be scored against simulator labels that only two of three tasks have. Here the ground
truth is free and exact — the gripper's future position is `observation.state[:3]` a few
frames later — so `gate_molmo_motion.py` scores the forecast in cm on every task with no
labelling step.

It reports, per task and horizon, the forecast error, the **static baseline** ("the gripper
does not move"), and the win rate against it. That baseline is the number to read first: if
the gripper travels 3 cm in a second and the forecast errs by 4 cm, the model is worse than
assuming nothing happens, however plausible the overlay video looks. Stage 3 shipped without
such a control.

It also sweeps the **history stride**, because the checkpoint was trained at 15 fps, the data
is 20 fps, and the processor passes the model no timestamps to reconcile them (it builds them
as `np.arange(H) * 1.0`). 1/15 s is 1.333 frames at 20 fps and so cannot be sampled exactly;
rounding to `[t-3, t-1, t]` would feed non-uniform spacing to a model that assumes uniform.
Integer strides keep the assumption intact and the gate measures which transfers, rather than
this pipeline picking one and hoping.

## Notes

- An earlier version of this pipeline used YOLO-World. It was removed: an open-vocabulary
  box detector cannot tell which drawer an instruction names, which forced the
  demonstration's grasp point in as a disambiguator — privileged information in what was
  supposed to be the unprivileged arm. MolmoPoint reads "left"/"right" from the instruction
  itself.
- The 1 cm voxel grid, not the point budget, is what currently caps these samplers: at
  `eef` roughly 97% of the points within 1σ are already taken. A better-aimed Gaussian
  changes *which* region saturates, not how finely it is resolved.
