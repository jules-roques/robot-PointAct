# ROI-Guided Point Sampling for PointAct (RoboCASA openDrawer PoC)

**Date:** 2026-07-26
**Status:** Design approved, pending spec review
**Scope:** RoboCASA `openDrawer` single-task proof-of-concept only.

## Problem

PointAct samples its point-cloud input by **uniform random subsampling** to a
fixed budget (`data_3d.py:208-237`, `np.random.choice`, `max_npoints=4096`).
This is task-agnostic: for `openDrawer` most of the 4096 points land on
task-irrelevant surfaces (table, walls, background), and the drawer/handle —
the region the policy actually needs at fine resolution — is undersampled and
washed out. Fine handle geometry that exists at the stored 1 cm resolution is
effectively lost to the uniform draw.

## Goal

Reallocate the **same 4096-point budget** toward task-relevant regions (drawer,
especially the handle) using a 2D segmentation model to identify the region of
interest (ROI), while keeping some coverage of the surroundings. **Every other
training hyperparameter and architecture detail stays identical to the baseline**
so the comparison isolates a single variable: the sampling distribution.

## Non-goals

- No re-running of data acquisition. We operate on the existing stored clouds
  (LMDB, xyzrgb, voxelized at 1 cm) and existing RGB frames.
- No change to the trained policy architecture. The segmentation model is a
  **perception/preprocessing module**; it never enters the PointAct network.
  This is what keeps the A/B comparison fair.
- No multi-task generalization in this PoC (openDrawer only).
- Semantic feature lifting (research line #1) and generative action heads
  (#2) are out of scope here.

## Key feasibility facts (verified in code)

1. **Pixel↔point correspondence.** `make_point_cloud` (`replay.py:186`) builds
   each camera's cloud from `observation.points.{cam}` (H×W×3 xyz) concatenated
   with aligned RGB, *before* `voxel_downsample`. A 2D mask maps to 3D points
   directly pre-voxel; post-voxel we recover the mapping by reprojection (below).
2. **Camera intrinsics + extrinsics are available at both stages.**
   `environments.py:394-395` emits `observation.camera_intrinsics.*` /
   `observation.camera_extrinsics.*` at rollout; the same matrices exist during
   replay. Clouds are in `robot_base` frame with cam2world extrinsics, so any 3D
   point can be reprojected into any camera image.
3. **ROI is a geometric label, not a per-camera subset.** After voxel merging,
   points carry no camera-of-origin tag. Reprojecting *all* points into the
   left/right masks means wrist-captured points are **not discarded** — a
   wrist-origin point that lands in a left/right mask is labeled ROI; points
   outside the mask remain in the background pool (still sampled).

## Chosen approach

**(A) Fully consistent path:** SAM2-guided sampling runs at **both** training and
evaluation, so train/eval see identically-distributed points. This is the honest
apples-to-apples test and the reason the segmentation models must be lightweight
(they run live in the rollout loop).

### Section 1 — Perception (per episode)

- **Cameras:** `left` + `right` (static). Wrist excluded from *segmentation*
  (camera motion complicates propagation) but wrist *points* are still
  ROI-labeled by reprojection into left/right masks.
- **Prompt:** single-task, so a **fixed** open-vocab text prompt
  `"drawer. drawer handle."` into **Grounding-DINO-tiny** on the first frame of
  each static camera → bounding box(es). No task-string parsing.
- **Mask + propagation:** box → **SAM2.1-tiny (video)** → mask, propagated across
  the episode frames via SAM2 memory. One prompt per episode.
- **Robustness:** empty / low-confidence detection on a frame → that frame falls
  back to **uniform** sampling (never worse than baseline).

### Section 2 — Lift + sample (count-matched to baseline)

- **Lift:** reproject each stored 3D point into left/right via
  intrinsics+extrinsics; point is **ROI** if it lands inside either mask.
- **Count parity:** preserve the baseline total-count rule exactly —
  `N = min(int(len(cloud) · U(0.8, 1.0)), 4096)` — so the *number* of points is
  identically distributed; only *which* points are selected changes.
- **Split (guarded, 2-tier):** allocate `round(roi_ratio · N)` (default
  `roi_ratio = 0.6`) to ROI points and the remainder to background, each sampled
  **without replacement**. If a pool has fewer points than its quota, top up from
  the other pool so the total is always exactly `N`.
- **Config knob:** `roi_ratio` (single new hyperparameter).
- **Extension (flagged, not built):** 3-tier weighting (handle > drawer > table)
  by keeping the two prompt masks separate. Trivial change to the weight vector.

### Section 3 — Wiring + fair-comparison controls

- **Offline cache:** a preprocessing pass runs Grounding-DINO + SAM2 over the
  existing episodes, computes a per-point ROI flag **aligned to each stored
  `ep-frame` cloud**, and writes it to a **parallel LMDB** (keyed identically:
  `f"{ep_idx}-{frame_idx}"`). Stored as a per-point `uint8`/bit-packed array in
  the same point order as the source cloud.
- **Training path:** `augment_point_cloud` (`data_3d.py:208`) loads the ROI flag
  alongside the cloud, applies the workspace filter to **both in lockstep**, then
  replaces the uniform `np.random.choice` with the guarded split. Near-zero
  training overhead (SAM2 already ran offline).
- **Eval path:** the same detector→SAM2→lift→split runs live in the RoboCASA
  rollout obs pipeline (novel scenes each episode → cannot be cached). Exact
  insertion point to be pinned during planning (candidate:
  `processing_vla_pointact.py::_prepare_robot_inputs`, `line 85`, and/or the
  rollout env obs assembly).
- **Frozen for fairness:** identical PTv3 backbone, action heads, all
  hyperparameters, seeds, color/rotation augmentation, and the 4096 budget. The
  only difference vs. baseline is the selection distribution.

### Section 4 — Visualization gate (hard gate before retraining)

After the offline ROI cache is built, a verification script runs on a handful of
sample frames/episodes and produces:

1. **2D mask overlays** — SAM2 masks drawn on left/right RGB frames (PNG) to
   confirm segmentation quality.
2. **3D ROI-colored cloud** — the stored cloud rendered with ROI vs. background
   highlighted, as a **self-contained interactive HTML** (rotate/zoom on a
   headless cluster, open locally), plus a `.ply` export.
3. **Post-sample view** — the actual 4096-point draw (ROI 60% / bg 40%) showing
   the final density the model trains on.

**Go/no-go:** a human reviews these before any training run. If wrist-only
surfaces are systematically missed, the flagged extension (run SAM2 on the wrist
stream too) is reconsidered.

## Success criteria

- Visualization gate shows the drawer/handle is visibly and correctly densified
  relative to uniform sampling, with background still represented.
- A training run with ROI-guided sampling (all else frozen) completes and is
  compared A/B against the uniform baseline on RoboCASA `openDrawer` success rate.
- Eval rollout runs the online path within acceptable latency (tiny models).

## Open items to resolve during planning

- Exact eval-time insertion point for the online sampler.
- Whether Grounding-DINO-tiny and SAM2.1-tiny weights are already available in
  the environment or need to be added (and their license/footprint).
- Storage format details for the ROI-flag LMDB (bit-packing vs. uint8).
- Workspace-filter ordering: ensure ROI flag and cloud are filtered together.
