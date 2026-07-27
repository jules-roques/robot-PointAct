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
especially the handle) using a lightweight 2D open-vocab detector to localize the region of
interest (ROI), while keeping some coverage of the surroundings. **Every other
training hyperparameter and architecture detail stays identical to the baseline**
so the comparison isolates a single variable: the sampling distribution.

## Non-goals

- No re-running of data acquisition. We operate on the existing stored clouds
  (LMDB, xyzrgb, voxelized at 1 cm) and existing RGB frames.
- No change to the trained policy architecture. The detector is a
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

**(A) Fully consistent path:** ROI-guided sampling runs at **both** training and
evaluation, so train/eval see identically-distributed points. This is the honest
apples-to-apples test and the reason the localizer must be lightweight (it runs
live in the rollout loop).

**Detector + halo, not segmentation.** We never consume a mask boundary — only
"where roughly is the object." So the primary design uses a lightweight
open-vocab **detector** to find the object's *epicenter*, then draws a 3D
**halo** around it. This is far lighter than any SAM variant and avoids SAM 3's
python-3.12 requirement in the eval simulator.

### Section 1 — Perception (per frame)

- **Cameras:** `left` + `right` (static). Wrist excluded from *detection*
  (camera motion) but wrist *points* are still ROI-labeled by the 3D halo test.
- **Model:** **YOLO-World** (small variant) — open-vocab, text-promptable,
  real-time (tens of M params), shipped in `ultralytics`. Fixed prompt
  `"drawer handle"` (single task). Run **per frame** on left/right → 2D box(es).
  Per-frame detection is cheap, so no video tracking is needed (the drawer moving
  as it opens is handled by simply re-detecting).
- **Fallbacks (documented, not built):** if detection quality is too poor,
  escalate to (a) SAM 3.1 (single text-promptable model, ~840M params, needs
  python 3.12) or (b) Grounding-DINO-tiny (box) + SAM2.1-tiny (mask/propagation).
- **Robustness:** empty / low-confidence detection on a frame → that frame falls
  back to **uniform** sampling (never worse than baseline).

### Section 2 — Lift + sample (count-matched to baseline)

- **2D box → 3D epicenter (robust):** reproject the stored cloud into the camera
  (intrinsics + extrinsics), keep points whose (u,v) fall inside the box, and take
  their **3D centroid** as the anchor. Robust to a sloppy box and to depth holes
  (uses many points, not one back-projected pixel). Left/right anchors are merged.
- **Halo → ROI (sphere):** a point is **ROI** if it lies within radius `r` of the
  3D anchor. `r` is auto-scaled from the spread of the in-box points
  (`r = halo_scale · std_of_in_box_points`), so it adapts to object size. Wrist
  points inside the halo are included; nothing is discarded.
  - *Halo-shape alternatives (flagged, not built):* frustum from the 2D box +
    depth band (hugs an elongated drawer front better); soft Gaussian falloff
    (continuous weight instead of hard in/out).
- **Camera calibration:** intrinsics/extrinsics are **not** stored in the dataset.
  Offline, fetch them once via a RoboCASA env reset per episode (cameras are
  static for fixed-base OpenDrawer; `_get_camera_matrices`, robot-base frame via
  `robot0_base_pos/quat`). Online, they arrive for free in the eval obs
  (`observation.camera_intrinsics.*` / `camera_extrinsics.*`).
- **Count parity:** preserve the baseline total-count rule exactly —
  `N = min(int(len(cloud) · U(0.8, 1.0)), 4096)` — so the *number* of points is
  identically distributed; only *which* points are selected changes.
- **Split (guarded, 2-tier):** allocate `round(roi_ratio · N)` (default
  `roi_ratio = 0.6`) to ROI points and the remainder to background, each sampled
  **without replacement**. If a pool has fewer points than its quota, top up from
  the other pool so the total is always exactly `N`.
- **Config knobs:** `roi_ratio` (budget split, default 0.6) and `halo_scale`
  (radius multiplier).

### Section 3 — Wiring + fair-comparison controls

- **Offline cache:** a preprocessing pass runs YOLO-World over the existing
  episodes' left/right frames, computes the halo, and derives a per-point ROI
  flag **aligned to each stored `ep-frame` cloud**, written to a **parallel
  LMDB** (keyed identically:
  `f"{ep_idx}-{frame_idx}"`). Stored as a per-point `uint8`/bit-packed array in
  the same point order as the source cloud.
- **Training path:** `augment_point_cloud` (`data_3d.py:208`) loads the ROI flag
  alongside the cloud, applies the workspace filter to **both in lockstep**, then
  replaces the uniform `np.random.choice` with the guarded split. Near-zero
  training overhead (YOLO-World already ran offline).
- **Eval path (follow-up, not required for the first training):** the same
  YOLO-World→epicenter→halo→split runs live in the RoboCASA rollout obs pipeline
  (novel scenes each episode → cannot be cached; camera matrices come from the
  obs). Exact insertion point to be pinned during planning (candidate:
  `processing_vla_pointact.py::_prepare_robot_inputs`, `line 85`, and/or the
  rollout env obs assembly). The first 5-epoch training run exercises only the
  offline cache + dataloader change; the online path is wired before eval.
- **Frozen for fairness:** identical PTv3 backbone, action heads, all
  hyperparameters, seeds, color/rotation augmentation, and the 4096 budget. The
  only difference vs. baseline is the selection distribution.

### Section 4 — Visualization gate (hard gate before retraining)

After the offline ROI cache is built, a verification script runs on a handful of
sample frames/episodes and produces:

1. **2D box + halo overlays** — YOLO-World boxes (and the projected halo circle)
   drawn on left/right RGB frames (PNG) to confirm localization quality.
2. **3D ROI-colored cloud** — the stored cloud rendered with ROI vs. background
   highlighted, as a **self-contained interactive HTML** (rotate/zoom on a
   headless cluster, open locally), plus a `.ply` export.
3. **Post-sample view** — the actual 4096-point draw (ROI 60% / bg 40%) showing
   the final density the model trains on.

**Go/no-go:** a human reviews these before any training run. If wrist-only
surfaces are systematically missed, the flagged extension (also detect on the
wrist stream) is reconsidered.

## Success criteria

- Visualization gate shows the drawer/handle is visibly and correctly densified
  relative to uniform sampling, with background still represented.
- A training run with ROI-guided sampling (all else frozen) completes and is
  compared A/B against the uniform baseline on RoboCASA `openDrawer` success rate.
- Eval rollout runs the online path within acceptable latency (YOLO-World small).

## Scope of the first deliverable (5-epoch OpenDrawer PoC)

1. Base work on `robocasa365-integration` (all RoboCASA training machinery lives
   there; `main` lacks it).
2. Stage YOLO-World: add `ultralytics`, pre-download small weights on an
   internet-enabled node to a fixed path (compute nodes are offline).
3. Fetch static left/right camera matrices via the RoboCASA sim env.
4. Offline preprocessing → per-point ROI-flag parallel LMDB for OpenDrawer.
5. Dataloader: guarded ROI/background split in `augment_point_cloud`
   (`data_3d.py`), with uniform fallback.
6. Visualization gate (interactive HTML) → eyeball.
7. Launch 5-epoch OpenDrawer training (ROI variant, and a matched uniform
   baseline if none exists).

The online eval path is the **next** deliverable, after training shows signal.

## Findings from implementation (2026-07-27)

**The instruction's "left"/"right" cannot be decoded geometrically.** RoboCASA OpenDrawer
instructions name a side, but the robot parks at a different yaw in each kitchen, so the
same instruction grasps at both signs of the base-frame Y axis (left-drawer episodes at
Y = -0.607, -0.459, +0.473, -0.418, +0.503). Neither an image-space "leftmost box" rule
nor a base-frame axis test works — both scored 3/6 episodes.

**Drawer selection now uses the demonstration's grasp point.** `observation.state[0:3]`
(`base_to_eef_pos`, the point-cloud frame) at the first frame `action[7] > 0.5`
(gripper close) is the handle the demonstration reached for. It is a per-episode constant,
so it disambiguates every frame including frame 0, without the ROI trailing the arm.
Precomputed for all 496 episodes by `dump_grasp_anchors.py` into a JSON sidecar.
Result: **6/6 preview episodes correct, mean lateral error 0.079 m** (was 0.335 m).

**Unconfirmed frames fall back to uniform.** When no detected candidate lies within
0.45 m of the grasp point, the target drawer was not found; emitting no halo (uniform
sampling) keeps such frames baseline-equivalent, whereas guessing the best-supported
candidate put one episode's ROI ~1 m away on the wrong drawer.

**Coverage is partial — the treatment is diluted.** Over the full 496-episode cache only
**43.3% of frames** carry a confirmed ROI (per-episode median 37%; 35% of episodes below
25%). The remaining frames sample uniformly, so the A/B measures a *partial* treatment.
Raising coverage without weakening the correctness guard is the obvious next lever: lower
the detector confidence threshold and raise `--max-det` so the correct drawer is more
often among the candidates (selection is nearest-to-grasp, so extra candidates are safe).

**Eval-time limitation (open).** The grasp point exists only in demonstrations. The online
eval path therefore still needs a standalone selector — candidates: the policy's own early
reach, or prompting the VLM for the target drawer. This does not affect the training-time
A/B but must be solved before the consistent train/eval path is complete.

## Open items to resolve during planning

- Exact eval-time insertion point for the online sampler.
- Storage format details for the ROI-flag LMDB (bit-packing vs. uint8).
- Workspace-filter ordering: ensure ROI flag and cloud are filtered together.
- Whether static camera matrices are constant across OpenDrawer episodes (fetch
  per-episode to be safe; collapse to one fetch if verified constant).
