# ROI-Guided Point Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bias PointAct's 4096-point sampling toward the drawer/handle on RoboCASA `openDrawer` using SAM 3.1 masks, at both train and eval, with everything else frozen — and gate on a 3D visualization before retraining.

**Architecture:** SAM 3.1 (text prompt `"drawer handle"`) segments the left/right video streams. A shared geometry module reprojects the stored/observed 3D point cloud into those masks to label each point ROI vs. background. A shared count-matched split sampler keeps exactly the baseline's point count but allocates `roi_ratio` of it to ROI points. Offline, this is precomputed into a parallel LMDB consumed by the dataloader; online, the same segment→reproject→split runs live in the eval processor. SAM 3.1 never enters the trained policy.

**Tech Stack:** Python 3.10, PyTorch 2.7, NumPy 1.26.4, open3d 0.18, lmdb + msgpack, SAM 3.1 (`facebookresearch/sam3`), pytest.

## Global Constraints

- **RoboCASA `openDrawer` single-task only.** No other tasks/benchmarks.
- **No re-running data acquisition.** Stored point clouds (`points_3views` LMDB, xyzrgb, 1 cm voxel) are read-only inputs. Reading static camera calibration via a lightweight env reset (no rollout stepping) is permitted; regenerating clouds is not.
- **Policy architecture and all training hyperparameters frozen** vs. baseline: same PTv3, heads, seeds, `augment_pc_rot: 30`, color aug, and `max_npoints: 4096`. The only new hyperparameter is `roi_ratio` (default `0.6`).
- **Point count parity:** the number of sampled points must follow the exact baseline rule `N = min(int(len(cloud) * U(0.8, 1.0)), 4096)` — only the *selection distribution* changes.
- **Never worse than baseline fallback:** empty/failed mask on a frame → plain uniform sampling.
- **Point clouds are in `robot_base` frame.** Cameras `left`/`right` are world-fixed; `wrist` moves. Stored RGB images are vertically flipped (`environments.py:465`, `image[::-1]`); reprojection must honor this.
- All new deps go under an extra, not the core `dependencies` list. Run tests with `python -m pytest`.

---

## File Structure

**New package `pointact/data/roi_sampling/` (pure, shared by train + eval):**
- `reproject.py` — geometry: robot-base points → pixel coords → per-point ROI mask lookup.
- `roi_sample.py` — count-matched guarded split sampler.
- `segmenter.py` — SAM 3.1 wrapper (text → per-frame masks), lazy-loaded, with graceful fallback.
- `visualize.py` — build self-contained interactive HTML + PLY + 2D overlays.

**New data-prep scripts:**
- `data_prep/robocasa365_to_lerobot/extract_camera_calib.py` — one-time per-episode static calib sidecar.
- `data_prep/robocasa365_to_lerobot/build_roi_cache.py` — offline ROI-flag LMDB builder.
- `scripts/visualize_roi.py` — CLI wrapper around `visualize.py` for the gate.

**Modified:**
- `pointact/data/robot/data_3d.py` — load ROI flag, filter in lockstep, guarded split.
- `pointact/model/backbone/processor_base.py` — online ROI path in `_prepare_point_cloud_for_sample`.
- `experiments/13_robocasa365/data_configs/data-robocasa365-opendrawer-point.yaml` — add `roi_ratio`, `roi_cache_dirname`.
- `pyproject.toml` — add `[project.optional-dependencies].roi` = `["sam3", ...]` (or documented install).

**Tests:** `tests/roi_sampling/` (new; create `tests/__init__.py` and `tests/roi_sampling/__init__.py`).

---

## Task 1: Count-matched guarded split sampler (pure)

**Files:**
- Create: `pointact/data/roi_sampling/__init__.py` (empty)
- Create: `pointact/data/roi_sampling/roi_sample.py`
- Create: `tests/__init__.py`, `tests/roi_sampling/__init__.py` (empty)
- Test: `tests/roi_sampling/test_roi_sample.py`

**Interfaces:**
- Produces: `roi_guided_indices(roi_mask: np.ndarray, n_total: int, roi_ratio: float, rng: np.random.Generator) -> np.ndarray` — returns exactly `min(n_total, len(roi_mask))` unique indices into the cloud, drawn without replacement, allocating `round(roi_ratio * n_total)` to ROI points and the rest to background, topping up from the other pool when short. `roi_mask` is a bool array (True = ROI). If `roi_mask` is all-False or all-True, degrades to a uniform draw of `n_total`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/roi_sampling/test_roi_sample.py
import numpy as np
import pytest
from pointact.data.roi_sampling.roi_sample import roi_guided_indices


def _rng():
    return np.random.default_rng(0)


def test_returns_exact_count_and_unique():
    roi_mask = np.array([True] * 50 + [False] * 50)
    idx = roi_guided_indices(roi_mask, n_total=40, roi_ratio=0.6, rng=_rng())
    assert idx.shape == (40,)
    assert len(np.unique(idx)) == 40
    assert idx.min() >= 0 and idx.max() < 100


def test_ratio_allocation():
    roi_mask = np.array([True] * 50 + [False] * 50)
    idx = roi_guided_indices(roi_mask, n_total=40, roi_ratio=0.6, rng=_rng())
    n_roi = int(roi_mask[idx].sum())
    assert n_roi == 24  # round(0.6 * 40)


def test_tops_up_when_roi_pool_short():
    roi_mask = np.array([True] * 5 + [False] * 95)  # only 5 ROI points
    idx = roi_guided_indices(roi_mask, n_total=40, roi_ratio=0.6, rng=_rng())
    assert idx.shape == (40,)
    assert len(np.unique(idx)) == 40
    assert int(roi_mask[idx].sum()) == 5  # all ROI taken, rest from background


def test_tops_up_when_background_pool_short():
    roi_mask = np.array([True] * 95 + [False] * 5)
    idx = roi_guided_indices(roi_mask, n_total=40, roi_ratio=0.6, rng=_rng())
    assert idx.shape == (40,)
    assert int((~roi_mask[idx]).sum()) == 5  # all background taken


def test_all_false_mask_degrades_to_uniform():
    roi_mask = np.zeros(100, dtype=bool)
    idx = roi_guided_indices(roi_mask, n_total=40, roi_ratio=0.6, rng=_rng())
    assert idx.shape == (40,)
    assert len(np.unique(idx)) == 40


def test_n_total_capped_at_cloud_size():
    roi_mask = np.array([True] * 10 + [False] * 10)
    idx = roi_guided_indices(roi_mask, n_total=100, roi_ratio=0.6, rng=_rng())
    assert idx.shape == (20,)
    assert len(np.unique(idx)) == 20
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/roi_sampling/test_roi_sample.py -v`
Expected: FAIL with `ModuleNotFoundError`/`ImportError` for `roi_guided_indices`.

- [ ] **Step 3: Implement `roi_guided_indices`**

```python
# pointact/data/roi_sampling/roi_sample.py
import numpy as np


def roi_guided_indices(
    roi_mask: np.ndarray,
    n_total: int,
    roi_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick exactly min(n_total, len(roi_mask)) unique cloud indices, allocating
    round(roi_ratio * n_total) to ROI points (roi_mask True) and the remainder to
    background, without replacement. Short pools are topped up from the other pool.
    Degenerate masks (all-True / all-False) fall back to a uniform draw."""
    n_points = len(roi_mask)
    n_total = int(min(n_total, n_points))
    all_idx = np.arange(n_points)

    roi_idx = all_idx[roi_mask]
    bg_idx = all_idx[~roi_mask]
    if len(roi_idx) == 0 or len(bg_idx) == 0:
        return rng.choice(n_points, size=n_total, replace=False)

    n_roi = int(round(roi_ratio * n_total))
    n_roi = min(n_roi, len(roi_idx))
    n_bg = n_total - n_roi
    if n_bg > len(bg_idx):  # background short: shift remainder to ROI
        n_bg = len(bg_idx)
        n_roi = n_total - n_bg

    picked_roi = rng.choice(roi_idx, size=n_roi, replace=False)
    picked_bg = rng.choice(bg_idx, size=n_bg, replace=False)
    out = np.concatenate([picked_roi, picked_bg])
    rng.shuffle(out)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/roi_sampling/test_roi_sample.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add pointact/data/roi_sampling/__init__.py pointact/data/roi_sampling/roi_sample.py tests/__init__.py tests/roi_sampling/__init__.py tests/roi_sampling/test_roi_sample.py
git commit -m "feat(roi): count-matched guarded split sampler"
```

---

## Task 2: Reprojection + mask lift (pure geometry)

**Files:**
- Create: `pointact/data/roi_sampling/reproject.py`
- Test: `tests/roi_sampling/test_reproject.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `robot_base_to_world(points_rb: np.ndarray, base_pos: np.ndarray, base_quat_xyzw: np.ndarray) -> np.ndarray` — inverse of `RoboCasa365Env.convert_points_to_robot_base_frame`.
  - `project_to_pixels(points_world: np.ndarray, intrinsic: np.ndarray, extrinsic_cam2world: np.ndarray, image_hw: tuple[int, int], flip_v: bool = True) -> tuple[np.ndarray, np.ndarray]` — returns `(uv_int, valid)`: integer pixel coords `(M, 2)` as `(row, col)` and a bool `valid` mask (in front of camera and inside image bounds). `flip_v=True` mirrors the stored-image vertical flip.
  - `lift_masks_to_roi(points_rb: np.ndarray, base_pos, base_quat_xyzw, cams: list[dict]) -> np.ndarray` — bool `(N,)` ROI flag; a point is ROI if it lands inside any camera's mask. Each `cam` dict has keys `intrinsic`, `extrinsic`, `mask` (HxW bool), `image_hw`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/roi_sampling/test_reproject.py
import numpy as np
from scipy.spatial.transform import Rotation as R
from pointact.data.roi_sampling.reproject import (
    robot_base_to_world, project_to_pixels, lift_masks_to_roi,
)


def test_robot_base_to_world_roundtrip():
    rng = np.random.default_rng(0)
    pts_world = rng.normal(size=(20, 3))
    base_pos = np.array([0.5, -0.2, 0.1])
    base_quat = R.from_euler("z", 0.7).as_quat()  # xyzw
    # forward (world->rb), mimicking env.convert_points_to_robot_base_frame
    rot = R.from_quat(base_quat)
    pts_rb = rot.inv().apply(pts_world - base_pos)
    back = robot_base_to_world(pts_rb, base_pos, base_quat)
    assert np.allclose(back, pts_world, atol=1e-5)


def test_project_center_point():
    # camera at origin looking down -Z (cam2world = identity), point 2m in front
    intrinsic = np.array([[100.0, 0, 32.0], [0, 100.0, 32.0], [0, 0, 1.0]])
    extrinsic = np.eye(4)
    # OpenCV convention: point in front has +Z in cam frame. Place cam2world so
    # that a world point maps to cam +Z; use a cam that looks along +Z:
    pts_world = np.array([[0.0, 0.0, 2.0]])
    uv, valid = project_to_pixels(pts_world, intrinsic, extrinsic, (64, 64), flip_v=False)
    assert valid[0]
    assert tuple(uv[0]) == (32, 32)  # (row, col) at principal point


def test_project_behind_camera_invalid():
    intrinsic = np.array([[100.0, 0, 32.0], [0, 100.0, 32.0], [0, 0, 1.0]])
    extrinsic = np.eye(4)
    pts_world = np.array([[0.0, 0.0, -2.0]])  # behind
    uv, valid = project_to_pixels(pts_world, intrinsic, extrinsic, (64, 64), flip_v=False)
    assert not valid[0]


def test_lift_masks_to_roi_marks_points_in_mask():
    intrinsic = np.array([[100.0, 0, 32.0], [0, 100.0, 32.0], [0, 0, 1.0]])
    extrinsic = np.eye(4)
    mask = np.zeros((64, 64), dtype=bool)
    mask[30:35, 30:35] = True
    pts_rb = np.array([[0.0, 0.0, 2.0], [1.5, 0.0, 2.0]])  # first projects to center
    base_pos = np.zeros(3)
    base_quat = np.array([0.0, 0.0, 0.0, 1.0])
    cams = [{"intrinsic": intrinsic, "extrinsic": extrinsic, "mask": mask, "image_hw": (64, 64)}]
    roi = lift_masks_to_roi(pts_rb, base_pos, base_quat, cams, flip_v=False)
    assert roi[0] and not roi[1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/roi_sampling/test_reproject.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Implement `reproject.py`**

Note conventions: robosuite `get_camera_extrinsic_matrix` returns cam2world; `world2cam = inv(extrinsic)`. Projection uses OpenCV pinhole `[u; v; 1] ~ K [X; Y; Z]_cam` with pixel `col=u, row=v`. `flip_v` mirrors the stored vertical image flip via `row = H - 1 - row`.

```python
# pointact/data/roi_sampling/reproject.py
import numpy as np
from scipy.spatial.transform import Rotation as R


def robot_base_to_world(points_rb, base_pos, base_quat_xyzw):
    rot = R.from_quat(np.asarray(base_quat_xyzw))
    return rot.apply(np.asarray(points_rb, dtype=np.float64)) + np.asarray(base_pos, dtype=np.float64)


def project_to_pixels(points_world, intrinsic, extrinsic_cam2world, image_hw, flip_v=True):
    h, w = image_hw
    pts = np.asarray(points_world, dtype=np.float64)
    world2cam = np.linalg.inv(np.asarray(extrinsic_cam2world, dtype=np.float64))
    homog = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    cam = (world2cam @ homog.T).T[:, :3]
    z = cam[:, 2]
    in_front = z > 1e-6
    z_safe = np.where(in_front, z, 1.0)
    proj = (np.asarray(intrinsic, dtype=np.float64) @ (cam / z_safe[:, None]).T).T
    col = np.round(proj[:, 0]).astype(np.int64)
    row = np.round(proj[:, 1]).astype(np.int64)
    if flip_v:
        row = (h - 1) - row
    valid = in_front & (col >= 0) & (col < w) & (row >= 0) & (row < h)
    uv = np.stack([row, col], axis=1)
    return uv, valid


def lift_masks_to_roi(points_rb, base_pos, base_quat_xyzw, cams, flip_v=True):
    pts_world = robot_base_to_world(points_rb, base_pos, base_quat_xyzw)
    roi = np.zeros(len(points_rb), dtype=bool)
    for cam in cams:
        uv, valid = project_to_pixels(
            pts_world, cam["intrinsic"], cam["extrinsic"], cam["image_hw"], flip_v=flip_v
        )
        idx = np.where(valid)[0]
        if len(idx) == 0:
            continue
        hit = cam["mask"][uv[idx, 0], uv[idx, 1]]
        roi[idx[hit]] = True
    return roi
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/roi_sampling/test_reproject.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add pointact/data/roi_sampling/reproject.py tests/roi_sampling/test_reproject.py
git commit -m "feat(roi): reprojection and mask-to-point lift"
```

---

## Task 3: SAM 3.1 segmenter wrapper

**Files:**
- Create: `pointact/data/roi_sampling/segmenter.py`
- Modify: `pyproject.toml` (add `roi` optional extra + install note)
- Test: `tests/roi_sampling/test_segmenter.py`

**Interfaces:**
- Produces:
  - `class RoiSegmenter` with `__init__(self, text_prompt: str = "drawer handle", model_id: str | None = None, device: str = "cuda")` (lazy-loads SAM 3.1 on first use), and
  - `segment_video(self, frames: np.ndarray) -> np.ndarray` — input `(T, H, W, 3)` uint8, output `(T, H, W)` bool masks (union of all matched instances per frame; empty frame → all-False).
  - `segment_image(self, frame: np.ndarray) -> np.ndarray` — `(H, W, 3)` → `(H, W)` bool.
  - A module-level `masks_are_usable(masks: np.ndarray, min_pixels: int = 20) -> np.ndarray` returning a per-frame bool of whether that frame's mask is usable (else caller falls back to uniform).

- [ ] **Step 1: Write the failing test (no GPU / no SAM3 needed — test the fallback contract)**

```python
# tests/roi_sampling/test_segmenter.py
import numpy as np
from pointact.data.roi_sampling.segmenter import masks_are_usable


def test_masks_are_usable_flags_small_and_empty():
    masks = np.zeros((3, 64, 64), dtype=bool)
    masks[1, 10:40, 10:40] = True          # usable
    masks[2, 0:2, 0:2] = True              # too small
    usable = masks_are_usable(masks, min_pixels=20)
    assert usable.tolist() == [False, True, False]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/roi_sampling/test_segmenter.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Implement `segmenter.py` (lazy SAM 3.1 load; pure `masks_are_usable`)**

```python
# pointact/data/roi_sampling/segmenter.py
import numpy as np


def masks_are_usable(masks: np.ndarray, min_pixels: int = 20) -> np.ndarray:
    counts = masks.reshape(len(masks), -1).sum(axis=1)
    return counts >= min_pixels


class RoiSegmenter:
    """Thin wrapper over SAM 3.1 promptable concept segmentation. Lazy-loads the
    model so importing this module (and running unit tests) needs no GPU/weights."""

    def __init__(self, text_prompt: str = "drawer handle", model_id: str | None = None, device: str = "cuda"):
        self.text_prompt = text_prompt
        self.model_id = model_id
        self.device = device
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            # Import inside method so unit tests don't require sam3 installed.
            from sam3 import build_sam3_video_predictor  # noqa: fill exact import at impl time
            self._model = build_sam3_video_predictor(self.model_id, device=self.device)
        return self._model

    def segment_video(self, frames: np.ndarray) -> np.ndarray:
        model = self._ensure_model()
        # Run PCS with self.text_prompt over the T-frame clip; union instance masks
        # per frame into a single bool (T, H, W). Exact SAM3 call filled at impl time
        # per facebookresearch/sam3 README.
        masks = model.predict_concept(frames, text=self.text_prompt)  # noqa
        return np.asarray(masks, dtype=bool)

    def segment_image(self, frame: np.ndarray) -> np.ndarray:
        return self.segment_video(frame[None])[0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/roi_sampling/test_segmenter.py -v`
Expected: PASS.

- [ ] **Step 5: Add the optional dependency + install note to `pyproject.toml`**

Under `[project.optional-dependencies]` add:
```toml
roi = [
    # SAM 3.1 — install from source per facebookresearch/sam3 (not on PyPI at pin time):
    #   pip install "git+https://github.com/facebookresearch/sam3.git"
    "plotly>=5.20.0",   # interactive HTML visualization
]
```

- [ ] **Step 6: Verify the SAM 3.1 install path on the target machine (record exact import + builder)**

Run (on a GPU node): `pip install "git+https://github.com/facebookresearch/sam3.git"` then in Python confirm the exact predictor builder + concept-segmentation call, and update `_ensure_model`/`segment_video` to the real API. Record the exact model id / checkpoint used.
Expected: `RoiSegmenter(text_prompt="drawer handle").segment_image(frame)` returns a bool `(H, W)` on a sample RoboCASA left image with the drawer visible.

- [ ] **Step 7: Commit**

```bash
git add pointact/data/roi_sampling/segmenter.py pyproject.toml tests/roi_sampling/test_segmenter.py
git commit -m "feat(roi): SAM 3.1 segmenter wrapper + roi extra"
```

---

## Task 4: Static camera-calibration sidecar (offline)

**Files:**
- Create: `data_prep/robocasa365_to_lerobot/extract_camera_calib.py`
- Test: `tests/roi_sampling/test_camera_calib_io.py` (pure I/O round-trip only; env part is manual)

**Interfaces:**
- Produces a per-episode JSON sidecar at `<cache_dir>/camera_calib/episode_{idx:06d}.json` with, for `left` and `right`: `intrinsic` (3x3), `extrinsic` (cam2world 4x4), `image_hw` `[H, W]`. Also helpers `save_calib(path, calib: dict)` / `load_calib(path) -> dict` (numpy arrays).
- Consumes: `RoboCasa365Env.reset(initial_state_dir=..., step_after_reset=False)` and `_get_camera_matrices` (`environments.py:509`).

- [ ] **Step 1: Write the failing I/O round-trip test**

```python
# tests/roi_sampling/test_camera_calib_io.py
import numpy as np
from data_prep.robocasa365_to_lerobot.extract_camera_calib import save_calib, load_calib


def test_calib_roundtrip(tmp_path):
    calib = {
        "left": {"intrinsic": np.eye(3), "extrinsic": np.eye(4), "image_hw": [256, 256]},
        "right": {"intrinsic": np.eye(3) * 2, "extrinsic": np.eye(4), "image_hw": [256, 256]},
    }
    p = tmp_path / "episode_000000.json"
    save_calib(p, calib)
    got = load_calib(p)
    assert np.allclose(got["left"]["intrinsic"], np.eye(3))
    assert np.allclose(got["right"]["intrinsic"], np.eye(3) * 2)
    assert got["left"]["image_hw"] == [256, 256]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/roi_sampling/test_camera_calib_io.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Implement `extract_camera_calib.py`**

```python
# data_prep/robocasa365_to_lerobot/extract_camera_calib.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def save_calib(path: Path, calib: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = {cam: {"intrinsic": np.asarray(v["intrinsic"]).tolist(),
                 "extrinsic": np.asarray(v["extrinsic"]).tolist(),
                 "image_hw": list(v["image_hw"])}
           for cam, v in calib.items()}
    Path(path).write_text(json.dumps(out))


def load_calib(path: Path) -> dict:
    raw = json.loads(Path(path).read_text())
    return {cam: {"intrinsic": np.asarray(v["intrinsic"], dtype=np.float64),
                  "extrinsic": np.asarray(v["extrinsic"], dtype=np.float64),
                  "image_hw": list(v["image_hw"])}
            for cam, v in raw.items()}


def main() -> None:
    p = argparse.ArgumentParser(description="Dump static left/right camera calib per episode (no rollout).")
    p.add_argument("--input-dir", required=True, type=Path)
    p.add_argument("--cache-dir", required=True, type=Path)  # where camera_calib/ is written
    p.add_argument("--episodes", nargs="*", type=int, default=None)
    p.add_argument("--image-resolution", type=int, default=256)
    p.add_argument("--env-name", type=str, required=True)
    p.add_argument("--split", type=str, default="target")
    args = p.parse_args()

    from pointact.robot_envs.robocasa365_utils.environments import RoboCasa365Env
    env = RoboCasa365Env(env_name=args.env_name, split=args.split, image_resolution=args.image_resolution,
                         use_depth=True, use_point_cloud=False, enable_render=True, terminate_on_success=False)
    # RoboCASA agentview cameras are world-fixed; robosuite camera names for left/right:
    cam_names = {"left": "robot0_agentview_left", "right": "robot0_agentview_right"}  # verify at impl time
    try:
        for ep in args.episodes or []:
            ep_dir = args.input_dir / "extras" / f"episode_{ep:06d}"
            env.reset(initial_state_dir=ep_dir, step_after_reset=False)
            calib = {}
            for out_name, cam_name in cam_names.items():
                intr, extr = env._get_camera_matrices(cam_name)
                calib[out_name] = {"intrinsic": intr, "extrinsic": extr,
                                   "image_hw": [args.image_resolution, args.image_resolution]}
            save_calib(args.cache_dir / "camera_calib" / f"episode_{ep:06d}.json", calib)
    finally:
        env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run I/O test to verify it passes**

Run: `python -m pytest tests/roi_sampling/test_camera_calib_io.py -v`
Expected: PASS.

- [ ] **Step 5: Verify robosuite camera names + world-fixedness on one episode**

Run (GPU node) `extract_camera_calib.py` on episodes `0 1`, then assert the two episodes' `left` extrinsics are equal (world-fixed cameras) up to 1e-4. If they differ, cameras are not world-fixed → the ROI builder (Task 5) must read per-frame calib instead; note that finding. Fix `cam_names` to the actual robosuite names if the run errors on unknown camera.
Expected: extrinsics match across episodes; sample JSON written.

- [ ] **Step 6: Commit**

```bash
git add data_prep/robocasa365_to_lerobot/extract_camera_calib.py tests/roi_sampling/test_camera_calib_io.py
git commit -m "feat(roi): static camera-calibration sidecar extractor"
```

---

## Task 5: Offline ROI-flag LMDB builder

**Files:**
- Create: `data_prep/robocasa365_to_lerobot/build_roi_cache.py`
- Test: `tests/roi_sampling/test_roi_cache_io.py`

**Interfaces:**
- Consumes: `roi_sample`/`reproject`/`segmenter` (Tasks 1-3), `load_calib` (Task 4), the `points_3views` LMDB, and the LeRobot dataset's left/right image sequences + per-frame base pose (`state.base_position`/`state.base_rotation`).
- Produces: a parallel LMDB at `<root>/OpenDrawer/<roi_cache_dirname>` keyed `f"{ep}-{frame}"` → msgpack-packed `np.uint8` array of length `N` (1 = ROI), aligned to the stored cloud's point order. Helpers `pack_roi_flag(flag: np.ndarray) -> bytes` / `unpack_roi_flag(buf) -> np.ndarray`.

- [ ] **Step 1: Write the failing pack/unpack test**

```python
# tests/roi_sampling/test_roi_cache_io.py
import numpy as np
from data_prep.robocasa365_to_lerobot.build_roi_cache import pack_roi_flag, unpack_roi_flag


def test_roi_flag_roundtrip():
    flag = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
    out = unpack_roi_flag(pack_roi_flag(flag))
    assert out.dtype == np.uint8
    assert np.array_equal(out, flag)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/roi_sampling/test_roi_cache_io.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Implement `build_roi_cache.py`**

Core loop per episode: load left+right image sequences from the LeRobot dataset; `RoiSegmenter.segment_video` each → per-frame masks; `masks_are_usable` per frame; for each frame, read the stored cloud from `points_3views` LMDB, read per-frame `base_pos`/`base_quat` from the dataset state, build `cams` dicts from `load_calib` + that frame's masks, call `lift_masks_to_roi`; if a frame's masks are unusable, write an all-zero flag (dataloader treats all-zero as "uniform fallback"). Write `pack_roi_flag(roi.astype(uint8))` to the ROI LMDB under `f"{ep}-{frame}"`.

```python
# data_prep/robocasa365_to_lerobot/build_roi_cache.py  (key helpers shown; loop wired at impl time)
import msgpack, msgpack_numpy, numpy as np
msgpack_numpy.patch()


def pack_roi_flag(flag: np.ndarray) -> bytes:
    return msgpack.packb(np.ascontiguousarray(flag.astype(np.uint8)))


def unpack_roi_flag(buf) -> np.ndarray:
    return msgpack.unpackb(buf).astype(np.uint8)
```

The `main()` mirrors `replay.py`'s episode iteration and LMDB writing (see `convert.py` for the `lmdb.open(..., map_size=...)` + `txn.put(key.encode("ascii"), value)` pattern). Base quat ordering: `state.base_rotation` is xyzw (matches `R.from_quat` in `environments.py:428`).

- [ ] **Step 4: Run pack/unpack test to verify it passes**

Run: `python -m pytest tests/roi_sampling/test_roi_cache_io.py -v`
Expected: PASS.

- [ ] **Step 5: Build the ROI cache for a handful of episodes (integration smoke)**

Run (GPU node): `build_roi_cache.py` for episodes `0 1 2`. Assert every written flag length equals the corresponding stored cloud length, and that mean ROI fraction is in a sane range (e.g. 0.02–0.6) on frames with usable masks.
Expected: ROI LMDB entries created; length + fraction assertions pass.

- [ ] **Step 6: Commit**

```bash
git add data_prep/robocasa365_to_lerobot/build_roi_cache.py tests/roi_sampling/test_roi_cache_io.py
git commit -m "feat(roi): offline ROI-flag LMDB builder"
```

---

## Task 6: Visualization gate (interactive HTML + PLY + 2D overlays)

**Files:**
- Create: `pointact/data/roi_sampling/visualize.py`
- Create: `scripts/visualize_roi.py`
- Test: `tests/roi_sampling/test_visualize.py`

**Interfaces:**
- Produces:
  - `write_roi_html(points_xyz: np.ndarray, roi_mask: np.ndarray, out_path: str, sampled_idx: np.ndarray | None = None) -> None` — self-contained Plotly HTML (`include_plotlyjs=True`) with ROI points highlighted vs. background, and a second trace for the post-sample draw when `sampled_idx` is given.
  - `write_ply(points_xyz, colors_rgb, out_path) -> None`.
  - `write_mask_overlay(image, mask, out_path) -> None` (PNG via opencv).

- [ ] **Step 1: Write the failing tests (file creation + self-containment)**

```python
# tests/roi_sampling/test_visualize.py
import numpy as np
from pointact.data.roi_sampling.visualize import write_roi_html, write_ply, write_mask_overlay


def test_write_roi_html_is_self_contained(tmp_path):
    pts = np.random.default_rng(0).normal(size=(100, 3))
    roi = np.zeros(100, dtype=bool); roi[:30] = True
    out = tmp_path / "roi.html"
    write_roi_html(pts, roi, str(out), sampled_idx=np.arange(40))
    html = out.read_text()
    assert out.exists() and len(html) > 10000
    assert "plotly" in html.lower()  # bundled JS, opens offline


def test_write_ply_and_overlay(tmp_path):
    pts = np.random.default_rng(0).normal(size=(10, 3))
    cols = np.zeros((10, 3), dtype=np.uint8)
    write_ply(pts, cols, str(tmp_path / "c.ply"))
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=bool); mask[10:20, 10:20] = True
    write_mask_overlay(img, mask, str(tmp_path / "o.png"))
    assert (tmp_path / "c.ply").exists() and (tmp_path / "o.png").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/roi_sampling/test_visualize.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Implement `visualize.py` (Plotly HTML, PLY writer, overlay) and `scripts/visualize_roi.py`**

`write_roi_html` uses `plotly.graph_objects.Scatter3d` (background gray, ROI red, sampled draw as a third trace), `fig.write_html(out_path, include_plotlyjs=True, full_html=True)`. `write_ply` writes an ASCII PLY (or via open3d, already a dep). `write_mask_overlay` blends `mask` in red over `image` and `cv2.imwrite`s. `scripts/visualize_roi.py` takes `--root --repo-id OpenDrawer --roi-cache-dirname --episodes --frames --out-dir`, loads stored cloud + ROI flag + images, applies the guarded split (`roi_guided_indices`), and writes all three artifacts per selected frame.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/roi_sampling/test_visualize.py -v`
Expected: PASS.

- [ ] **Step 5: Generate the gate artifacts + HUMAN GO/NO-GO**

Run: `scripts/visualize_roi.py` on episodes `0 1 2`, a few frames each (early/approach/contact). Open the HTML locally; confirm the drawer/handle is densified and correctly localized, 2D overlays look right, and reprojection has no flip/sign error (ROI points sit on the drawer, not mirrored). **Do not proceed to Task 7-9 until a human approves.** If reprojection is mirrored, toggle/inspect `flip_v`.
Expected: human approval recorded.

- [ ] **Step 6: Commit**

```bash
git add pointact/data/roi_sampling/visualize.py scripts/visualize_roi.py tests/roi_sampling/test_visualize.py
git commit -m "feat(roi): interactive HTML/PLY/overlay visualization gate"
```

---

## Task 7: Dataloader integration (training path)

**Files:**
- Modify: `pointact/data/robot/data_3d.py` (`__init__` ~L34-97, `filter_point_cloud_by_workspace` L174-187, `augment_point_cloud` L208-237, add `load_roi_flag`)
- Modify: `experiments/13_robocasa365/data_configs/data-robocasa365-opendrawer-point.yaml`
- Test: `tests/roi_sampling/test_dataloader_split.py`

**Interfaces:**
- Consumes: `roi_guided_indices` (Task 1), the ROI LMDB (Task 5).
- Produces: when `roi_cache_dirname` is set, `augment_point_cloud` uses `roi_guided_indices` with the config `roi_ratio`; when unset, behavior is byte-for-byte the baseline. New `__init__` kwargs: `roi_cache_dirname: str | None = None`, `roi_ratio: float = 0.6`.

- [ ] **Step 1: Write the failing test (behavior parity + ROI biasing on a synthetic subclass)**

```python
# tests/roi_sampling/test_dataloader_split.py
import numpy as np
from pointact.data.roi_sampling.roi_sample import roi_guided_indices

# Unit-level guard: the dataloader delegates selection to roi_guided_indices with
# N computed by the baseline rule. This test locks the N rule + ratio wiring the
# dataloader must use (mirrored from augment_point_cloud).

def _baseline_n(n_points, cap, u):
    return min(int(n_points * u), cap)

def test_dataloader_selection_contract():
    rng = np.random.default_rng(0)
    n_points, cap = 6000, 4096
    n = _baseline_n(n_points, cap, u=0.9)   # 5400 -> capped 4096
    assert n == 4096
    roi = np.zeros(n_points, dtype=bool); roi[:1000] = True
    idx = roi_guided_indices(roi, n_total=n, roi_ratio=0.6, rng=rng)
    assert idx.shape == (4096,)
    # ROI over-represented vs. its 1000/6000 base rate
    assert roi[idx].mean() > 1000 / 6000
```

- [ ] **Step 2: Run test to verify it fails/passes appropriately**

Run: `python -m pytest tests/roi_sampling/test_dataloader_split.py -v`
Expected: PASS once Task 1 exists (this test guards the contract the edit must honor). If it fails, Task 1 is broken — fix there.

- [ ] **Step 3: Add ROI loading + guarded split to `data_3d.py`**

In `__init__`, store `self.roi_cache_dirname`, `self.roi_ratio`, and (mirroring the point LMDB handles) lazy per-pid handles for the ROI LMDB when `roi_cache_dirname` is not None. Add `load_roi_flag(ep_idx, frame_idx) -> np.ndarray | None` (returns None if key missing → uniform fallback). Refactor `filter_point_cloud_by_workspace` to also accept/return an aligned flag (or compute the boolean mask once and apply to both cloud and flag in `__getitem__`). In `augment_point_cloud`, keep the exact `max_npoints` line, then:

```python
# inside augment_point_cloud, replacing the np.random.choice block:
n_total = max_npoints
if roi_flag is not None and roi_flag.any() and len(point_cloud) > n_total:
    rng = np.random.default_rng()  # match existing global-random semantics
    ridxs = roi_guided_indices(roi_flag.astype(bool), n_total, self.roi_ratio, rng)
    point_cloud = point_cloud[ridxs]
elif len(point_cloud) > n_total:
    ridxs = np.random.choice(len(point_cloud), n_total, replace=False)
    point_cloud = point_cloud[ridxs]
```

Thread `roi_flag` from `__getitem__` (load it, apply the same workspace mask). **Critical:** the workspace mask must be applied identically to `point_cloud` and `roi_flag` so indices stay aligned.

- [ ] **Step 4: Run the full roi test suite**

Run: `python -m pytest tests/roi_sampling -v`
Expected: PASS.

- [ ] **Step 5: Wire the config**

In `data-robocasa365-opendrawer-point.yaml` add under the `OpenDrawer` entry:
```yaml
    roi_cache_dirname: points_3views_roi   # parallel LMDB from build_roi_cache.py
    roi_ratio: 0.6
```
Confirm `LeRobotPointCloudDataset.__init__` accepts these (added in Step 3) and the config loader forwards unknown keys (they already flow via `**kwargs`).

- [ ] **Step 6: Commit**

```bash
git add pointact/data/robot/data_3d.py experiments/13_robocasa365/data_configs/data-robocasa365-opendrawer-point.yaml tests/roi_sampling/test_dataloader_split.py
git commit -m "feat(roi): ROI-guided sampling in the training dataloader"
```

---

## Task 8: Online eval integration (processor)

**Files:**
- Modify: `pointact/model/backbone/processor_base.py` (`_prepare_point_cloud_for_sample` L289-308, add `_roi_subsample_point_cloud`)
- Modify: `pointact/model/vla_pointact/processing_vla_pointact.py` (`select_action` L176, `_prepare_robot_inputs` L85 — pass ROI config + ensure left/right images + camera matrices reach the mini_batch)
- Test: `tests/roi_sampling/test_online_hook.py`

**Interfaces:**
- Consumes: `RoiSegmenter` (Task 3), `lift_masks_to_roi` (Task 2), `roi_guided_indices` (Task 1).
- Produces: when ROI is enabled on the processor, `_prepare_point_cloud_for_sample` replaces `_subsample_point_cloud` with segment(left,right)→lift→`roi_guided_indices`; empty/failed masks fall back to the existing uniform `_subsample_point_cloud`. A single `RoiSegmenter` is cached on the processor instance.

- [ ] **Step 1: Write the failing test (fallback path, no SAM3/GPU)**

```python
# tests/roi_sampling/test_online_hook.py
import numpy as np
from pointact.data.roi_sampling.roi_sample import roi_guided_indices

def test_online_falls_back_to_uniform_on_empty_mask():
    # Contract: all-False ROI => roi_guided_indices behaves like a uniform draw.
    rng = np.random.default_rng(0)
    roi = np.zeros(500, dtype=bool)
    idx = roi_guided_indices(roi, n_total=256, roi_ratio=0.6, rng=rng)
    assert idx.shape == (256,)
    assert len(np.unique(idx)) == 256
```

- [ ] **Step 2: Run test to verify it passes (guards the fallback contract)**

Run: `python -m pytest tests/roi_sampling/test_online_hook.py -v`
Expected: PASS.

- [ ] **Step 3: Implement the online ROI path**

Add to `processor_base.py`:
```python
def _roi_subsample_point_cloud(self, point_cloud, mini_batch, repo_id, base_pos, base_quat, cams):
    from pointact.data.roi_sampling.reproject import lift_masks_to_roi
    from pointact.data.roi_sampling.roi_sample import roi_guided_indices
    n_total = self.robot_config["max_npoints"][repo_id]
    if len(point_cloud) <= n_total or not cams:
        return self._subsample_point_cloud(point_cloud, n_total)
    roi = lift_masks_to_roi(point_cloud[:, :3], base_pos, base_quat, cams)
    if not roi.any():
        return self._subsample_point_cloud(point_cloud, n_total)
    rng = np.random.default_rng()
    idx = roi_guided_indices(roi, n_total, self._roi_ratio, rng)
    return point_cloud[idx]
```
In `_prepare_point_cloud_for_sample`, when `self._roi_enabled`, build `cams` from `mini_batch["observation.images.{left,right}"]` via the cached `RoiSegmenter.segment_image` + `mini_batch["observation.camera_intrinsics/extrinsics.{left,right}"]`, read base pose from the state, and call `_roi_subsample_point_cloud` instead of `_subsample_point_cloud` (line 307). Gate everything behind `self._roi_enabled` (default False) so non-ROI models are untouched. Add processor attrs `_roi_enabled`, `_roi_ratio`, `_roi_segmenter` set from config/kwargs in `select_action`.

- [ ] **Step 4: Ensure eval batch carries left/right images + camera matrices**

In `processing_vla_pointact.py::select_action` / the rollout obs assembly, confirm `observation.images.left_image`, `observation.images.right_image`, and `observation.camera_intrinsics/extrinsics.left/right` are present in `batch` (the env `get_observation` produces them; the eval client must forward them). Add them to the forwarded obs if missing. Document this in the eval config.

- [ ] **Step 5: Run the roi suite**

Run: `python -m pytest tests/roi_sampling -v`
Expected: PASS.

- [ ] **Step 6: Live smoke test (GPU node, one short rollout)**

Run a single `openDrawer` eval episode with ROI enabled; confirm no crash, per-frame latency is logged, and VRAM fits alongside the policy. If SAM 3.1 is too heavy, record the number and trigger the documented Grounding-DINO-tiny + SAM2.1-tiny fallback (spec Section 1).
Expected: rollout completes; latency/VRAM recorded.

- [ ] **Step 7: Commit**

```bash
git add pointact/model/backbone/processor_base.py pointact/model/vla_pointact/processing_vla_pointact.py tests/roi_sampling/test_online_hook.py
git commit -m "feat(roi): online ROI-guided sampling in eval processor"
```

---

## Task 9: A/B training run wiring + docs

**Files:**
- Modify: `experiments/13_robocasa365/data_configs/data-robocasa365-opendrawer-point.yaml` (already has ROI keys; add a commented baseline variant note)
- Create: `docs/superpowers/plans/roi-ab-runbook.md`

**Interfaces:** none (operational).

- [ ] **Step 1: Write the A/B runbook**

Document: (1) build the ROI cache (`extract_camera_calib.py` → `build_roi_cache.py`) for the full openDrawer split; (2) baseline run = config with `roi_cache_dirname` unset (identical to current); (3) ROI run = same config + ROI keys; **all other hyperparameters, seeds, and steps identical**; (4) eval both with matching processor ROI flags (baseline eval uniform, ROI eval with `_roi_enabled=True`); (5) compare `openDrawer` success rate. Include the exact train/eval commands used in this repo.

- [ ] **Step 2: Full ROI cache build**

Run `extract_camera_calib.py` then `build_roi_cache.py` over all openDrawer episodes. Assert coverage: every `{ep}-{frame}` key in `points_3views` has a corresponding key in `points_3views_roi`.
Expected: complete cache; coverage assertion passes.

- [ ] **Step 3: Launch A/B (operational — outside test scope)**

Kick off baseline + ROI training with identical configs save for the ROI keys. This is the experiment, not a unit test.

- [ ] **Step 4: Commit the runbook**

```bash
git add docs/superpowers/plans/roi-ab-runbook.md experiments/13_robocasa365/data_configs/data-robocasa365-opendrawer-point.yaml
git commit -m "docs(roi): A/B runbook for openDrawer ROI-sampling PoC"
```

---

## Self-Review

**Spec coverage:**
- Perception (SAM 3.1, left/right, text prompt) → Task 3, used in Tasks 5 & 8. ✓
- Lift via reprojection (camera matrices at both stages; base pose; image flip) → Task 2 (geometry), Task 4 (offline calib), Task 8 (online obs). ✓
- Count-matched guarded 60/40 split, `roi_ratio` knob → Task 1, wired in Tasks 7 & 8. ✓
- Wrist points kept (geometric labeling, not per-camera subset) → inherent in Task 2 `lift_masks_to_roi` (operates on the full merged cloud). ✓
- Offline parallel LMDB + dataloader swap, workspace-filter lockstep → Task 5, Task 7. ✓
- Online eval path (path A), never-worse fallback → Task 8. ✓
- Visualization gate (interactive HTML + PLY + overlays) before retraining → Task 6 (hard human gate). ✓
- Frozen everything else, A/B → Task 9. ✓
- Open item "measure SAM 3.1 VRAM/latency; fallback" → Task 8 Step 6. ✓
- Open item "SAM 3.1 weights availability" → Task 3 Step 6. ✓
- Open item "storage format" → Task 5 (uint8 msgpack). ✓
- Open item "eval insertion point" → Task 8 (resolved: `_prepare_point_cloud_for_sample`). ✓

**Placeholder scan:** SAM 3.1 exact import/call is intentionally deferred to Task 3 Step 6 (verify-on-machine) and robosuite camera names to Task 4 Step 5 — both are explicit verification steps with acceptance criteria, not silent TODOs. All pure modules have complete code + tests.

**Type consistency:** `roi_guided_indices(roi_mask, n_total, roi_ratio, rng)`, `lift_masks_to_roi(points_rb, base_pos, base_quat_xyzw, cams, flip_v=True)`, `pack/unpack_roi_flag`, `save/load_calib`, `write_roi_html/write_ply/write_mask_overlay` — names/signatures used consistently across Tasks 5, 7, 8. `cams` dict schema (`intrinsic`/`extrinsic`/`mask`/`image_hw`) identical in Tasks 2, 5, 8.

**Risk note:** Task 4 Step 5 tests the world-fixed-camera assumption; if false, Task 5 must read per-frame calib (fallback documented in that step). This is the only architectural assumption not provable from static code.
