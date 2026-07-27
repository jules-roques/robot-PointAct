"""Geometry for ROI-guided point sampling (RoboCASA OpenDrawer PoC).

The stored point clouds live in the **robot-base frame** (see
``RoboCasa365Env.convert_points_to_robot_base_frame``). To decide which points fall
inside a 2D detection box we reproject base-frame points into a camera image, then
build a 3D "halo" (a ball around the detected object's epicenter) and flag points
inside it as region-of-interest (ROI).

Reprojection is the exact inverse of the forward construction in
``pointact.utils.depth.depth_to_point_cloud`` composed with the base-frame
conversion:

    forward:  P_world = cam2world @ P_cam ;  P_base = R_base^{-1} (P_world - base_pos)
    inverse:  P_world = R_base P_base + base_pos ;  P_cam = world2cam @ P_world
              u = fx * Xc/Zc + cx ;  v = fy * Yc/Zc + cy   (OpenCV, +Z forward)

Because the agentview cameras are rigidly mounted to the robot, ``base2cam`` (and the
intrinsics) are constant across OpenDrawer episodes, so a single calibration is
reused for the whole dataset. The stored RGB frames and the depth used to build the
clouds are both vertically flipped (``[::-1]``) consistently, so a box measured on a
stored frame shares row indexing with the reprojected ``v`` — no extra flip needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def base2cam_from_extrinsic(
    cam2world: np.ndarray, base_pos: np.ndarray, base_quat_xyzw: np.ndarray
) -> np.ndarray:
    """Compose the constant base->camera 4x4 transform.

    Args:
        cam2world: (4, 4) camera-to-world matrix (robosuite extrinsic).
        base_pos: (3,) robot base position in world.
        base_quat_xyzw: (4,) robot base orientation quaternion (xyzw).

    Returns:
        (4, 4) base->camera transform.
    """
    from scipy.spatial.transform import Rotation as R

    cam2world = np.asarray(cam2world, dtype=np.float64).reshape(4, 4)
    base2world = np.eye(4, dtype=np.float64)
    base2world[:3, :3] = R.from_quat(np.asarray(base_quat_xyzw, dtype=np.float64)).as_matrix()
    base2world[:3, 3] = np.asarray(base_pos, dtype=np.float64).reshape(3)
    world2cam = np.linalg.inv(cam2world)
    return (world2cam @ base2world).astype(np.float64)


def project_base_points(
    points_base: np.ndarray, intrinsic: np.ndarray, base2cam: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project base-frame points into a camera image.

    Args:
        points_base: (N, 3) points in the robot-base frame.
        intrinsic: (3, 3) camera intrinsic matrix.
        base2cam: (4, 4) base->camera transform.

    Returns:
        uv: (N, 2) float pixel coordinates (col=u, row=v).
        z_cam: (N,) depth along the camera +Z axis.
        in_front: (N,) bool, True where z_cam > 0.
    """
    points_base = np.asarray(points_base, dtype=np.float64)
    n = len(points_base)
    homo = np.concatenate([points_base, np.ones((n, 1))], axis=1)  # (N, 4)
    p_cam = (base2cam @ homo.T).T[:, :3]  # (N, 3)
    z = p_cam[:, 2]
    in_front = z > 1e-6
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    safe_z = np.where(in_front, z, 1.0)
    u = fx * p_cam[:, 0] / safe_z + cx
    v = fy * p_cam[:, 1] / safe_z + cy
    uv = np.stack([u, v], axis=1)
    return uv, z, in_front


def points_in_box(
    points_base: np.ndarray,
    intrinsic: np.ndarray,
    base2cam: np.ndarray,
    box_xyxy: np.ndarray,
    image_hw: tuple[int, int],
) -> np.ndarray:
    """Boolean mask of base-frame points whose projection lands inside a 2D box.

    Args:
        box_xyxy: (4,) [x0, y0, x1, y1] in pixels (stored-frame convention).
        image_hw: (H, W) of the frame the box was measured on.
    """
    h, w = image_hw
    uv, _z, in_front = project_base_points(points_base, intrinsic, base2cam)
    x0, y0, x1, y1 = box_xyxy
    u, v = uv[:, 0], uv[:, 1]
    inside = (
        in_front
        & (u >= x0) & (u <= x1)
        & (v >= y0) & (v <= y1)
        & (u >= 0) & (u < w)
        & (v >= 0) & (v < h)
    )
    return inside


def halo_weights(
    points_base: np.ndarray,
    anchor: np.ndarray,
    radius: float,
    mode: str = "hard",
    softness: float = 1.0,
) -> np.ndarray:
    """Per-point ROI weight from a stored (anchor, radius) halo.

    Kept separate from halo construction so radius scaling and hard/soft selection are
    *dataloader* knobs: the cache stores only the anchor and radius, so these can change
    without rebuilding anything.

    Args:
        mode: "hard" -> 1.0 inside the ball, 0.0 outside (sharp boundary, exact counts).
              "soft" -> Gaussian falloff exp(-d^2 / (2 sigma^2)) with sigma = softness*radius,
              which removes the boundary artifact and down-weights the periphery smoothly.
        softness: sigma as a multiple of radius (soft mode only).

    Returns:
        (N,) float weights in [0, 1].
    """
    d = np.linalg.norm(np.asarray(points_base, dtype=np.float64) - np.asarray(anchor, dtype=np.float64), axis=1)
    if mode == "hard":
        return (d <= radius).astype(np.float64)
    if mode == "soft":
        sigma = max(1e-6, softness * radius)
        return np.exp(-0.5 * (d / sigma) ** 2)
    raise ValueError(f"unknown halo mode {mode!r} (expected 'hard' or 'soft')")


@dataclass
class HaloResult:
    """Outcome of building a halo for one frame."""

    roi_mask: np.ndarray  # (N,) bool over the input cloud
    anchor: np.ndarray | None  # (3,) base-frame epicenter, or None
    radius: float  # halo radius (meters), 0.0 if no detection
    n_in_box: int  # number of points that fell inside any detection box


def select_box_by_side(boxes: np.ndarray, side: str | None) -> np.ndarray:
    """Pick the drawer box matching the instructed side.

    RoboCASA OpenDrawer instructions name the target ("Open the left drawer." /
    "Open the right drawer."), but an open-vocab "drawer" prompt fires on every drawer
    in view — so the most-confident box is often the wrong one. Disambiguate by image
    x-position: leftmost box center for "left", rightmost for "right". The instruction
    is available at training AND at eval, so this works in both paths.

    Args:
        boxes: (K, 4) xyxy candidate boxes (any order).
        side: "left", "right", or None (no disambiguation -> return all boxes).

    Returns:
        (1, 4) chosen box, or the input when side is None / boxes is empty.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if side is None or len(boxes) <= 1:
        return boxes
    centers_x = 0.5 * (boxes[:, 0] + boxes[:, 2])
    idx = int(np.argmin(centers_x)) if side == "left" else int(np.argmax(centers_x))
    return boxes[idx : idx + 1]


def parse_task_side(task: str | None) -> str | None:
    """Extract the target side from a RoboCASA OpenDrawer instruction."""
    if not task:
        return None
    t = task.lower()
    if "left" in t:
        return "left"
    if "right" in t:
        return "right"
    return None


def build_halo_roi(
    points_base: np.ndarray,
    cameras: list[dict],
    detections: dict[str, np.ndarray],
    image_hw: tuple[int, int],
    halo_scale: float = 2.0,
    min_radius: float = 0.05,
    max_radius: float = 0.6,
    min_in_box: int = 20,
) -> HaloResult:
    """Compute a per-point ROI mask from 2D detections via an epicenter + halo.

    Args:
        points_base: (N, 3) cloud points in robot-base frame.
        cameras: list of {"name", "intrinsic" (3,3), "base2cam" (4,4)} dicts.
        detections: {camera_name: (K, 4) boxes xyxy} for the same frame. A camera
            with no detection may be absent or map to an empty array.
        image_hw: (H, W) of the frames the boxes were measured on.
        halo_scale: radius = halo_scale * robust_spread(in_box points).
        min_radius, max_radius: clamp on the halo radius (meters).
        min_in_box: if fewer than this many points fall in all boxes combined, the
            detection is considered unreliable and no ROI is returned (caller falls
            back to uniform sampling).

    Returns:
        HaloResult. ``roi_mask`` is all-False and ``anchor`` is None when detection
        is missing/unreliable.
    """
    n = len(points_base)
    in_box_any = np.zeros(n, dtype=bool)
    for cam in cameras:
        boxes = detections.get(cam["name"])
        if boxes is None or len(boxes) == 0:
            continue
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        for box in boxes:
            in_box_any |= points_in_box(
                points_base, cam["intrinsic"], cam["base2cam"], box, image_hw
            )

    n_in_box = int(in_box_any.sum())
    if n_in_box < min_in_box:
        return HaloResult(np.zeros(n, dtype=bool), None, 0.0, n_in_box)

    in_box_pts = np.asarray(points_base, dtype=np.float64)[in_box_any]
    anchor = np.median(in_box_pts, axis=0)  # robust epicenter
    # Robust spread: RMS distance to the anchor.
    spread = float(np.sqrt(np.mean(np.sum((in_box_pts - anchor) ** 2, axis=1))))
    radius = float(np.clip(halo_scale * spread, min_radius, max_radius))

    dists = np.linalg.norm(np.asarray(points_base, dtype=np.float64) - anchor, axis=1)
    roi_mask = dists <= radius
    return HaloResult(roi_mask, anchor.astype(np.float32), radius, n_in_box)


if __name__ == "__main__":
    # Self-test: build points via the FORWARD mapping (cam -> base) so the reprojection
    # round-trip is well-posed, then verify pixels are recovered and the halo selects a
    # spatially compact region around a planted cluster.
    from scipy.spatial.transform import Rotation as R

    rng = np.random.default_rng(0)
    H = W = 256
    fx = fy = 200.0
    cx = cy = 128.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    cam2world = np.eye(4)
    cam2world[:3, :3] = R.from_euler("xyz", [20, 30, 10], degrees=True).as_matrix()
    cam2world[:3, 3] = np.array([1.5, 0.3, 0.7])
    base_pos = np.array([0.2, -0.1, 0.0])
    base_quat = R.from_euler("z", 25, degrees=True).as_quat()  # xyzw
    b2c = base2cam_from_extrinsic(cam2world, base_pos, base_quat)
    cam2base = np.linalg.inv(b2c)

    def cam_to_base(p_cam):
        homo = np.concatenate([p_cam, np.ones((len(p_cam), 1))], axis=1)
        return (cam2base @ homo.T).T[:, :3]

    # Object: a compact cluster at a known image location and depth.
    obj_uv = np.array([160.0, 100.0])
    obj_z = 0.8
    obj_cam = np.zeros((300, 3))
    obj_cam[:, 2] = obj_z
    obj_cam[:, 0] = (obj_uv[0] - cx) * obj_z / fx
    obj_cam[:, 1] = (obj_uv[1] - cy) * obj_z / fy
    obj_cam += 0.01 * rng.standard_normal((300, 3))
    # Background: spread across the frustum at varied depths.
    bg_cam = np.zeros((3000, 3))
    bg_cam[:, 2] = rng.uniform(0.4, 1.5, 3000)
    bg_cam[:, 0] = (rng.uniform(0, W, 3000) - cx) * bg_cam[:, 2] / fx
    bg_cam[:, 1] = (rng.uniform(0, H, 3000) - cy) * bg_cam[:, 2] / fy

    obj_base = cam_to_base(obj_cam)
    bg_base = cam_to_base(bg_cam)
    pts = np.concatenate([obj_base, bg_base], axis=0)

    uv, z, front = project_base_points(pts, K, b2c)
    assert front[:300].all(), "object points must project in front of camera"
    recovered = uv[:300].mean(axis=0)
    assert np.allclose(recovered, obj_uv, atol=2.0), f"pixel round-trip off: {recovered} vs {obj_uv}"

    pad = 4
    ouv = uv[:300]
    box = np.array([ouv[:, 0].min() - pad, ouv[:, 1].min() - pad,
                    ouv[:, 0].max() + pad, ouv[:, 1].max() + pad])
    cams = [{"name": "left", "intrinsic": K, "base2cam": b2c}]
    res = build_halo_roi(pts, cams, {"left": box[None]}, (H, W),
                         halo_scale=2.0, min_in_box=10)
    frac_obj = res.roi_mask[:300].mean()
    frac_bg = res.roi_mask[300:].mean()
    print(f"pixel round-trip OK ({recovered} ~= {obj_uv})")
    print(f"n_in_box={res.n_in_box} radius={res.radius:.3f} "
          f"obj_captured={frac_obj:.2f} bg_captured={frac_bg:.2f}")
    assert frac_obj > 0.9, "halo should capture most object points"
    assert frac_bg < 0.5, "halo should be selective vs background"
    print("geometry self-test OK")
