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


def eef_density_weights(
    points_xyz: np.ndarray,
    eef_pos: np.ndarray,
    sigma: float,
    floor: float,
) -> np.ndarray:
    """Per-point sampling weight: a Gaussian bump around the end-effector with a density floor.

    ``weight = floor + (1 - floor) * exp(-d^2 / (2 sigma^2))``, so every point keeps a
    non-zero weight (background is never starved) while points within ~sigma of the eef
    dominate the budget. Unlike :func:`halo_weights`, there is no radius/box detection step:
    ``eef_pos`` comes straight from the frame's proprioceptive state, so this needs no cache.

    Args:
        points_xyz: (N, 3) cloud points, same frame as ``eef_pos`` (robot-base frame).
        eef_pos: (3,) end-effector position.
        sigma: Gaussian bandwidth (meters).
        floor: minimum weight at infinite distance, in [0, 1).

    Returns:
        (N,) float weights in [floor, 1].
    """
    d = np.linalg.norm(np.asarray(points_xyz, dtype=np.float64) - np.asarray(eef_pos, dtype=np.float64), axis=1)
    sigma = max(1e-6, float(sigma))
    gauss = np.exp(-0.5 * (d / sigma) ** 2)
    return floor + (1.0 - floor) * gauss


@dataclass
class HaloResult:
    """Outcome of building a halo for one frame."""

    roi_mask: np.ndarray  # (N,) bool over the input cloud
    anchor: np.ndarray | None  # (3,) base-frame epicenter, or None
    radius: float  # halo radius (meters), 0.0 if no detection
    n_in_box: int  # number of points that fell inside any detection box


#: Weak statistical tendency only — DO NOT use as a per-episode rule.
#: Over 80 episodes the eef-at-grasp averages Y=+0.156 for "left" and Y=-0.148 for
#: "right", but individual episodes contradict it freely (left-drawer episodes grasp at
#: Y=-0.607, -0.459, +0.473, ...): the robot parks at a different yaw in each kitchen, so
#: no fixed base-frame axis — and no image-space side — decodes the instruction.
#: Selection therefore uses the demonstration's grasp point (see select_anchor_by_grasp).
LEFT_IS_HIGHER_Y = True


def select_box_by_side(boxes: np.ndarray, side: str | None) -> np.ndarray:
    """Deprecated image-space selector, kept only for the geometry self-test.

    Use :func:`select_anchor_by_side`, which disambiguates in 3D. Image-space x is
    camera-dependent and empirically selects the wrong drawer on this dataset.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if side is None or len(boxes) <= 1:
        return boxes
    centers_x = 0.5 * (boxes[:, 0] + boxes[:, 2])
    idx = int(np.argmin(centers_x)) if side == "left" else int(np.argmax(centers_x))
    return boxes[idx : idx + 1]


def candidate_anchors(
    points_base: np.ndarray,
    cameras: list[dict],
    detections: dict[str, np.ndarray],
    image_hw: tuple[int, int],
    min_in_box: int = 20,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """One 3D anchor per detected box, so candidates can be compared in the world.

    Returns:
        List of (anchor_xyz, in_box_mask) for every box with enough support.
    """
    out = []
    for cam in cameras:
        boxes = detections.get(cam["name"])
        if boxes is None or len(boxes) == 0:
            continue
        for box in np.asarray(boxes, dtype=np.float64).reshape(-1, 4):
            mask = points_in_box(points_base, cam["intrinsic"], cam["base2cam"], box, image_hw)
            if int(mask.sum()) < min_in_box:
                continue
            anchor = np.median(np.asarray(points_base, dtype=np.float64)[mask], axis=0)
            out.append((anchor, mask))
    return out


def select_anchor_by_grasp(
    cands: list[tuple[np.ndarray, np.ndarray]],
    grasp_point: np.ndarray | None,
    max_dist: float = 0.45,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Choose the detected drawer the demonstration actually grasped.

    ``grasp_point`` is the end-effector position (robot-base frame, i.e. the point-cloud
    frame) at the frame the gripper closes — the handle the demonstration reached for.
    It is a per-episode constant, so it disambiguates every frame of that episode,
    including the first, without making the ROI follow the arm around.

    This replaces instruction-side heuristics, which do not work on this dataset: the
    robot parks at a different yaw per kitchen, so neither a base-frame axis nor an
    image side maps consistently onto "left"/"right" (see LEFT_IS_HIGHER_Y).

    When a grasp point is available but no candidate lies within ``max_dist`` of it, the
    detector did not find the target drawer — so this returns None and the caller falls
    back to uniform sampling. Guessing the best-supported candidate there would
    concentrate the budget on the *wrong* drawer, which is worse than not concentrating
    at all; uniform keeps such frames merely baseline-equivalent.

    With no grasp point at all (e.g. eval time) it returns the best-supported candidate.
    """
    if not cands:
        return None
    if grasp_point is None:
        return max(cands, key=lambda c: int(c[1].sum()))
    g = np.asarray(grasp_point, dtype=np.float64).reshape(3)
    dists = np.array([float(np.linalg.norm(c[0] - g)) for c in cands])
    best = int(np.argmin(dists))
    if dists[best] > max_dist:
        return None
    return cands[best]


def select_anchor_by_side(
    cands: list[tuple[np.ndarray, np.ndarray]], side: str | None
) -> tuple[np.ndarray, np.ndarray] | None:
    """Choose the candidate drawer named by the instruction, comparing in 3D.

    "left" -> the higher-base-Y candidate, "right" -> the lower (see LEFT_IS_HIGHER_Y).
    The comparison is relative among this frame's candidates, so it does not depend on
    the robot's absolute base pose or on which camera saw the drawer.
    """
    if not cands:
        return None
    if side is None or len(cands) == 1:
        # No side information: fall back to the best-supported candidate.
        return max(cands, key=lambda c: int(c[1].sum()))
    ys = np.array([c[0][1] for c in cands])
    want_max = (side == "left") == LEFT_IS_HIGHER_Y
    return cands[int(np.argmax(ys)) if want_max else int(np.argmin(ys))]


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


def halo_from_candidate(
    points_base: np.ndarray,
    cand: tuple[np.ndarray, np.ndarray] | None,
    halo_scale: float = 1.0,
    min_radius: float = 0.02,
    max_radius: float = 0.25,
) -> HaloResult:
    """Size a halo around an already-selected candidate (anchor, in_box_mask)."""
    n = len(points_base)
    if cand is None:
        return HaloResult(np.zeros(n, dtype=bool), None, 0.0, 0)
    anchor, mask = cand
    pts = np.asarray(points_base, dtype=np.float64)
    in_box_pts = pts[mask]
    spread = float(np.sqrt(np.mean(np.sum((in_box_pts - anchor) ** 2, axis=1))))
    radius = float(np.clip(halo_scale * spread, min_radius, max_radius))
    roi = np.linalg.norm(pts - anchor, axis=1) <= radius
    return HaloResult(roi, anchor.astype(np.float32), radius, int(mask.sum()))


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
