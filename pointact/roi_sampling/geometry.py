"""Geometry for ROI-guided point sampling on RoboCASA.

The stored point clouds live in the **robot-base frame** (see
``RoboCasa365Env.convert_points_to_robot_base_frame``). A 2D detector (MolmoPoint, see
``data_prep/roi_sampling/build_molmo_cache.py``) names a pixel; to turn that into a 3D
anchor we reproject base-frame points into the camera image and take the median of the
points landing in a small window around it. Lifting this way rather than reading the
depth map at that pixel means invalid-depth pixels reduce support instead of corrupting
the anchor.

The anchor then centres the Gaussian-with-floor density of :func:`eef_density_weights`,
the same one the ``eef`` and ``oracle`` arms use — so those arms differ only in where
the bump sits. An earlier YOLO-World + "halo" (hard/soft ball) variant lived here and
was removed: it could not tell which drawer an instruction named, which forced a
privileged grasp point in as a disambiguator.

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


def eef_density_weights(
    points_xyz: np.ndarray,
    anchors: np.ndarray,
    sigma: float,
    floor: float,
) -> np.ndarray:
    """Per-point sampling weight: a Gaussian bump around one or more anchors, with a floor.

    ``weight = floor + (1 - floor) * max_k exp(-d_k^2 / (2 sigma^2))``, so every point keeps
    a non-zero weight (background is never starved) while points within ~sigma of *any*
    anchor dominate the budget.

    The ``max`` over anchors — rather than a sum — keeps the peak weight at 1.0 no matter how
    many anchors there are, so adding a second region of interest redistributes the budget
    instead of inflating it. With a single anchor this reduces **exactly** to the original
    one-centre form, which is what keeps the ``eef`` and ``oracle`` arms bit-identical to the
    stage 0-2 runs they are compared against.

    Despite the name, the anchor need not be the end-effector: the ``eef`` arm passes
    ``state[:3]``, ``oracle`` passes the ground-truth handle centroid, and the MolmoPoint arm
    passes one or two cached detections.

    Args:
        points_xyz: (N, 3) cloud points, same frame as the anchors (robot-base frame).
        anchors: (3,) or (K, 3) anchor position(s).
        sigma: Gaussian bandwidth (meters).
        floor: minimum weight at infinite distance, in [0, 1).

    Returns:
        (N,) float weights in [floor, 1].
    """
    pts = np.asarray(points_xyz, dtype=np.float64)
    a = np.atleast_2d(np.asarray(anchors, dtype=np.float64))
    if a.shape[-1] != 3:
        raise ValueError(f"anchors must be (3,) or (K, 3), got {a.shape}")
    sigma = max(1e-6, float(sigma))
    # (K, N) distances; max over K collapses to the nearest anchor's bump.
    d = np.linalg.norm(pts[None, :, :] - a[:, None, :], axis=2)
    gauss = np.exp(-0.5 * (d / sigma) ** 2).max(axis=0)
    return floor + (1.0 - floor) * gauss


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

    print(f"pixel round-trip OK ({recovered} ~= {obj_uv})")

    # The MolmoPoint lift: a single pixel is padded into a window, and the anchor is the
    # median of the cloud points projecting into it. Point at the object's pixel and check
    # we recover its true 3D centroid.
    win = 8
    cams = [{"name": "left", "intrinsic": K, "base2cam": b2c}]
    window = np.array([obj_uv[0] - win, obj_uv[1] - win, obj_uv[0] + win, obj_uv[1] + win])
    cands = candidate_anchors(pts, cams, {"left": window[None]}, (H, W), min_in_box=10)
    assert len(cands) == 1, f"expected one candidate from one window, got {len(cands)}"
    anchor, mask = cands[0]
    truth = obj_base.mean(axis=0)
    err = float(np.linalg.norm(anchor - truth))
    print(f"lift: n_in_window={int(mask.sum())} anchor_err={err*100:.1f} cm")
    assert err < 0.05, f"lifted anchor too far from the planted cluster: {err:.3f} m"

    # Single-anchor weights must stay bit-identical to the original one-centre form, or the
    # eef/oracle arms stop being comparable with the stage 0-2 runs.
    sigma, floor = 0.08, 0.05
    d = np.linalg.norm(pts - truth, axis=1)
    reference = floor + (1.0 - floor) * np.exp(-0.5 * (d / sigma) ** 2)
    w1 = eef_density_weights(pts, truth, sigma, floor)
    assert np.array_equal(w1, reference), "single-anchor weights drifted from the 1-centre form"
    assert np.array_equal(w1, eef_density_weights(pts, truth[None, :], sigma, floor)), \
        "(3,) and (1, 3) anchors must agree"

    # Two anchors: max, so the peak stays 1.0 and each anchor keeps its own bump.
    second = truth + np.array([0.4, 0.0, 0.0])
    w2 = eef_density_weights(pts, np.stack([truth, second]), sigma, floor)
    assert (w2 >= w1 - 1e-12).all(), "adding an anchor must not lower any weight"
    assert w2.max() <= 1.0 + 1e-12, "max-combination must keep the peak at 1.0"
    near_second = np.linalg.norm(pts - second, axis=1) < sigma
    if near_second.any():
        assert (w2[near_second] > w1[near_second] + 1e-6).any(), \
            "points near the second anchor should gain weight"
    print(f"weights: 1-anchor max={w1.max():.3f}  2-anchor max={w2.max():.3f}")
    print("geometry self-test OK")
