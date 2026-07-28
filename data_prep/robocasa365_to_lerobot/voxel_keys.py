"""Packed voxel identity for merged point clouds.

Kept dependency-free (numpy only) on purpose: the replay side runs in the simulator env and the
export side in the training env, and neither can import the other's stack.

Replaying a RoboCASA episode is not bit-deterministic across machines -- a point sitting on a
voxel boundary can land either side -- so a frame's cloud can gain or lose a point or two
between runs. Labels therefore cannot be attached to a cloud by array position. A merged
point's voxel coordinates are its canonical identity, and `voxel_downsample` emits points
sorted by them, so joining two independently-produced clouds on this key is exact and ordered.
"""

from __future__ import annotations

import numpy as np

# The RoboCASA workspace is +/-0.8 m in x/y and 0..1.0 m in z; at 0.01 m voxels that is indices
# in [-80, 100]. The +128 offset and 256 radix keep every key positive and well inside int32.
VOXEL_KEY_OFFSET = 128
VOXEL_KEY_RADIX = 256


def pack_voxel_keys(voxel_indices: np.ndarray) -> np.ndarray:
    """Pack (N, 3) integer voxel coordinates into (N,) int32 keys.

    The packing is order-preserving: sorting by key equals sorting lexicographically by
    (ix, iy, iz), which is the order `np.unique(..., axis=0)` produces in the merge.
    """
    shifted = np.asarray(voxel_indices, dtype=np.int64) + VOXEL_KEY_OFFSET
    if shifted.size and (shifted.min() < 0 or shifted.max() >= VOXEL_KEY_RADIX):
        raise ValueError(
            "Voxel index outside the packable range: "
            f"{np.asarray(voxel_indices).min()}..{np.asarray(voxel_indices).max()}"
        )
    return (
        shifted[:, 0] * VOXEL_KEY_RADIX * VOXEL_KEY_RADIX
        + shifted[:, 1] * VOXEL_KEY_RADIX
        + shifted[:, 2]
    ).astype(np.int32)


def voxel_keys_for_points(points_xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxel key of each point of an already-merged cloud.

    A merged point is the mean of its voxel's members and so lies inside that voxel; flooring
    it recovers the index the merge used.
    """
    points_xyz = np.asarray(points_xyz)
    if len(points_xyz) == 0:
        return np.empty((0,), dtype=np.int32)
    return pack_voxel_keys(np.floor(points_xyz / float(voxel_size)).astype(np.int64))
