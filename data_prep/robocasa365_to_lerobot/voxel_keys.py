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

# The RoboCASA workspace reaches 1.0 m from the base-frame origin on its longest axis
# (+/-0.8 m in x/y, 0..1.0 m in z), so a voxel index is bounded by ceil(1.0 / voxel_size).
WORKSPACE_EXTENT_M = 1.0
# Keys are int32, so radix ** 3 - 1 (the largest key) must fit: 1024 is the last power of two
# that does. That puts the floor on voxel size at 2 mm; finer would need int64 keys and a
# matching dtype change in the cached *_voxel_keys.npy arrays.
MAX_VOXEL_KEY_RADIX = 1024

# Defaults preserved for the 0.01 m grid every existing cache was built on: voxel_key_params
# returns exactly these for voxel_size=0.01, so those caches stay byte-identical.
VOXEL_KEY_OFFSET = 128
VOXEL_KEY_RADIX = 256


def voxel_key_params(voxel_size: float) -> tuple[int, int]:
    """(offset, radix) for a given grid, so both sides of a join derive the same packing.

    The packing has to be a pure function of the voxel size and nothing else: replay writes
    the key arrays and export_point_labels recomputes them from the stored cloud, in different
    environments and often months apart. Deriving from the data (e.g. the observed index range
    of one frame) would let two frames disagree and silently corrupt the join.
    """
    voxel_size = float(voxel_size)
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")
    # Indices span [-limit, +limit], so the radix has to cover 2 * limit + 1 values.
    limit = int(np.ceil(WORKSPACE_EXTENT_M / voxel_size)) + 1
    radix = 1 << int(np.ceil(np.log2(2 * limit + 1)))
    if radix > MAX_VOXEL_KEY_RADIX:
        raise ValueError(
            f"voxel_size={voxel_size} needs radix {radix}, above the int32 limit of "
            f"{MAX_VOXEL_KEY_RADIX} (~2 mm). Finer grids need int64 keys."
        )
    return radix // 2, radix


def pack_voxel_keys(
    voxel_indices: np.ndarray,
    offset: int = VOXEL_KEY_OFFSET,
    radix: int = VOXEL_KEY_RADIX,
) -> np.ndarray:
    """Pack (N, 3) integer voxel coordinates into (N,) int32 keys.

    The packing is order-preserving: sorting by key equals sorting lexicographically by
    (ix, iy, iz), which is the order `np.unique(..., axis=0)` produces in the merge. That
    holds for any radix, so widening it for a finer grid does not change the emitted order.

    Callers working at a voxel size other than 0.01 must pass the pair from
    :func:`voxel_key_params`; the defaults only fit the 0.01 m grid.
    """
    shifted = np.asarray(voxel_indices, dtype=np.int64) + offset
    if shifted.size and (shifted.min() < 0 or shifted.max() >= radix):
        raise ValueError(
            "Voxel index outside the packable range "
            f"[{-offset}, {radix - offset - 1}]: "
            f"{np.asarray(voxel_indices).min()}..{np.asarray(voxel_indices).max()}"
        )
    return (
        shifted[:, 0] * radix * radix
        + shifted[:, 1] * radix
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
    offset, radix = voxel_key_params(voxel_size)
    return pack_voxel_keys(
        np.floor(points_xyz / float(voxel_size)).astype(np.int64), offset, radix
    )
