"""Metric voxel downsampling for point clouds that are already voxelized once.

Mirrors data_prep/robocasa365_to_lerobot/replay.py:voxel_downsample -- an absolute grid
(``floor(xyz / voxel_size)``, origin at 0) and a per-voxel mean over every column -- so that
re-voxelizing a cache onto its own grid is an exact no-op and onto a coarser one merges the
same way the replay would have. Kept here rather than imported because replay.py lives in the
simulator env and pulls in its stack.
"""

from __future__ import annotations

import numpy as np

# Voxel indices are offset into [0, 2**21) per axis and packed into one int64 key, which
# covers +/-10 km at 1 mm. Nothing in the workspace comes near that; the assert makes sure.
_KEY_BITS = 21
_KEY_OFFSET = 1 << (_KEY_BITS - 1)


def voxel_downsample(point_cloud: np.ndarray, voxel_size: float) -> np.ndarray:
    """Merge every point sharing a ``voxel_size`` cell into their mean (all columns)."""
    if voxel_size <= 0 or len(point_cloud) == 0:
        return point_cloud
    idx = np.floor(point_cloud[:, :3] / float(voxel_size)).astype(np.int64) + _KEY_OFFSET
    assert idx.min() >= 0 and idx.max() < (1 << _KEY_BITS), "voxel index out of key range"
    keys = (idx[:, 0] << (2 * _KEY_BITS)) | (idx[:, 1] << _KEY_BITS) | idx[:, 2]
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    inverse = np.ravel(inverse)
    merged = np.empty((len(counts), point_cloud.shape[1]), dtype=point_cloud.dtype)
    for column in range(point_cloud.shape[1]):
        merged[:, column] = np.bincount(
            inverse, weights=point_cloud[:, column].astype(np.float64), minlength=len(counts)
        ) / counts
    return merged
