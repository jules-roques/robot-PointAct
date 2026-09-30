"""Coordinate scale + point grid: the two knobs a resolution arm turns.

CPU-only and data-free, so it runs anywhere the training env does.
"""

import numpy as np
import torch

from pointact.data.robot.voxel import voxel_downsample
from pointact.model.vla_pointact.action_head_3d.ptv3_backbone import scale_point_coords
from pointact.train.run_config import group_from_meta, run_name_from_meta


def _grid_cloud(spacing, n=2000, seed=0):
    """Points sitting at the centres of a `spacing` grid, as a replay cache stores them."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(-40, 40, size=(n, 3))
    idx = np.unique(idx, axis=0)
    xyz = (idx + 0.5) * spacing
    rgb = rng.random((len(xyz), 3))
    return np.concatenate([xyz, rgb], axis=1).astype(np.float32)


def test_revoxelize_onto_own_grid_is_noop():
    cloud = _grid_cloud(0.0025)
    out = voxel_downsample(cloud, 0.0025)
    assert len(out) == len(cloud)
    order_in = np.lexsort(cloud[:, :3].T)
    order_out = np.lexsort(out[:, :3].T)
    np.testing.assert_allclose(out[order_out], cloud[order_in], atol=1e-6)


def test_coarser_grid_merges_to_one_point_per_cell():
    cloud = _grid_cloud(0.0025, n=20000)
    for grid in (0.01 / np.sqrt(2), 0.005, 0.01):
        out = voxel_downsample(cloud, grid)
        cells = np.floor(out[:, :3] / grid).astype(np.int64)
        assert len(np.unique(cells, axis=0)) == len(out), grid
        assert len(out) < len(cloud)


def test_merge_is_a_mean():
    cloud = np.array([[0.001, 0.001, 0.001, 0.0, 0.0, 0.0],
                      [0.003, 0.003, 0.003, 1.0, 1.0, 1.0]], dtype=np.float32)
    out = voxel_downsample(cloud, 0.005)
    np.testing.assert_allclose(out, [[0.002, 0.002, 0.002, 0.5, 0.5, 0.5]], atol=1e-6)


def test_scale_touches_xyz_only():
    pc = torch.randn(10, 6)
    out = scale_point_coords(pc, 4.0)
    torch.testing.assert_close(out[:, :3], pc[:, :3] * 4.0)
    torch.testing.assert_close(out[:, 3:], pc[:, 3:])
    assert scale_point_coords(pc, 1.0) is pc


def test_names_keep_resolution_arms_apart():
    base = {"task": "OpenDrawer", "sampling": "oracle", "npoints": 8192, "seed": 0}
    assert run_name_from_meta(base) == "od-oracle-n8192-s0"
    assert run_name_from_meta({**base, "scale": 1.0, "grid": 0.01}) == "od-oracle-n8192-s0"
    fine = {**base, "scale": 2.0, "grid": 0.005}
    control = {**base, "scale": 2.0, "grid": 0.01}
    assert run_name_from_meta(fine) == "od-oracle-n8192-x2-g5mm-s0"
    assert run_name_from_meta(control) == "od-oracle-n8192-x2-s0"
    assert group_from_meta(fine) != group_from_meta(control) != group_from_meta(base)
    root2 = {**base, "scale": 2 ** 0.5, "grid": 0.01 / 2 ** 0.5}
    assert run_name_from_meta(root2) == "od-oracle-n8192-x1.41421-g7.07mm-s0"
