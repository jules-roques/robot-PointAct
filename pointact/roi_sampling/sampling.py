"""Guarded ROI/background subsampling.

Replaces the uniform ``np.random.choice`` draw in the dataloader while preserving
the baseline's total point count exactly. The budget is split between ROI points
(inside the detected halo) and background points; if either pool is short its quota
spills over to the other so the returned count is always ``min(n_total, len(cloud))``.
When no ROI is available (missing/unreliable detection) the caller uses a plain
uniform draw, so behaviour never degrades below the baseline.
"""

from __future__ import annotations

import numpy as np


def roi_guided_indices(
    num_points: int,
    n_total: int,
    roi_mask: np.ndarray,
    roi_ratio: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Select ``min(n_total, num_points)`` indices, biased toward ROI points.

    Args:
        num_points: size of the cloud being sampled (``len(cloud)``).
        n_total: desired number of points (the baseline count rule result).
        roi_mask: (num_points,) bool, True for ROI points.
        roi_ratio: fraction of the budget allocated to ROI points (0..1).
        rng: numpy Generator (defaults to a fresh default_rng).

    Returns:
        1-D int array of selected indices (unordered), length ``min(n_total, num_points)``.
    """
    if rng is None:
        rng = np.random.default_rng()
    roi_mask = np.asarray(roi_mask, dtype=bool)
    assert roi_mask.shape[0] == num_points, "roi_mask length must match cloud size"

    target = int(min(n_total, num_points))
    idx_roi = np.flatnonzero(roi_mask)
    idx_bg = np.flatnonzero(~roi_mask)

    # No ROI or no background -> plain uniform draw over everything.
    if len(idx_roi) == 0 or len(idx_bg) == 0:
        return rng.choice(num_points, target, replace=False)

    n_roi = int(round(roi_ratio * target))
    n_bg = target - n_roi

    # Clamp to availability, spilling any shortfall to the other pool.
    if n_roi > len(idx_roi):
        n_roi = len(idx_roi)
        n_bg = target - n_roi
    if n_bg > len(idx_bg):
        n_bg = len(idx_bg)
        n_roi = target - n_bg
    # Both pools together always hold >= target points, so this is satisfiable.
    n_roi = min(n_roi, len(idx_roi))

    sel_roi = rng.choice(idx_roi, n_roi, replace=False)
    sel_bg = rng.choice(idx_bg, n_bg, replace=False)
    out = np.concatenate([sel_roi, sel_bg])
    rng.shuffle(out)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Typical case: plenty of both pools, exact count + ratio honoured.
    m = 6000
    mask = np.zeros(m, dtype=bool)
    mask[:3000] = True
    idx = roi_guided_indices(m, 4096, mask, 0.6, rng)
    assert len(idx) == 4096, len(idx)
    assert len(np.unique(idx)) == 4096, "no duplicates"
    n_roi = mask[idx].sum()
    assert abs(n_roi - round(0.6 * 4096)) <= 1, n_roi
    print(f"typical: total={len(idx)} roi={n_roi} (~{round(0.6*4096)})")

    # ROI-poor: fewer ROI than quota -> spill to background, still exact total.
    mask2 = np.zeros(m, dtype=bool)
    mask2[:100] = True
    idx2 = roi_guided_indices(m, 4096, mask2, 0.6, rng)
    assert len(idx2) == 4096 and len(np.unique(idx2)) == 4096
    assert mask2[idx2].sum() == 100, mask2[idx2].sum()
    print(f"roi-poor: total={len(idx2)} roi={mask2[idx2].sum()} (all 100 available)")

    # Small cloud: fewer points than budget -> return all.
    idx3 = roi_guided_indices(300, 4096, mask[:300], 0.6, rng)
    assert len(idx3) == 300 and len(np.unique(idx3)) == 300
    print(f"small-cloud: total={len(idx3)} (=cloud size)")

    # Empty ROI -> uniform fallback of exact size.
    idx4 = roi_guided_indices(m, 4096, np.zeros(m, dtype=bool), 0.6, rng)
    assert len(idx4) == 4096 and len(np.unique(idx4)) == 4096
    print("empty-roi: uniform fallback OK")
    print("sampling self-test OK")
