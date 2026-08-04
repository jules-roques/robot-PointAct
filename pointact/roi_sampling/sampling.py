"""Density-weighted subsampling.

Replaces the uniform ``np.random.choice`` draw in the dataloader while preserving the
baseline's total point count exactly: points are drawn without replacement with
probability proportional to a per-point weight (see
``pointact.roi_sampling.geometry.eef_density_weights``), so a region of interest gets
most of the budget while the weight floor keeps every other point reachable.
"""

from __future__ import annotations

import numpy as np


def density_weighted_indices(
    num_points: int,
    n_total: int,
    weights: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Select ``min(n_total, num_points)`` indices by plain weight-proportional sampling.

    Every point is drawn without replacement with probability proportional to ``weights``.
    Intended for a strictly-positive density (:func:`pointact.roi_sampling.geometry.
    eef_density_weights`, which has a floor), so there is no "empty ROI" fallback case: the
    floor guarantees background points stay reachable rather than being quota'd in.

    Args:
        num_points: size of the cloud being sampled (``len(cloud)``).
        n_total: desired number of points (the baseline count rule result).
        weights: (num_points,) float, strictly positive.
        rng: numpy Generator (defaults to a fresh default_rng).

    Returns:
        1-D int array of selected indices (unordered), length ``min(n_total, num_points)``.
    """
    if rng is None:
        rng = np.random.default_rng()
    w = np.asarray(weights, dtype=np.float64)
    assert w.shape[0] == num_points, "weights length must match cloud size"
    target = int(min(n_total, num_points))

    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return rng.choice(num_points, target, replace=False)
    return rng.choice(num_points, target, replace=False, p=w / s)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    m = 6000
    d = np.linalg.norm(rng.standard_normal((m, 3)) * 0.3, axis=1)

    # Density-weighted: near-anchor points dominate, the floor keeps far points reachable.
    # sigma=0.25 against a cloud of scale 0.3 so a meaningful share of points sit under the
    # bump -- at sigma=0.08 barely 1% do, and the test would pass on any implementation.
    wts_floor = 0.05 + 0.95 * np.exp(-0.5 * (d / 0.25) ** 2)
    idx = density_weighted_indices(m, 4096, wts_floor, rng)
    assert len(idx) == 4096 and len(np.unique(idx)) == 4096, "exact count, no duplicates"
    near, base = (d[idx] < 0.25).mean(), (d < 0.25).mean()
    assert near > base + 0.02, f"the bump must oversample its neighbourhood: {near:.3f} vs {base:.3f}"
    # The floor must still reach the tail: nothing is starved to zero probability.
    assert (d[idx] > 3 * 0.25).any(), "the floor should keep far points reachable"
    print(f"density: total={len(idx)} near_frac={near:.2f} (base rate {base:.2f})")

    # Small cloud: fewer points than the budget -> return them all.
    idx2 = density_weighted_indices(300, 4096, wts_floor[:300], rng)
    assert len(idx2) == 300 and len(np.unique(idx2)) == 300
    print(f"small-cloud: total={len(idx2)} (=cloud size)")

    # Degenerate weights -> uniform fallback of the exact size, never a crash.
    idx3 = density_weighted_indices(m, 4096, np.zeros(m), rng)
    assert len(idx3) == 4096 and len(np.unique(idx3)) == 4096
    print("zero-weights: uniform fallback OK")
    print("sampling self-test OK")
