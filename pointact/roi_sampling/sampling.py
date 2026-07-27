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


def soft_guided_indices(
    num_points: int,
    n_total: int,
    weights: np.ndarray,
    roi_ratio: float,
    rng: np.random.Generator | None = None,
    tau: float = 0.5,
) -> np.ndarray:
    """Soft variant: sample without replacement using continuous ROI weights.

    Removes the hard radius boundary. Points are split into a "core" pool (weight >= tau,
    i.e. the well-inside-the-halo points) and the rest; the core receives ``roi_ratio`` of
    the budget and the remainder goes to the rest, but *within* each pool selection is
    probability-weighted by the Gaussian weights, so peripheral points fade out smoothly
    instead of being cut off.

    Falls back to a uniform draw when the weights carry no usable signal.
    """
    if rng is None:
        rng = np.random.default_rng()
    w = np.asarray(weights, dtype=np.float64)
    assert w.shape[0] == num_points, "weights length must match cloud size"
    target = int(min(n_total, num_points))

    core = np.flatnonzero(w >= tau)
    rest = np.flatnonzero(w < tau)
    if len(core) == 0 or len(rest) == 0:
        return rng.choice(num_points, target, replace=False)

    n_core = int(round(roi_ratio * target))
    n_rest = target - n_core
    if n_core > len(core):
        n_core = len(core)
        n_rest = target - n_core
    if n_rest > len(rest):
        n_rest = len(rest)
        n_core = target - n_rest
    n_core = min(n_core, len(core))

    def draw(idx, k, prob):
        if k <= 0:
            return np.empty(0, dtype=np.int64)
        p = prob[idx].astype(np.float64)
        s = p.sum()
        if not np.isfinite(s) or s <= 0:
            return rng.choice(idx, k, replace=False)
        return rng.choice(idx, k, replace=False, p=p / s)

    # Core: weight-proportional. Rest: inverse-ish so nearer-the-halo context is favoured
    # slightly, but every background point keeps a non-zero chance.
    sel_core = draw(core, n_core, w)
    sel_rest = draw(rest, n_rest, w + 1e-3)
    out = np.concatenate([sel_core, sel_rest])
    rng.shuffle(out)
    return out


def density_weighted_indices(
    num_points: int,
    n_total: int,
    weights: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Select ``min(n_total, num_points)`` indices by plain weight-proportional sampling.

    No pool split / quota, unlike :func:`roi_guided_indices` and :func:`soft_guided_indices`:
    every point is drawn without replacement with probability proportional to ``weights``.
    Intended for a strictly-positive density (e.g. :func:`pointact.roi_sampling.geometry.
    eef_density_weights`, which has a floor), so there is no "empty ROI" fallback case.

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

    # Soft variant: exact count, core share honoured, no hard boundary.
    d = np.linalg.norm(rng.standard_normal((m, 3)) * 0.3, axis=1)
    wts = np.exp(-0.5 * (d / 0.25) ** 2)
    idx5 = soft_guided_indices(m, 4096, wts, 0.7, rng)
    assert len(idx5) == 4096 and len(np.unique(idx5)) == 4096
    core_frac = (wts[idx5] >= 0.5).mean()
    print(f"soft: total={len(idx5)} core_share={core_frac:.2f} (target ~0.70 capped by availability)")

    # Density-weighted: near-eef points dominate but the floor keeps far points reachable.
    near = np.exp(-0.5 * (d / 0.08) ** 2)
    wts_floor = 0.05 + 0.95 * near
    idx6 = density_weighted_indices(m, 4096, wts_floor, rng)
    assert len(idx6) == 4096 and len(np.unique(idx6)) == 4096
    near_frac = (d[idx6] < 0.08).mean()
    print(f"density: total={len(idx6)} near_eef_frac={near_frac:.2f} (base rate {(d < 0.08).mean():.2f})")
    print("sampling self-test OK")
