"""Fuse per-view MolmoMotion gate runs into a combined arm, without touching the GPU.

MolmoMotion is single-camera, so scoring two agentviews means two independent forwards on
the same instant. Whether averaging them helps is a question about the *geometry of their
mistakes*, not about the model, and it is answerable from stored predictions alone -- so it
does not belong inside the GPU job. Running one job per view and fusing here also keeps each
job inside the 2-hour dev-QoS wall clock (docs/clusters/jean-zay.md), which a single
two-view job does not comfortably fit.

Requires runs recorded after pred_base/truth_base were added to gate_records.npz; older runs
stored error magnitudes only and cannot be fused retroactively.

    python -m data_prep.roi_sampling.fuse_gate_views \
        --records $SCRATCH/viz/.../P1_left/gate_records.npz \
                  $SCRATCH/viz/.../P1_right/gate_records.npz \
        --out $SCRATCH/viz/.../fused_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from data_prep.roi_sampling.gate_molmo_motion import fuse_views, summarise


def load_records(path: Path) -> list[dict]:
    d = np.load(path, allow_pickle=True)
    if "pred_base" not in d.files:
        raise SystemExit(
            f"{path} has no pred_base -- it predates position recording and cannot be "
            f"fused. Re-run the gate for this view.")
    n = len(d["err_m"])
    recs = []
    for i in range(n):
        pred = np.asarray(d["pred_base"][i], dtype=np.float64)
        if not np.all(np.isfinite(pred)):
            continue
        recs.append({
            "task": str(d["task"][i]), "view": str(d["view"][i]),
            "episode": int(d["episode"][i]), "t0": int(d["t0"][i]),
            "stride": int(d["stride"][i]), "horizon_s": float(d["horizon_s"][i]),
            "err_m": float(d["err_m"][i]), "static_err_m": float(d["static_err_m"][i]),
            "pred_base": pred.tolist(),
            "truth_base": np.asarray(d["truth_base"][i], dtype=np.float64).tolist(),
        })
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", type=Path, required=True,
                    help="gate_records.npz from each view's run.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    records: list[dict] = []
    for p in args.records:
        r = load_records(p)
        views = sorted({x["view"] for x in r})
        print(f"{p}: {len(r)} records, views={views}")
        records.extend(r)

    views = sorted({r["view"] for r in records})
    if len(views) < 2:
        raise SystemExit(f"need at least two distinct views to fuse, got {views}")

    fused = fuse_views(records)
    if not fused:
        raise SystemExit(
            "no sample was seen from more than one view -- the runs must cover the same "
            "tasks, episodes, t0 grid, stride and horizons to be fusable")
    print(f"\nfused {len(fused)} samples across {views}")

    rows = summarise(records + fused)
    print(f"\n{'task':<26}{'view':<7}{'hor':<6}{'n':<6}"
          f"{'err_cm':<9}{'motion_cm':<11}{'win_vs_static':<14}")
    for r in rows:
        print(f"{r['task']:<26}{r['view']:<7}{r['horizon_s']:<6}{r['n']:<6}"
              f"{r['median_err_cm']:<9.1f}{r['median_motion_cm']:<11.1f}"
              f"{r['win_rate_vs_static']:<14.2f}")

    # The comparison the fused arm exists to answer: does averaging beat the *better* single
    # view? Beating their average is guaranteed by the triangle inequality and means nothing.
    print("\nfused vs the best single view (negative = fusion wins):")
    for task in sorted({r["task"] for r in rows}):
        for sec in sorted({r["horizon_s"] for r in rows if r["task"] == task}):
            sel = {r["view"]: r for r in rows
                   if r["task"] == task and r["horizon_s"] == sec}
            if "fused" not in sel:
                continue
            singles = {v: r["median_err_cm"] for v, r in sel.items() if v != "fused"}
            best_view = min(singles, key=singles.get)
            delta = sel["fused"]["median_err_cm"] - singles[best_view]
            print(f"  {task:<26}{sec:<6}{sel['fused']['median_err_cm']:>6.1f} vs "
                  f"{singles[best_view]:>5.1f} ({best_view:<5}) "
                  f"delta={delta:+.1f} cm  win={sel['fused']['win_rate_vs_static']:.2f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"rows": rows, "n_fused": len(fused)}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
