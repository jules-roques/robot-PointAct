"""Visualization gate: render ROI-segmented clouds as one interactive HTML.

Reads sample frames from the points LMDB + the ROI-flag LMDB and writes a single
self-contained HTML (embedded plotly.js) with a dropdown to switch frames. Per frame:
  - background points (their true RGB, dimmed),
  - ROI points (red),
  - the actual guarded 4096-point draw (green, hidden by default — toggle in legend).

Open the HTML locally to rotate/zoom. Run this before launching training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lmdb
import msgpack
import msgpack_numpy
import numpy as np
import plotly.graph_objects as go

from pointact.roi_sampling.sampling import roi_guided_indices

msgpack_numpy.patch()


def load_cloud(txn, ep, f):
    buf = txn.get(f"{ep}-{f}".encode("ascii"))
    return None if buf is None else msgpack.unpackb(bytes(buf)).astype(np.float32)


def load_flag(txn, ep, f, m):
    buf = txn.get(f"{ep}-{f}".encode("ascii"))
    if buf is None:
        return np.zeros(m, dtype=bool)
    return np.unpackbits(np.frombuffer(bytes(buf), dtype=np.uint8))[:m].astype(bool)


def rgb_str(colors: np.ndarray) -> list[str]:
    c = np.clip(colors, 0.0, 1.0)
    c = (c * 255).astype(int)
    return [f"rgb({r},{g},{b})" for r, g, b in c]


def scatter(pts, color, name, size, visible=True, opacity=1.0):
    return go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
        marker=dict(size=size, color=color, opacity=opacity),
        name=name, visible=visible,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--roi-dirname", default="points_3views_roi")
    ap.add_argument("--points-dirname", default="points_3views")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--samples", nargs="*", default=None,
                    help="Explicit 'ep-frame' ids; default auto-picks a spread.")
    ap.add_argument("--episodes", nargs="*", type=int, default=None,
                    help="Restrict auto-picking to these episodes (e.g. a built subset).")
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--max-points", type=int, default=9000, help="Cap points drawn per frame.")
    ap.add_argument("--roi-ratio", type=float, default=0.6)
    ap.add_argument("--max-npoints", type=int, default=4096)
    ap.add_argument("--fragment", action="store_true",
                    help="Write body-only HTML (for embedding, e.g. an Artifact) instead of a full page.")
    args = ap.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    p_env = lmdb.open(str(dataset_dir / args.points_dirname), readonly=True, lock=False, readahead=False)
    r_env = lmdb.open(str(dataset_dir / args.roi_dirname), readonly=True, lock=False, readahead=False)

    # Choose sample frames: prefer frames that actually have ROI detections.
    if args.samples:
        samples = [tuple(int(x) for x in s.split("-")) for s in args.samples]
    else:
        lengths = {}
        with open(dataset_dir / "meta" / "episodes.jsonl") as f:
            for line in f:
                row = json.loads(line)
                lengths[int(row["episode_index"])] = int(row["length"])
        eps = sorted(lengths)
        if args.episodes:
            eps = [e for e in eps if e in set(args.episodes)]
        step = max(1, len(eps) // args.num_samples)
        cand_eps = eps[::step][: args.num_samples]
        samples = []
        with p_env.begin(buffers=True) as ptxn, r_env.begin(buffers=True) as rtxn:
            for ep in cand_eps:
                # pick a mid-episode frame that has a detection if possible
                n = lengths[ep]
                chosen = n // 2
                for f in range(n // 2, n):
                    cloud = load_cloud(ptxn, ep, f)
                    if cloud is None:
                        continue
                    if load_flag(rtxn, ep, f, len(cloud)).any():
                        chosen = f
                        break
                samples.append((ep, chosen))

    rng = np.random.default_rng(0)
    traces = []
    frame_trace_ranges = []
    with p_env.begin(buffers=True) as ptxn, r_env.begin(buffers=True) as rtxn:
        for (ep, f) in samples:
            cloud = load_cloud(ptxn, ep, f)
            if cloud is None:
                continue
            m = len(cloud)
            flag = load_flag(rtxn, ep, f, m)
            pts, rgb = cloud[:, :3], cloud[:, 3:6]
            # RGB may be stored in [-1,1] (dataloader rescales); map to [0,1] for display.
            disp_rgb = (rgb + 1) / 2 if rgb.min() < 0 else rgb

            # Downsample for display, keeping all ROI points.
            roi_idx = np.flatnonzero(flag)
            bg_idx = np.flatnonzero(~flag)
            budget_bg = max(0, args.max_points - len(roi_idx))
            if len(bg_idx) > budget_bg:
                bg_idx = rng.choice(bg_idx, budget_bg, replace=False)

            start = len(traces)
            traces.append(scatter(pts[bg_idx], rgb_str(disp_rgb[bg_idx]),
                                  f"{ep}-{f} background", 1.5, opacity=0.35))
            traces.append(scatter(pts[roi_idx], "red", f"{ep}-{f} ROI ({len(roi_idx)})", 2.5))

            # Post-sample view (green), hidden by default.
            if flag.any() and not flag.all():
                sel = roi_guided_indices(m, args.max_npoints, flag, args.roi_ratio, rng)
            else:
                sel = rng.choice(m, min(args.max_npoints, m), replace=False)
            traces.append(scatter(pts[sel], "green",
                                  f"{ep}-{f} post-sample 4096 ({flag[sel].sum()} roi)",
                                  2.0, visible="legendonly"))
            frame_trace_ranges.append((start, len(traces)))

    p_env.close(); r_env.close()

    # Dropdown: show one frame's traces at a time.
    buttons = []
    for i, ((ep, f), (s, e)) in enumerate(zip(samples, frame_trace_ranges)):
        vis = [False] * len(traces)
        for t in range(s, e):
            # keep the post-sample trace as legendonly, others visible
            vis[t] = "legendonly" if traces[t].visible == "legendonly" else True
        buttons.append(dict(label=f"{ep}-{f}", method="update",
                            args=[{"visible": vis}, {"title": f"ROI cloud — episode {ep} frame {f}"}]))

    # Default: first frame visible only.
    for i, t in enumerate(traces):
        s0, e0 = frame_trace_ranges[0]
        if not (s0 <= i < e0):
            traces[i].visible = False

    fig = go.Figure(data=traces)
    fig.update_layout(
        updatemenus=[dict(buttons=buttons, x=0.0, y=1.12, xanchor="left")],
        scene=dict(aspectmode="data"),
        title=f"ROI cloud — episode {samples[0][0]} frame {samples[0][1]}",
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(itemsizing="constant"),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(args.out), include_plotlyjs=True, full_html=not args.fragment)
    print(f"wrote interactive viz ({len(samples)} frames) -> {args.out}")


if __name__ == "__main__":
    main()
