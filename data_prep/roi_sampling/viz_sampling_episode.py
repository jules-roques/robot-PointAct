"""Animate one episode's sampled point cloud, one HTML per sampling strategy.

Where ``viz_roi.py`` shows isolated frames side by side, this shows the *evolution*: a
play/pause + slider animation over the frames of a single episode, so you can watch where
the fixed 4096-point budget goes as the arm approaches, grasps and pulls the drawer.

Strategies (each writes its own self-contained HTML, matching the dataloader exactly):

  uniform  ``np.random.choice`` over the whole cloud — the baseline.
  eef      ``density_weighted_indices`` on ``eef_density_weights``: a Gaussian bump around
           the frame's own end-effector (``observation.state[:3]``) with a density floor.
  roi      ``roi_guided_indices``/``soft_guided_indices`` on a halo centred on the *handle*,
           tracked proprioceptively (the oracle ROI of ``build_roi_cache_proprio.py``): the
           eef while the gripper is closed, held at the grasp/release pose outside that run.
           Computed on the fly here, so no ROI LMDB cache is required.

The anchor each strategy is centred on is drawn as a marker, and for ``roi`` the halo
sphere is outlined, so "where the budget went" is visible rather than inferred.

Unlike training, the point count is deterministic (``min(len(cloud), max_npoints)``, no
0.8-1.0 jitter) so frames and strategies are comparable frame-for-frame.

Example:
    python -m data_prep.roi_sampling.viz_sampling_episode \
        --dataset-dir $DATA/lerobot_point_lmdb/OpenDrawer --episode 0 \
        --method all --out-dir /scratch/$USER/viz
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
import pyarrow.parquet as pq

from data_prep.roi_sampling.build_roi_cache_proprio import handle_trajectory
from pointact.roi_sampling.geometry import eef_density_weights, halo_weights
from pointact.roi_sampling.sampling import (
    density_weighted_indices,
    roi_guided_indices,
    soft_guided_indices,
)

msgpack_numpy.patch()

METHODS = ("uniform", "eef", "roi")

METHOD_LABEL = {
    "uniform": "Uniform (baseline)",
    "eef": "EEF-density (Gaussian + floor around the end-effector)",
    "roi": "Oracle ROI (halo on the proprioceptively tracked handle)",
}

ANCHOR_LABEL = {"uniform": "end-effector (reference only)", "eef": "end-effector (anchor)",
                "roi": "handle (ROI anchor)"}
ANCHOR_COLOR = {"uniform": "#9aa0a6", "eef": "#ff2d95", "roi": "#ff2d95"}


def load_cloud(txn, ep: int, f: int) -> np.ndarray | None:
    buf = txn.get(f"{ep}-{f}".encode("ascii"))
    return None if buf is None else msgpack.unpackb(bytes(buf)).astype(np.float32)


def read_episode_arrays(dataset_dir: Path, ep: int, chunks: int) -> tuple[np.ndarray, np.ndarray]:
    """(states, actions) for one episode, from the LeRobot parquet shard."""
    p = dataset_dir / "data" / f"chunk-{ep // chunks:03d}" / f"episode_{ep:06d}.parquet"
    tb = pq.read_table(p, columns=["observation.state", "action"])
    states = np.array([np.asarray(x, np.float32) for x in tb.column("observation.state").to_pylist()])
    actions = np.array([np.asarray(x, np.float32) for x in tb.column("action").to_pylist()])
    return states, actions


def hex_colors(rgb: np.ndarray) -> list[str]:
    """Stored RGB (either [0,1] or [-1,1]) -> '#rrggbb'. Hex keeps the HTML ~2x smaller."""
    c = (rgb + 1) / 2 if rgb.min() < 0 else rgb
    c = (np.clip(c, 0.0, 1.0) * 255).astype(np.uint8)
    return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in c]


def select_indices(
    method: str,
    cloud: np.ndarray,
    n_target: int,
    anchor: np.ndarray | None,
    args,
    rng: np.random.Generator,
) -> np.ndarray:
    """The dataloader's own selection, for one frame (see data_3d.augment_point_cloud)."""
    m = len(cloud)
    if method == "eef" and anchor is not None:
        w = eef_density_weights(cloud[:, :3], anchor, args.eef_sigma, args.eef_floor)
        return density_weighted_indices(m, n_target, w, rng)
    if method == "roi" and anchor is not None:
        w = halo_weights(cloud[:, :3], anchor, args.roi_radius * args.roi_radius_scale,
                         mode=args.roi_mode, softness=args.roi_softness)
        if args.roi_mode == "soft":
            if np.any(w >= 0.5) and not np.all(w >= 0.5):
                return soft_guided_indices(m, n_target, w, args.roi_ratio, rng)
        else:
            mask = w > 0
            if mask.any() and not mask.all():
                return roi_guided_indices(m, n_target, mask, args.roi_ratio, rng)
    return rng.choice(m, n_target, replace=False)


def sphere_wireframe(center: np.ndarray, radius: float, n_ring: int = 48) -> tuple[np.ndarray, ...]:
    """Three orthogonal great circles, as one line trace with NaN breaks."""
    t = np.linspace(0, 2 * np.pi, n_ring)
    nan = np.array([np.nan])
    cos, sin = np.cos(t) * radius, np.sin(t) * radius
    zero = np.zeros_like(cos)
    xs = np.concatenate([cos, nan, cos, nan, zero])
    ys = np.concatenate([sin, nan, zero, nan, cos])
    zs = np.concatenate([zero, nan, sin, nan, sin])
    return xs + center[0], ys + center[1], zs + center[2]


def build_figure(method: str, frames_data: list[dict], ep: int, task: str, args) -> go.Figure:
    """Assemble the animated scatter (slider + play/pause) for one strategy."""
    first = frames_data[0]
    all_xyz = np.concatenate([fd["pts"] for fd in frames_data], axis=0)
    lo, hi = all_xyz.min(axis=0), all_xyz.max(axis=0)
    pad = 0.05 * float(np.max(hi - lo))
    ranges = [[float(lo[i] - pad), float(hi[i] + pad)] for i in range(3)]

    def traces_for(fd: dict) -> list[go.Scatter3d]:
        pts = fd["pts"]
        out = [go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
            marker=dict(size=args.point_size, color=fd["colors"]),
            name=f"sampled points ({fd['n_sel']})", hoverinfo="skip",
        )]
        a = fd["anchor"]
        out.append(go.Scatter3d(
            x=np.array([np.nan]) if a is None else np.array([a[0]]),
            y=np.array([np.nan]) if a is None else np.array([a[1]]),
            z=np.array([np.nan]) if a is None else np.array([a[2]]),
            mode="markers",
            marker=dict(size=6, color=ANCHOR_COLOR[method], symbol="diamond",
                        line=dict(width=1, color="#ffffff")),
            name=ANCHOR_LABEL[method], hoverinfo="skip",
        ))
        if method == "roi":
            if a is None:
                sx = sy = sz = np.array([np.nan])
            else:
                sx, sy, sz = sphere_wireframe(a, args.roi_radius * args.roi_radius_scale)
            out.append(go.Scatter3d(
                x=sx, y=sy, z=sz, mode="lines",
                line=dict(width=2, color="rgba(255,45,149,0.55)"),
                name="halo radius", hoverinfo="skip",
            ))
        return out

    frames = [
        go.Frame(data=traces_for(fd), name=str(fd["frame"]),
                 layout=go.Layout(title=dict(text=frame_title(method, ep, task, fd, args))))
        for fd in frames_data
    ]

    fig = go.Figure(data=traces_for(first), frames=frames)
    steps = [dict(method="animate", label=str(fd["frame"]),
                  args=[[str(fd["frame"])],
                        dict(mode="immediate",
                             frame=dict(duration=args.frame_ms, redraw=True),
                             transition=dict(duration=0))])
             for fd in frames_data]
    fig.update_layout(
        title=dict(text=frame_title(method, ep, task, first, args)),
        scene=dict(
            aspectmode="data",
            xaxis=dict(range=ranges[0], title="x (m, base frame)"),
            yaxis=dict(range=ranges[1], title="y"),
            zaxis=dict(range=ranges[2], title="z"),
        ),
        margin=dict(l=0, r=0, t=76, b=0),
        legend=dict(itemsizing="constant", x=0.0, y=0.98),
        template="plotly_dark" if args.dark else "plotly_white",
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.02, y=0.02, xanchor="left", yanchor="bottom",
            direction="right", pad=dict(t=0, r=6),
            buttons=[
                dict(label="▶ play", method="animate",
                     args=[None, dict(mode="immediate", fromcurrent=True,
                                      frame=dict(duration=args.frame_ms, redraw=True),
                                      transition=dict(duration=0))]),
                dict(label="⏸ pause", method="animate",
                     args=[[None], dict(mode="immediate",
                                        frame=dict(duration=0, redraw=False),
                                        transition=dict(duration=0))]),
            ],
        )],
        sliders=[dict(active=0, x=0.12, len=0.86, y=0.02, xanchor="left", yanchor="bottom",
                      pad=dict(b=8, t=0), currentvalue=dict(prefix="frame ", visible=True),
                      steps=steps)],
    )
    return fig


def frame_title(method: str, ep: int, task: str, fd: dict, args) -> str:
    near = f"{fd['near_frac']:.0%}" if fd["near_frac"] is not None else "n/a"
    return (f"<b>{METHOD_LABEL[method]}</b><br>"
            f"<sup>ep {ep} · frame {fd['frame']}/{fd['n_frames'] - 1} · “{task}” · "
            f"{fd['n_sel']} of {fd['n_cloud']} pts · "
            f"{near} within {args.near_radius:g} m of the handle · "
            f"phase: {fd['phase']}</sup>")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--points-dirname", default="points_3views")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--method", default="all", choices=(*METHODS, "all"))
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    ap.add_argument("--out-prefix", default="sampling")
    ap.add_argument("--stride", type=int, default=0,
                    help="Frame stride; 0 = auto, chosen to yield ~--num-frames frames.")
    ap.add_argument("--num-frames", type=int, default=40, help="Target frame count when --stride 0.")
    ap.add_argument("--max-npoints", type=int, default=4096, help="The sampling budget (as in training).")
    ap.add_argument("--display-points", type=int, default=2500,
                    help="Uniform thinning of the selection for rendering only (keeps the HTML small; "
                         "relative density is preserved). 0 = draw all selected points.")
    ap.add_argument("--point-size", type=float, default=2.0)
    ap.add_argument("--frame-ms", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dark", action="store_true", help="Dark plotly template.")
    ap.add_argument("--near-radius", type=float, default=0.15,
                    help="Radius used for the reported 'fraction of budget near the handle' stat.")
    # eef-density knobs (defaults match the data config)
    ap.add_argument("--eef-sigma", type=float, default=0.08)
    ap.add_argument("--eef-floor", type=float, default=0.05)
    # ROI knobs (defaults match build_roi_cache_proprio + the data config)
    ap.add_argument("--roi-radius", type=float, default=0.15)
    ap.add_argument("--roi-radius-scale", type=float, default=1.0)
    ap.add_argument("--roi-ratio", type=float, default=0.7)
    ap.add_argument("--roi-mode", default="hard", choices=("hard", "soft"))
    ap.add_argument("--roi-softness", type=float, default=1.0)
    ap.add_argument("--plotlyjs", default="inline", choices=("inline", "cdn"),
                    help="'inline' is self-contained (~4.5 MB bigger); 'cdn' needs internet to view.")
    ap.add_argument("--fragment", action="store_true",
                    help="Body-only HTML (for embedding, e.g. an Artifact).")
    args = ap.parse_args()

    d = args.dataset_dir.expanduser().resolve()
    info = json.loads((d / "meta" / "info.json").read_text())
    chunks = int(info.get("chunks_size", 1000))

    lengths, tasks = {}, {}
    with open(d / "meta" / "episodes.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            lengths[int(r["episode_index"])] = int(r["length"])
            tk = r.get("tasks") or [""]
            tasks[int(r["episode_index"])] = tk[0] if tk else ""

    ep = args.episode
    if ep not in lengths:
        raise SystemExit(f"episode {ep} not in {d/'meta'/'episodes.jsonl'}")
    n_frames = lengths[ep]
    task = tasks.get(ep, "")

    states, actions = read_episode_arrays(d, ep, chunks)
    handle = handle_trajectory(states, actions)
    if handle is None:
        print(f"warning: no closed-gripper run in ep {ep}; the ROI anchor is unavailable "
              f"(that strategy falls back to uniform, exactly as the dataloader would)")
    closed = actions[:, 7] > 0.5 if actions.ndim == 2 and actions.shape[1] > 7 else None

    stride = args.stride if args.stride > 0 else max(1, n_frames // max(1, args.num_frames))
    frame_ids = list(range(0, n_frames, stride))
    methods = list(METHODS) if args.method == "all" else [args.method]

    env = lmdb.open(str(d / args.points_dirname), readonly=True, lock=False, readahead=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    with env.begin(buffers=True) as txn:
        # Load once, sample per method: identical clouds across the three files, so any
        # visible difference is the sampling strategy and nothing else.
        clouds = {}
        for f in frame_ids:
            c = load_cloud(txn, ep, f)
            if c is not None:
                clouds[f] = c
        if not clouds:
            raise SystemExit(f"no clouds found for ep {ep} in {d/args.points_dirname}")

        for method in methods:
            rng = np.random.default_rng(args.seed)
            disp_rng = np.random.default_rng(args.seed + 1)
            frames_data = []
            for f in sorted(clouds):
                cloud = clouds[f]
                anchor = None
                if method == "eef" and f < len(states):
                    anchor = states[f, :3].astype(np.float64)
                elif method == "roi" and handle is not None and f < len(handle):
                    anchor = handle[f]
                elif method == "uniform":  # reference marker only, never used for sampling
                    anchor = states[f, :3].astype(np.float64) if f < len(states) else None

                n_target = int(min(len(cloud), args.max_npoints))
                sel = select_indices(method, cloud, n_target, anchor if method != "uniform" else None,
                                     args, rng)
                sel_pts = cloud[sel]

                # Stat measured on the FULL selection, before display thinning.
                ref = handle[f] if (handle is not None and f < len(handle)) else None
                near_frac = None
                if ref is not None:
                    dist = np.linalg.norm(sel_pts[:, :3].astype(np.float64) - ref, axis=1)
                    near_frac = float((dist <= args.near_radius).mean())

                if args.display_points and len(sel_pts) > args.display_points:
                    keep = disp_rng.choice(len(sel_pts), args.display_points, replace=False)
                    sel_pts = sel_pts[keep]

                if closed is not None and f < len(closed):
                    phase = "gripper closed" if closed[f] else "gripper open"
                else:
                    phase = "—"

                frames_data.append(dict(
                    frame=f, n_frames=n_frames, pts=sel_pts[:, :3].astype(np.float32),
                    colors=hex_colors(sel_pts[:, 3:6]), anchor=anchor,
                    n_sel=n_target, n_cloud=len(cloud), near_frac=near_frac, phase=phase,
                ))

            fig = build_figure(method, frames_data, ep, task, args)
            out = args.out_dir / f"{args.out_prefix}_ep{ep:04d}_{method}.html"
            fig.write_html(str(out), include_plotlyjs=(True if args.plotlyjs == "inline" else "cdn"),
                           full_html=not args.fragment, auto_play=False)
            mb = out.stat().st_size / 1e6
            near = [fd["near_frac"] for fd in frames_data if fd["near_frac"] is not None]
            near_txt = f"{np.mean(near):.1%}" if near else "n/a"
            print(f"[{method:7s}] {len(frames_data)} frames -> {out} ({mb:.1f} MB) "
                  f"mean budget within {args.near_radius:g} m of handle: {near_txt}")
            written.append(out)

    env.close()
    print(f"done: {len(written)} file(s) in {args.out_dir}")


if __name__ == "__main__":
    main()
