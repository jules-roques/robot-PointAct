"""Animate one episode's sampled point cloud, one HTML per sampling strategy.

A play/pause + slider animation over the frames of a single episode, so you can watch where
the fixed 4096-point budget goes as the arm approaches, grasps and manipulates its target.

Strategies (each writes its own self-contained HTML, matching the dataloader exactly):

  uniform  ``np.random.choice`` over the whole cloud — the baseline.
  eef      ``density_weighted_indices`` on ``eef_density_weights``: a Gaussian bump around
           the frame's own end-effector (``observation.state[:3]``) with a density floor.
  molmo    The same Gaussian-with-floor draw as ``eef``, centred on the anchor(s) a frozen
           MolmoPoint model put on the task's object(s), read from the cache built by
           ``build_molmo_cache.py``. With two queries selected (object + destination) the
           weights take their max, so the budget is shared between the two regions.
  oracle   The variant actually trained (``data-robocasa365-opendrawer-point-oracle.yaml``):
           the anchor is the centroid of the points MuJoCo *labels* as the handle (label 4,
           falling back to the door panel, label 3, when the handle is occluded), and the draw
           is the same Gaussian-with-floor density as ``eef`` — only the centre differs.
           Needs the sibling ``points_3views_labels`` LMDB.

Every anchor a strategy is centred on is drawn as a marker, so "where the budget went" is
visible rather than inferred.

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

from pointact.roi_sampling import molmo_cache
from pointact.roi_sampling.geometry import eef_density_weights
from pointact.roi_sampling.sampling import density_weighted_indices

msgpack_numpy.patch()

METHODS = ("uniform", "eef", "molmo", "oracle")

#: What ``--method all`` builds. ``molmo`` is excluded because it needs a cache that only
#: exists once build_molmo_cache.py has run for the dataset; ask for it by name.
DEFAULT_METHODS = ("uniform", "eef", "oracle")

METHOD_LABEL = {
    "uniform": "Uniform (baseline)",
    "eef": "EEF-density (Gaussian + floor around the end-effector)",
    "molmo": "MolmoPoint (Gaussian on the frozen pointer's detection)",
    "oracle": "Oracle GT (Gaussian on the MuJoCo-labelled handle)",
}

ANCHOR_LABEL = {"uniform": "end-effector (reference only)", "eef": "end-effector (anchor)",
                "molmo": "MolmoPoint detection (anchor)", "oracle": "GT handle centroid (anchor)"}
ANCHOR_COLOR = {"uniform": "#9aa0a6", "eef": "#ff2d95", "molmo": "#ff2d95", "oracle": "#ff2d95"}

#: MuJoCo segmentation levels written by replay.py --labels-only (see the oracle data config).
LABEL_NAMES = {0: "background", 1: "robot", 2: "target fixture", 3: "door panel", 4: "handle"}

#: Colour axis for the sampling weight, in log2(× uniform). Diverging about 0 = ×1, so a
#: uniform draw reads as one neutral tone and any concentration reads as warmth. Stops are
#: bright enough to stay legible on the dark ground.
#: Bounds track the measured spread on OpenDrawer (log2 ratio runs about -0.8 .. +3.9 with
#: sigma=0.08, floor=0.05), so the ramp is spent on the range the data actually occupies.
CLOG_MIN, CLOG_MAX = -1.0, 4.0
WEIGHT_COLORSCALE = [
    [0.00, "#2f6fb0"],  # x0.5 - given up
    [0.12, "#8fb4d4"],
    [0.20, "#dbe3ea"],  # x1   - same as uniform
    [0.40, "#ffd166"],
    [0.70, "#ff9a3c"],
    [1.00, "#ff3b6b"],  # x16  - heavily concentrated
]
WEIGHT_TICKS = [-1, 0, 1, 2, 3, 4]
WEIGHT_TICKTEXT = ["×0.5", "×1", "×2", "×4", "×8", "×16"]


def load_cloud(txn, ep: int, f: int) -> np.ndarray | None:
    buf = txn.get(f"{ep}-{f}".encode("ascii"))
    return None if buf is None else msgpack.unpackb(bytes(buf)).astype(np.float32)


def open_labels_env(dataset_dir: Path, dirname: str):
    """Open the MuJoCo label LMDB, or None when it has not been built.

    ``lock=True`` (unlike the points env) registers this process in the reader table, so a
    concurrently running ``export_point_labels`` writer cannot reclaim pages out from under
    the read — with ``lock=False`` a partially built cache raises MDB_PAGE_NOTFOUND mid-scan.
    """
    p = dataset_dir / dirname
    if not p.exists():
        return None
    return lmdb.open(str(p), readonly=True, lock=True, readahead=False)


def load_labels(txn, ep: int, f: int) -> np.ndarray | None:
    buf = txn.get(f"{ep}-{f}".encode("ascii"))
    return None if buf is None else msgpack.unpackb(bytes(buf)).astype(np.uint8)


def oracle_anchor(
    cloud: np.ndarray, labels: np.ndarray | None,
    anchor_labels: tuple[int, ...], fallback_labels: tuple[int, ...],
) -> np.ndarray | None:
    """Centroid of the GT handle points, else of the door panel, else None.

    Mirrors ``LeRobotPointCloudDataset.oracle_anchor``: the handle is a compact few-cm
    cluster so its centroid is a good Gaussian centre, but it can be occluded by the gripper
    or face away from every camera, in which case the panel it sits on still gives guidance.
    """
    if labels is None:
        return None
    for group in (anchor_labels, fallback_labels):
        if not group:
            continue
        mask = np.isin(labels, group)
        if mask.any():
            return cloud[mask, :3].mean(axis=0).astype(np.float64)
    return None


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


def compute_weights(method: str, cloud: np.ndarray, anchor: np.ndarray | None, args):
    """Per-point sampling weight for this frame, or None when the draw is plain uniform."""
    if anchor is None:
        return None
    if method == "eef":
        return eef_density_weights(cloud[:, :3], anchor, args.eef_sigma, args.eef_floor)
    if method == "oracle":
        return eef_density_weights(cloud[:, :3], anchor, args.oracle_sigma, args.oracle_floor)
    if method == "molmo":
        return eef_density_weights(cloud[:, :3], anchor, args.molmo_sigma, args.molmo_floor)
    return None


def select_indices(
    method: str,
    cloud: np.ndarray,
    n_target: int,
    weights: np.ndarray | None,
    args,
    rng: np.random.Generator,
) -> np.ndarray:
    """The dataloader's own selection, for one frame (see data_3d.augment_point_cloud)."""
    m = len(cloud)
    if weights is not None:
        if method in ("eef", "oracle", "molmo"):
            return density_weighted_indices(m, n_target, weights, rng)
    return rng.choice(m, n_target, replace=False)


def oversampling_factor(weights: np.ndarray | None, n_points: int) -> np.ndarray:
    """Per-point keep-probability as a multiple of the uniform draw's.

    Raw weights are only defined up to a scale factor, so plotting them directly would make
    the colour scale arbitrary and non-comparable between strategies. Normalising by the
    uniform probability 1/N fixes that: ``N * w / sum(w)`` is exactly ×1 everywhere for a
    uniform draw, <1 where a strategy gives up points, and >1 where it concentrates them.
    """
    if weights is None:
        return np.ones(n_points, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return np.ones(n_points, dtype=np.float64)
    return n_points * w / s


def build_figure(method: str, frames_data: list[dict], ep: int, task: str, args) -> go.Figure:
    """Assemble the animated scatter (slider + play/pause) for one strategy."""
    first = frames_data[0]
    all_xyz = np.concatenate([fd["pts"] for fd in frames_data], axis=0)
    lo, hi = all_xyz.min(axis=0), all_xyz.max(axis=0)
    pad = 0.05 * float(np.max(hi - lo))
    ranges = [[float(lo[i] - pad), float(hi[i] + pad)] for i in range(3)]

    def traces_for(fd: dict) -> list[go.Scatter3d]:
        pts = fd["pts"]
        if args.color_by == "weight":
            marker = dict(
                size=args.point_size, color=fd["colors"], colorscale=WEIGHT_COLORSCALE,
                cmin=CLOG_MIN, cmax=CLOG_MAX,
                colorbar=dict(
                    title=dict(text="sampling<br>weight", side="right"),
                    tickvals=WEIGHT_TICKS, ticktext=WEIGHT_TICKTEXT,
                    len=0.55, thickness=13, x=0.99, xanchor="right", y=0.55,
                    outlinewidth=0, tickfont=dict(size=10),
                ),
            )
        else:
            marker = dict(size=args.point_size, color=fd["colors"])
        out = [go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers", marker=marker,
            name=f"sampled points ({fd['n_sel']})", hoverinfo="skip",
        )]
        # (K, 3): the molmo arm can carry two anchors (object + destination); the others
        # carry one. A NaN row keeps the trace present so the animation frames stay aligned.
        a = fd["anchor"]
        a = np.full((1, 3), np.nan) if a is None else np.atleast_2d(a)
        out.append(go.Scatter3d(
            x=a[:, 0], y=a[:, 1], z=a[:, 2],
            mode="markers",
            marker=dict(size=6, color=ANCHOR_COLOR[method], symbol="diamond",
                        line=dict(width=1, color="#ffffff")),
            name=ANCHOR_LABEL[method], hoverinfo="skip",
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
    # Both stats need the MuJoCo labels, which only OpenDrawer has. Dropping the phrase beats
    # printing "n/a ... of the handle" on a microwave button that has no handle at all.
    if fd["near_frac"] is not None:
        near = f"{fd['near_frac']:.0%} within {args.near_radius:g} m of the handle · "
    else:
        near = ""
    if fd["n_handle_total"]:
        handle = (f"GT handle points kept {fd['n_handle_sel']}/{fd['n_handle_total']}"
                  f" ({fd['handle_recall']:.0%}) · ")
    else:
        handle = ""
    return (f"<b>{METHOD_LABEL[method]}</b><br>"
            f"<sup>ep {ep} · frame {fd['frame']}/{fd['n_frames'] - 1} · “{task}” · "
            f"{fd['n_sel']} of {fd['n_cloud']} pts · {near}{handle}"
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
    ap.add_argument("--color-by", default="weight", choices=("weight", "flat", "rgb"),
                    help="'weight': colour each point by the sampling weight it was drawn with, "
                         "as a multiple of the uniform draw's probability (×1 = uniform). "
                         "'flat': one colour (--point-color). 'rgb': the point's true colour, "
                         "faithful but dark surfaces vanish against a dark ground.")
    ap.add_argument("--point-color", default="#4ade80",
                    help="Flat CSS colour used when --color-by flat.")
    ap.add_argument("--frame-ms", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dark", action="store_true", help="Dark plotly template.")
    ap.add_argument("--emit-json", action="store_true",
                    help="Also write the figure as JSON beside the html. write_html bundles a ~3.5 MB copy of plotly.js per file; a page showing several figures wants one shared copy and the data on its own.")
    ap.add_argument("--near-radius", type=float, default=0.15,
                    help="Radius used for the reported 'fraction of budget near the handle' stat.")
    # eef-density knobs (defaults match the data config)
    ap.add_argument("--eef-sigma", type=float, default=0.08)
    ap.add_argument("--eef-floor", type=float, default=0.05)
    # MolmoPoint knobs (defaults match the stage-3 data configs)
    ap.add_argument("--molmo-anchor-dirname", default="points_3views_molmo")
    ap.add_argument("--molmo-anchor-ids", nargs="*", type=int, default=[0],
                    help="Which pointing queries become Gaussian centres: 0 = the manipulated "
                         "object, 1 = the destination. '0 1' is the two-region arm.")
    ap.add_argument("--molmo-sigma", type=float, default=0.08)
    ap.add_argument("--molmo-floor", type=float, default=0.05)
    # Oracle-GT knobs (defaults match data-robocasa365-opendrawer-point-oracle.yaml)
    ap.add_argument("--labels-dirname", default="points_3views_labels")
    ap.add_argument("--oracle-anchor-labels", nargs="*", type=int, default=[4])
    ap.add_argument("--oracle-fallback-labels", nargs="*", type=int, default=[3])
    ap.add_argument("--oracle-sigma", type=float, default=0.08)
    ap.add_argument("--oracle-floor", type=float, default=0.05)
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
    closed = actions[:, 7] > 0.5 if actions.ndim == 2 and actions.shape[1] > 7 else None

    stride = args.stride if args.stride > 0 else max(1, n_frames // max(1, args.num_frames))
    frame_ids = list(range(0, n_frames, stride))
    methods = list(DEFAULT_METHODS) if args.method == "all" else [args.method]

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

        # MuJoCo per-point labels: the oracle's anchor, and the ground truth every strategy is
        # scored against. Absent (or still being exported) -> the oracle method is unavailable
        # and the near-target stats are simply omitted. Only OpenDrawer has these labels, so
        # on the other tasks the animation shows where the budget went without scoring it.
        labels = {}
        lab_env = open_labels_env(d, args.labels_dirname)
        if lab_env is not None:
            with lab_env.begin(buffers=True) as ltxn:
                for f in clouds:
                    lb = load_labels(ltxn, ep, f)
                    if lb is not None and len(lb) == len(clouds[f]):
                        labels[f] = lb
            lab_env.close()
        if "oracle" in methods and not labels:
            raise SystemExit(
                f"method 'oracle' needs {d/args.labels_dirname} (per-point MuJoCo labels, "
                f"built by replay.py --labels-only -> export_point_labels.py); "
                f"no labels found for ep {ep}")
        if labels and len(labels) < len(clouds):
            print(f"note: labels cover {len(labels)}/{len(clouds)} sampled frames "
                  f"(the export may still be running)")

        # MolmoPoint anchors, read with the same key the dataloader uses. Frames the pointer
        # missed are simply absent, and the method falls back to a uniform draw for them —
        # exactly what data_3d.augment_point_cloud does.
        molmo_anchors = {}
        if "molmo" in methods:
            mdir = d / args.molmo_anchor_dirname
            if not mdir.exists():
                raise SystemExit(
                    f"method 'molmo' needs {mdir}, built by "
                    f"data_prep/roi_sampling/build_molmo_cache.py")
            menv = lmdb.open(str(mdir), readonly=True, lock=False, readahead=False)
            with menv.begin(buffers=True) as mtxn:
                for f in clouds:
                    buf = mtxn.get(f"{ep}-{f}".encode("ascii"))
                    if buf is None:
                        continue
                    rec = np.frombuffer(bytes(buf), dtype=molmo_cache.RECORD_DTYPE)
                    a = molmo_cache.decode_anchors(rec, tuple(args.molmo_anchor_ids))
                    if a is not None:
                        molmo_anchors[f] = a
            menv.close()
            print(f"molmo anchors: {len(molmo_anchors)}/{len(clouds)} sampled frames "
                  f"(queries {args.molmo_anchor_ids})")

        for method in methods:
            rng = np.random.default_rng(args.seed)
            disp_rng = np.random.default_rng(args.seed + 1)
            frames_data = []
            for f in sorted(clouds):
                cloud = clouds[f]
                lb = labels.get(f)
                gt_anchor = oracle_anchor(cloud, lb, tuple(args.oracle_anchor_labels),
                                          tuple(args.oracle_fallback_labels))
                anchor = None
                if method == "eef" and f < len(states):
                    anchor = states[f, :3].astype(np.float64)
                elif method == "molmo":
                    anchor = molmo_anchors.get(f)
                elif method == "oracle":
                    anchor = gt_anchor
                elif method == "uniform":  # reference marker only, never used for sampling
                    anchor = states[f, :3].astype(np.float64) if f < len(states) else None

                n_target = int(min(len(cloud), args.max_npoints))
                weights = compute_weights(method, cloud, anchor, args) if method != "uniform" else None
                sel = select_indices(method, cloud, n_target, weights, args, rng)
                sel_pts = cloud[sel]
                # log2 of the keep-probability relative to uniform, for the colour axis.
                sel_logw = np.log2(oversampling_factor(weights, len(cloud))[sel])

                # Stats measured on the FULL selection, before display thinning. The reference
                # is the GT handle centroid where labels exist, so all methods are scored
                # against the same ground truth rather than each against its own anchor.
                ref = gt_anchor
                near_frac = None
                if ref is not None:
                    dist = np.linalg.norm(sel_pts[:, :3].astype(np.float64) - ref, axis=1)
                    near_frac = float((dist <= args.near_radius).mean())

                # How much of the actual handle survived the draw: the sharpest single number,
                # since the handle is only a few dozen points out of ~17k.
                n_handle_total = n_handle_sel = 0
                if lb is not None:
                    handle_mask = np.isin(lb, args.oracle_anchor_labels)
                    n_handle_total = int(handle_mask.sum())
                    n_handle_sel = int(handle_mask[sel].sum())
                recall = (n_handle_sel / n_handle_total) if n_handle_total else 0.0

                if args.display_points and len(sel_pts) > args.display_points:
                    keep = disp_rng.choice(len(sel_pts), args.display_points, replace=False)
                    sel_pts, sel_logw = sel_pts[keep], sel_logw[keep]

                if closed is not None and f < len(closed):
                    phase = "gripper closed" if closed[f] else "gripper open"
                else:
                    phase = "—"

                frames_data.append(dict(
                    frame=f, n_frames=n_frames, pts=sel_pts[:, :3].astype(np.float32),
                    colors=(sel_logw.astype(np.float32) if args.color_by == "weight"
                            else args.point_color if args.color_by == "flat"
                            else hex_colors(sel_pts[:, 3:6])),
                    anchor=anchor,
                    n_sel=n_target, n_cloud=len(cloud), near_frac=near_frac, phase=phase,
                    n_handle_total=n_handle_total, n_handle_sel=n_handle_sel,
                    handle_recall=recall,
                ))

            fig = build_figure(method, frames_data, ep, task, args)
            out = args.out_dir / f"{args.out_prefix}_ep{ep:04d}_{method}.html"
            if args.emit_json:
                jpath = out.with_suffix(".json")
                jpath.write_text(fig.to_json())
                print(f"[{method:7s}] figure json -> {jpath} "
                      f"({jpath.stat().st_size / 1e6:.1f} MB)")
            fig.write_html(str(out), include_plotlyjs=(True if args.plotlyjs == "inline" else "cdn"),
                           full_html=not args.fragment, auto_play=False)
            mb = out.stat().st_size / 1e6
            near = [fd["near_frac"] for fd in frames_data if fd["near_frac"] is not None]
            near_txt = f"{np.mean(near):.1%}" if near else "n/a"
            rec = [fd["handle_recall"] for fd in frames_data if fd["n_handle_total"]]
            rec_txt = f"{np.mean(rec):.1%}" if rec else "n/a"
            print(f"[{method:7s}] {len(frames_data)} frames -> {out} ({mb:.1f} MB) "
                  f"budget within {args.near_radius:g} m of handle: {near_txt} · "
                  f"GT handle points kept: {rec_txt}")
            written.append(out)

    env.close()
    print(f"done: {len(written)} file(s) in {args.out_dir}")


if __name__ == "__main__":
    main()
