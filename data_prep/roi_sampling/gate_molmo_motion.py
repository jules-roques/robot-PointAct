"""Stage 4.0 gate: does MolmoMotion's gripper forecast match what the demo actually does?

Asks one question, on **training** data, before anything is trained: given three history
frames and the episode's own instruction, does a frozen MolmoMotion-4B-H3-F30 predict where
the gripper actually goes?

This gate is stronger than the Stage 3 MolmoPoint one, and that is the point of running it
first. MolmoPoint had to be scored against hand-dumped simulator labels that only two of the
three tasks even have (see ``eval_molmo_accuracy.py``). Here the ground truth is **free and
exact**: the gripper's future position is just ``observation.state[:3]`` a few frames later,
in every episode of every task. So the forecast can be scored directly, in metres, with no
labelling step and no privileged information.

What it reports, per task and per history stride:

* **Error against the true future gripper track**, in cm, at fixed wall-clock horizons.
* **The static baseline** -- "the gripper stays exactly where it is" -- at the same horizons.
  This is the number that decides whether the model is worth its forwards at all. If the
  gripper only moves 3 cm in a second, a 4 cm forecast error is not a good forecast, it is a
  worse-than-nothing one, and no amount of plausible-looking video will change that. Stage 3
  had no such control and that is part of why a broken arm read as a plausible one.
* **Win rate**: the share of forecasts strictly closer to the truth than the static baseline.
* A per-replan overlay video: the predicted track and the true track drawn on the same
  frame, so a human can see *how* it fails, not just that it does.

The history-stride sweep is a measurement, not a knob to tune. The checkpoint was trained at
15 fps and our data is 20 fps, and the processor gives the model no timestamps to reconcile
them (see :mod:`pointact.roi_sampling.molmo_motion`). Rather than assume a mapping, the gate
runs each candidate stride and reports which one tracks ground truth. Whichever wins is then
the one an anchor arm would have to use.

Only the **left agentview** is used by default. MolmoMotion is single-camera by
construction, so a second view is a second forward rather than a free extra image the way it
was for MolmoPoint. ``--views left right`` measures whether that cost buys anything.

Because the agentviews are bolted to the robot base, ``base2cam`` is constant, so
"camera-frame-at-t0" is just the camera frame -- no per-frame extrinsics, unlike the wrist.

Runs in ``envs/molmo_motion`` on a GPU: unlike the Stage 3 gate, there is no cache to read
from, so the model runs here.

Usage:
    uv run --project envs/molmo_motion --no-sync python -m data_prep.roi_sampling.gate_molmo_motion \
        --dataset-root $SCRATCH/datasets/robot_data/robocasa365/lerobot_point_lmdb \
        --model-dir $SCRATCH/models/MolmoMotion-4B-H3-F30 \
        --out-dir $SCRATCH/viz/molmo_motion_gate --wandb
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

from pointact.roi_sampling.geometry import (
    base_to_camera,
    camera_to_base,
    project_base_points,
)
from pointact.roi_sampling.molmo_motion import (
    FUTURE_HORIZON,
    HISTORY_SIZE,
    history_frame_indices,
)

#: Dataset frame rate. The checkpoint's native 15 fps is not representable on this grid,
#: which is what the stride sweep exists to work around.
FPS = 20

DEFAULT_TASKS = ("OpenDrawer", "TurnOnMicrowave", "PickPlaceCounterToStove")

#: Candidate history spacings, in dataset frames. 1 is the densest the data allows (0.10 s
#: of context, against the 0.133 s the model saw in training); 2 overshoots it (0.20 s) but
#: reaches a 3.0 s horizon. The truth is between them and cannot be sampled exactly, so both
#: are run.
DEFAULT_HISTORY_STRIDES = (1, 2)

#: Wall-clock horizons to report, in seconds. Chosen so every stride can be compared at the
#: same real time rather than at the same output index -- an output step means a different
#: duration at each stride, and comparing across them by index would be meaningless.
HORIZONS_S = (0.25, 0.5, 1.0, 1.5, 2.0)

PRED_COLOR = (255, 45, 149)   # magenta: forecast
TRUE_COLOR = (60, 220, 120)   # green: what actually happened
EEF_COLOR = (255, 210, 40)    # amber: the query point at t0


def read_meta(meta_dir: Path) -> tuple[dict[int, int], dict[int, str]]:
    """(episode length, instruction) per episode index."""
    lengths, tasks = {}, {}
    with open(meta_dir / "episodes.jsonl") as f:
        for line in f:
            r = json.loads(line)
            ep = int(r["episode_index"])
            lengths[ep] = int(r["length"])
            tk = r.get("tasks") or [""]
            tasks[ep] = tk[0] if tk else ""
    return lengths, tasks


def read_episode_states(dataset_dir: Path, ep: int, chunks_size: int) -> np.ndarray:
    """(T, 16) proprio; [0:3] eef position in the robot-base frame."""
    f = dataset_dir / "data" / f"chunk-{ep // chunks_size:03d}" / f"episode_{ep:06d}.parquet"
    return np.stack(pq.read_table(str(f)).column("observation.state").to_pylist()).astype(np.float64)


def load_calib(path: Path) -> tuple[dict[str, dict], tuple[int, int]]:
    """Static agentview cameras by name, plus the image size they were measured at."""
    d = np.load(path, allow_pickle=True)
    cams = {v: {"name": v, "intrinsic": d[f"{v}_K"], "base2cam": d[f"{v}_base2cam"]}
            for v in ("left", "right")}
    return cams, tuple(int(x) for x in d["image_hw"])


def decode_video(path: Path) -> list[np.ndarray]:
    """Decode an mp4 to RGB uint8 frames in stored-frame orientation (no flip)."""
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def draw_disc(img: np.ndarray, x: float, y: float, color, radius: int = 3) -> None:
    """Filled square marker, clipped to the image. Deliberately not anti-aliased."""
    h, w = img.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - radius), min(w, xi + radius + 1)
    y0, y1 = max(0, yi - radius), min(h, yi + radius + 1)
    if x0 < x1 and y0 < y1:
        img[y0:y1, x0:x1] = color


def draw_polyline(img: np.ndarray, uv: np.ndarray, color, radius: int = 1) -> None:
    """Draw a track as exactly one marker per step -- no interpolation.

    An earlier version also marked the midpoint between consecutive steps, which made the
    track read as a continuous line but drew 2N-1 blobs for N predicted steps. That is
    actively misleading when the point of the overlay is to see what the model predicted:
    the eye counts marks and infers a horizon. One mark per step, so the visible density is
    the prediction.
    """
    for i in range(len(uv)):
        draw_disc(img, uv[i, 0], uv[i, 1], color, radius)


def write_video(path: Path, images: list[np.ndarray], fps: int = 4) -> Path | None:
    """Encode RGB frames to h264. Returns None when there is nothing to write."""
    if not images:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as out:
        stream = out.add_stream("libx264", rate=fps)
        stream.width, stream.height = images[0].shape[1], images[0].shape[0]
        stream.pix_fmt = "yuv420p"
        for img in images:
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return path


def horizon_indices(stride: int, n_future: int) -> dict[float, int]:
    """Wall-clock horizon (s) -> forecast step index, for horizons this stride can reach.

    Output step ``f`` (1-based) lands ``f * stride`` dataset frames ahead, so a stride of 1
    reaches 30/20 = 1.5 s and a stride of 2 reaches 3.0 s. Horizons beyond a stride's reach
    are omitted rather than clamped -- clamping would silently compare a 1.5 s prediction
    against a 2.0 s ground truth and flatter the shorter stride.

    A horizon is also omitted when it is not an exact whole number of steps at this stride.
    0.25 s is 2.5 steps at stride 2; rounding it to 2 would report a 0.2 s prediction in a
    row labelled 0.25 s, and the cross-stride comparison this table exists for would be
    silently off by 20%. Dropping the row loses one cell; rounding it corrupts a column.
    """
    out = {}
    for sec in HORIZONS_S:
        f_exact = sec * FPS / stride
        f = int(round(f_exact))
        if abs(f_exact - f) > 1e-9:
            continue
        if 1 <= f <= n_future:
            out[sec] = f
    return out


def fuse_views(records: list[dict], name: str = "fused") -> list[dict]:
    """Average the per-view predicted positions of every sample seen from >1 view.

    MolmoMotion is single-camera, so two views mean two independent forwards on the same
    instant. If their mistakes were independent, averaging would cut the error by up to
    sqrt(2); if they share a systematic bias -- e.g. both under-shoot the motion, which is
    what an error growing in step with the travelled distance looks like -- the error vectors
    are parallel and averaging changes nothing. This measures which regime we are in.

    The mean is taken in the **robot-base frame**, where both views' predictions already
    live, so no camera convention leaks into the fusion. Samples seen from only one view are
    dropped rather than passed through, so the fused rows are never a mix of fused and
    single-view predictions.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        if "pred_base" not in r:
            continue
        groups.setdefault(
            (r["task"], r["episode"], r["t0"], r["stride"], r["horizon_s"]), []).append(r)

    fused = []
    for (task, ep, t0, stride, sec), rs in groups.items():
        if len(rs) < 2:
            continue
        pred = np.mean([np.asarray(r["pred_base"], dtype=np.float64) for r in rs], axis=0)
        truth = np.asarray(rs[0]["truth_base"], dtype=np.float64)
        fused.append({
            "task": task, "view": name, "episode": ep, "t0": t0,
            "stride": stride, "horizon_s": sec,
            "err_m": float(np.linalg.norm(pred - truth)),
            "static_err_m": float(rs[0]["static_err_m"]),
            "pred_base": pred.tolist(), "truth_base": truth.tolist(),
        })
    return fused


def summarise(records: list[dict]) -> list[dict]:
    """Collapse per-sample errors into a per (task, view, stride, horizon) table."""
    rows = []
    keys = sorted({(r["task"], r["view"], r["stride"], r["horizon_s"]) for r in records})
    for task, view, stride, sec in keys:
        sel = [r for r in records
               if r["task"] == task and r["view"] == view
               and r["stride"] == stride and r["horizon_s"] == sec]
        err = np.array([r["err_m"] for r in sel])
        static = np.array([r["static_err_m"] for r in sel])
        rows.append({
            "task": task,
            "view": view,
            "history_stride": stride,
            "horizon_s": sec,
            "n": len(sel),
            "median_err_cm": float(np.median(err) * 100),
            "mean_err_cm": float(err.mean() * 100),
            "p90_err_cm": float(np.percentile(err, 90) * 100),
            # The distance the gripper actually travelled == the static baseline's error.
            "median_motion_cm": float(np.median(static) * 100),
            "static_median_err_cm": float(np.median(static) * 100),
            # The only line that matters: does forecasting beat assuming nothing moves?
            "win_rate_vs_static": float((err < static).mean()),
            "median_err_ratio": float(np.median(err) / max(1e-9, np.median(static))),
        })
    return rows


def process_episode(
    forecaster,
    dataset_dir: Path,
    ep: int,
    states: np.ndarray,
    instruction: str,
    cam: dict,
    view: str,
    frames: list[np.ndarray],
    strides: tuple[int, ...],
    replan_stride: int,
    n_future: int,
    viz_stride: int | None,
    reduce: str = "median",
) -> tuple[list[dict], dict[int, list[np.ndarray]], dict[str, int]]:
    """Score one episode at every history stride; optionally render one stride's overlay."""
    K, base2cam = cam["intrinsic"], cam["base2cam"]
    eef_base = states[:, :3]
    n_use = min(len(states), len(frames))
    records: list[dict] = []
    viz_frames: dict[int, list[np.ndarray]] = {}
    n_attempt = n_parse_fail = 0

    # Camera-frame EEF for the whole episode at once: base2cam is constant for an agentview,
    # so camera-frame-at-t0 is the camera frame and this does not need redoing per t0.
    eef_cam_all = base_to_camera(eef_base, base2cam)

    for stride in strides:
        h_idx_map = horizon_indices(stride, n_future)
        if not h_idx_map:
            continue
        overlay: list[np.ndarray] = []
        for t0 in range(0, n_use, replan_stride):
            hist = history_frame_indices(t0, HISTORY_SIZE, stride)
            if hist is None or hist[-1] >= n_use:
                continue
            # Need at least the shortest horizon's ground truth to score anything.
            if t0 + min(h_idx_map.values()) * stride >= n_use:
                continue

            uv0, _z, in_front = project_base_points(eef_base[t0:t0 + 1], K, base2cam)
            if not bool(in_front[0]):
                # The gripper is behind the camera plane: the model would get a query pixel
                # that does not correspond to it. Skip rather than feed a bogus point.
                continue

            n_attempt += 1
            pred_cam = forecaster.forecast_point(
                history_frames=[frames[i] for i in hist],
                point_2d_at_t0=uv0[0],
                point_3d_history_cam=eef_cam_all[hist],
                action=instruction,
                future_horizon=n_future,
                reduce=reduce,
            )
            if pred_cam is None:
                # No parseable tracks. Counted, never scored: the zeros the model returns in
                # this case are not a bad prediction, they are the absence of one, and
                # folding them into the error would report a confident forecast at the
                # camera origin.
                n_parse_fail += 1
                continue
            pred_base = camera_to_base(pred_cam, base2cam)  # (F, 3)

            for sec, f in h_idx_map.items():
                t_gt = t0 + f * stride
                if t_gt >= n_use:
                    continue
                if f > len(pred_base):
                    # The model may stop before the requested horizon; a short track is not
                    # an error, but it cannot be scored at horizons it never reached.
                    continue
                truth = eef_base[t_gt]
                records.append({
                    "task": dataset_dir.name, "view": view, "episode": ep, "t0": t0,
                    "stride": stride, "horizon_s": sec,
                    "err_m": float(np.linalg.norm(pred_base[f - 1] - truth)),
                    # The static baseline predicts the gripper does not move at all.
                    "static_err_m": float(np.linalg.norm(eef_base[t0] - truth)),
                    # Base-frame positions, not just the error magnitude. Required to fuse
                    # views after the fact: averaging predictions needs the vectors, and a
                    # run that stored only distances cannot be re-analysed without paying
                    # for the forwards again.
                    "pred_base": pred_base[f - 1].tolist(),
                    "truth_base": truth.tolist(),
                })

            if viz_stride is not None and stride == viz_stride:
                overlay.append(render_overlay(
                    frames[t0], eef_base, pred_base, t0, stride, n_use, K, base2cam))
        if overlay:
            viz_frames[stride] = overlay
    return records, viz_frames, {"attempts": n_attempt, "parse_fail": n_parse_fail}


def render_overlay(
    frame: np.ndarray,
    eef_base: np.ndarray,
    pred_base: np.ndarray,
    t0: int,
    stride: int,
    n_use: int,
    K: np.ndarray,
    base2cam: np.ndarray,
    scale: int = 3,
) -> np.ndarray:
    """One replan frame with the forecast and the truth drawn over it.

    Both tracks are projected through the same calibration, so a visible gap between them is
    a real 3D disagreement and not a rendering artefact.
    """
    img = np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)

    true_idx = [t0 + f * stride for f in range(1, len(pred_base) + 1)]
    true_idx = [i for i in true_idx if i < n_use]
    if true_idx:
        uv_true, _z, ok = project_base_points(eef_base[true_idx], K, base2cam)
        draw_polyline(img, uv_true[ok] * scale, TRUE_COLOR)

    uv_pred, _z, ok = project_base_points(pred_base, K, base2cam)
    draw_polyline(img, uv_pred[ok] * scale, PRED_COLOR)

    uv0, _z, _ok = project_base_points(eef_base[t0:t0 + 1], K, base2cam)
    draw_disc(img, uv0[0, 0] * scale, uv0[0, 1] * scale, EEF_COLOR, radius=3)
    return img


def self_test() -> None:
    """Check the horizon arithmetic and the summary, which need neither GPU nor dataset.

    These two are where a silent error would be most expensive: a wrong horizon index scores
    the forecast against the wrong moment in time, and every downstream number inherits it.
    """
    # Stride 1 reaches 30 frames = 1.5 s, so the 2.0 s horizon must be absent rather than
    # clamped -- clamping would score a 1.5 s prediction against 2.0 s of ground truth.
    h1 = horizon_indices(1, 30)
    assert h1[0.5] == 10 and h1[1.0] == 20 and h1[1.5] == 30, h1
    assert 2.0 not in h1, "stride 1 cannot reach 2.0 s and must not pretend to"
    # Stride 2 reaches 60 frames = 3.0 s, so every horizon is available and each is half the
    # step index of stride 1 -- same wall-clock, different output index.
    h2 = horizon_indices(2, 30)
    assert h2[0.5] == 5 and h2[1.0] == 10 and h2[2.0] == 20, h2
    # 0.25 s is 2.5 steps at stride 2 -- not representable, so it must be absent rather than
    # rounded to a row that claims 0.25 s and measures 0.2 s.
    assert 0.25 not in h2, "a non-representable horizon must be dropped, not rounded"
    # Wherever both strides report a horizon, they must mean the same wall-clock instant.
    shared = set(h1) & set(h2)
    assert shared >= {0.5, 1.0, 1.5}, shared
    assert all(h2[s] * 2 == h1[s] for s in shared), "shared horizons must agree in wall-clock"
    # A horizon below one step at this stride must drop out, not round to 0.
    assert 0.25 not in horizon_indices(16, 30), "a horizon below one step must be dropped"

    # summarise: win rate and the ratio must key off the static baseline, not the raw error.
    recs = [
        {"task": "T", "view": "left", "stride": 1, "horizon_s": 1.0,
         "err_m": 0.02, "static_err_m": 0.10},   # forecast much better than static
        {"task": "T", "view": "left", "stride": 1, "horizon_s": 1.0,
         "err_m": 0.30, "static_err_m": 0.10},   # forecast much worse
    ]
    (row,) = summarise(recs)
    assert row["n"] == 2 and row["win_rate_vs_static"] == 0.5, row
    assert abs(row["median_err_cm"] - 16.0) < 1e-6, row
    assert abs(row["median_motion_cm"] - 10.0) < 1e-6, row
    # A forecast that loses to "nothing moves" must show a ratio above 1, which is the single
    # number that says the model is not worth its forwards.
    assert row["median_err_ratio"] > 1.0, row

    # Geometry sanity: a track projected and scored must survive the frame round-trip, or a
    # real 3D agreement could read as a disagreement in the overlay.
    b2c = np.eye(4)
    b2c[:3, 3] = [0.1, -0.2, 0.8]
    pts = np.array([[0.3, 0.1, 0.2], [0.35, 0.12, 0.25]])
    assert np.allclose(camera_to_base(base_to_camera(pts, b2c), b2c), pts, atol=1e-12)

    # fuse_views: the two regimes that decide whether a second view is worth its forwards.
    def _rec(view, pred, t0=0):
        return {"task": "T", "view": view, "episode": 0, "t0": t0, "stride": 2,
                "horizon_s": 1.0, "err_m": 0.0, "static_err_m": 0.10,
                "pred_base": pred, "truth_base": [0.0, 0.0, 0.0]}

    # Opposed errors: averaging cancels them exactly.
    (f,) = fuse_views([_rec("left", [0.1, 0, 0]), _rec("right", [-0.1, 0, 0])])
    assert f["err_m"] < 1e-12 and f["view"] == "fused", f
    # Identical (systematically biased) errors: averaging changes nothing at all. This is the
    # regime a shared under-shoot puts us in, and the reason a second view can be worthless.
    (f,) = fuse_views([_rec("left", [0.1, 0, 0]), _rec("right", [0.1, 0, 0])])
    assert abs(f["err_m"] - 0.1) < 1e-12, f
    # The static baseline must be carried through unchanged: fusing predictions must not
    # quietly redefine what they are compared against.
    assert abs(f["static_err_m"] - 0.10) < 1e-12, f
    # A sample seen from one view only is not a fusion and must be dropped, not passed
    # through as if it had been.
    assert fuse_views([_rec("left", [0.1, 0, 0])]) == []

    # draw_polyline must draw exactly one marker per step. It used to interpolate midpoints,
    # which drew 2N-1 blobs for N steps and made the overlays read as a longer forecast than
    # they were. Well-separated points, radius 1 -> a 3x3 block each, none overlapping.
    canvas = np.zeros((60, 60, 3), dtype=np.uint8)
    track = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]])
    draw_polyline(canvas, track, (255, 0, 255))
    lit = int((canvas.any(axis=2)).sum())
    assert lit == 9 * len(track), f"expected {9 * len(track)} lit pixels, got {lit}"
    # Distinct samples must not be pooled into one another.
    assert len(fuse_views([_rec("left", [0.1, 0, 0], t0=0), _rec("right", [0, 0, 0], t0=0),
                           _rec("left", [0.1, 0, 0], t0=8), _rec("right", [0, 0, 0], t0=8)])) == 2
    print("gate self-test OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true",
                    help="Check the horizon arithmetic and summary; no GPU or dataset needed.")
    ap.add_argument("--dataset-root", type=Path)
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--tasks", nargs="*", default=list(DEFAULT_TASKS))
    ap.add_argument("--episodes", type=int, default=8,
                    help="Episodes per task. 8 is a gate, not a benchmark.")
    ap.add_argument("--episode-list", nargs="*", type=int, default=None)
    ap.add_argument("--views", nargs="*", default=["left"],
                    help="MolmoMotion is single-camera, so each extra view is a full extra "
                         "forward per frame -- not free the way MolmoPoint's were.")
    ap.add_argument("--history-strides", nargs="*", type=int,
                    default=list(DEFAULT_HISTORY_STRIDES))
    ap.add_argument("--replan-stride", type=int, default=8,
                    help="Match the policy's replan cadence: the gate should score the "
                         "forecast at the frames an anchor arm would actually request one.")
    ap.add_argument("--future-horizon", type=int, default=FUTURE_HORIZON,
                    help="The checkpoint's native 30 by default. Lower it if the gate "
                         "reports a high parse-failure rate: the token budget is 160x this "
                         "and the context window is 2560, so a long horizon can truncate.")
    ap.add_argument("--num-points", type=int, default=None,
                    help="Override config.num_points (default 8). It is a config field, not "
                         "a weight shape; 1 asks for the single track we actually want and "
                         "is far cheaper on this decode-bound model.")
    ap.add_argument("--reduce", choices=("median", "first"), default="median",
                    help="How to collapse the checkpoint's 8 fixed point slots, which all "
                         "carry the same replicated gripper query.")
    ap.add_argument("--viz-stride", type=int, default=None,
                    help="History stride to render overlays for (default: the first one).")
    ap.add_argument("--viz-episodes", type=int, default=2, help="Episodes to render per task.")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="pointact-robocasa365")
    ap.add_argument("--wandb-entity", default="diffusion4robots")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    missing = [n for n in ("dataset_root", "model_dir", "out_dir") if getattr(args, n) is None]
    if missing:
        ap.error("required unless --self-test: " + ", ".join("--" + m.replace("_", "-")
                                                             for m in missing))

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_stride = args.viz_stride if args.viz_stride is not None else args.history_strides[0]

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name="stage4.0-gate-molmomotion", job_type="viz",
            # Same stage tag the arms will carry, so one filter returns the gate together
            # with whatever it gated; `gate` is what tells them apart (see
            # robocasa365-experiment-conventions).
            tags=["Stage 4: MolmoMotion trajectory anchor", "gate"],
            config={
                "model": str(args.model_dir), "tasks": args.tasks,
                "episodes_per_task": args.episodes, "views": args.views,
                "history_strides": args.history_strides,
                "replan_stride": args.replan_stride, "future_horizon": args.future_horizon,
                "exp_stage": "Stage 4: MolmoMotion trajectory anchor",
            },
        )

    from pointact.roi_sampling.molmo_motion import MolmoMotionForecaster
    forecaster = MolmoMotionForecaster(str(args.model_dir), future_horizon=args.future_horizon,
                                       num_points=args.num_points)
    print(f"num_points={forecaster.num_points}", flush=True)
    if run is not None:
        run.config["num_points"] = forecaster.num_points

    all_records: list[dict] = []
    media: dict[str, Path] = {}
    attempts = parse_fail = 0
    t_start = time.time()

    for task in args.tasks:
        dataset_dir = args.dataset_root / task
        calib_path = dataset_dir / "roi_meta" / "camera_calib.npz"
        if not calib_path.exists():
            print(f"{task}: no calibration at {calib_path} -- run dump_camera_calib.py first",
                  file=sys.stderr)
            continue
        cams, _image_hw = load_calib(calib_path)
        lengths, instructions = read_meta(dataset_dir / "meta")
        info = json.loads((dataset_dir / "meta" / "info.json").read_text())
        chunks_size = int(info.get("chunks_size", 1000))

        episodes = args.episode_list or sorted(lengths)[: args.episodes]
        for view in args.views:
            if view not in cams:
                print(f"{task}: no {view} camera in the calibration", file=sys.stderr)
                continue
            for n_done, ep in enumerate(episodes):
                video = (dataset_dir / "videos" / f"chunk-{ep // chunks_size:03d}"
                         / f"observation.images.{view}_image" / f"episode_{ep:06d}.mp4")
                if not video.exists():
                    print(f"{task} ep{ep}: no {view} video at {video}", file=sys.stderr)
                    continue
                states = read_episode_states(dataset_dir, ep, chunks_size)
                frames = decode_video(video)
                want_viz = n_done < args.viz_episodes

                recs, viz, counts = process_episode(
                    forecaster, dataset_dir, ep, states, instructions.get(ep, ""),
                    cams[view], view, frames, tuple(args.history_strides),
                    args.replan_stride, args.future_horizon,
                    viz_stride if want_viz else None, args.reduce,
                )
                all_records.extend(recs)
                attempts += counts["attempts"]
                parse_fail += counts["parse_fail"]
                for stride, imgs in viz.items():
                    name = f"{task}_{view}_ep{ep}_s{stride}"
                    p = write_video(out_dir / f"{name}.mp4", imgs)
                    if p is not None:
                        media[name] = p
                print(f"{task} {view} ep{ep}: {len(recs)} scored samples "
                      f"({time.time() - t_start:.0f}s elapsed)", flush=True)

    fail_rate = parse_fail / max(1, attempts)
    if not all_records:
        raise SystemExit(
            f"no samples scored ({attempts} forwards, {parse_fail} unparseable) -- "
            f"check the dataset root and calibration"
            + (", and lower --future-horizon: every forward failed to parse"
               if attempts and parse_fail == attempts else ""))

    # A "fused" pseudo-view, only when more than one real view was scored. It costs no extra
    # forwards: it reuses the predictions already made.
    if len({r["view"] for r in all_records}) > 1:
        fused = fuse_views(all_records)
        print(f"fused {len(fused)} samples across "
              f"{sorted({r['view'] for r in all_records})}")
        all_records.extend(fused)

    rows = summarise(all_records)
    (out_dir / "gate_summary.json").write_text(json.dumps(
        {"forwards": attempts, "parse_failures": parse_fail,
         "parse_fail_rate": fail_rate, "reduce": args.reduce,
         "future_horizon": args.future_horizon, "rows": rows}, indent=2))
    np.savez_compressed(
        out_dir / "gate_records.npz",
        **{k: np.array([r[k] for r in all_records])
           for k in ("episode", "t0", "stride", "horizon_s", "err_m", "static_err_m")},
        task=np.array([r["task"] for r in all_records]),
        view=np.array([r["view"] for r in all_records]),
        # Positions, so a later question about combining views can be answered from the
        # records instead of by re-running the model.
        pred_base=np.array([r.get("pred_base", [np.nan] * 3) for r in all_records]),
        truth_base=np.array([r.get("truth_base", [np.nan] * 3) for r in all_records]),
    )

    print(f"\nforwards={attempts} unparseable={parse_fail} ({fail_rate:.1%})")
    if fail_rate > 0.05:
        print("  WARNING: a high unparseable rate usually means the generation was "
              "truncated -- lower --future-horizon and re-run.")
    print(f"\n{'task':<26}{'view':<6}{'s':<3}{'hor':<6}{'n':<6}"
          f"{'err_cm':<9}{'motion_cm':<11}{'win_vs_static':<14}")
    for r in rows:
        print(f"{r['task']:<26}{r['view']:<6}{r['history_stride']:<3}{r['horizon_s']:<6}"
              f"{r['n']:<6}{r['median_err_cm']:<9.1f}{r['median_motion_cm']:<11.1f}"
              f"{r['win_rate_vs_static']:<14.2f}")

    if run is not None:
        import wandb
        table = wandb.Table(columns=list(rows[0].keys()),
                            data=[list(r.values()) for r in rows])
        log = {"gate/summary": table,
               "gate/forwards": attempts,
               "gate/parse_failures": parse_fail,
               "gate/parse_fail_rate": fail_rate}
        log.update({f"gate/video/{k}": wandb.Video(str(v), format="mp4")
                    for k, v in media.items()})
        # One error-vs-horizon curve per (task, view, stride), against its own static
        # baseline -- the baseline is what makes the curve readable.
        for task in sorted({r["task"] for r in rows}):
            for view in sorted({r["view"] for r in rows if r["task"] == task}):
                for stride in sorted({r["history_stride"] for r in rows
                                      if r["task"] == task and r["view"] == view}):
                    sel = [r for r in rows if r["task"] == task and r["view"] == view
                           and r["history_stride"] == stride]
                    sel.sort(key=lambda r: r["horizon_s"])
                    tbl = wandb.Table(
                        columns=["horizon_s", "median_err_cm", "static_median_err_cm"],
                        data=[[r["horizon_s"], r["median_err_cm"],
                               r["static_median_err_cm"]] for r in sel])
                    log[f"gate/curve/{task}_{view}_s{stride}"] = tbl
        run.log(log)
        run.finish()

    print(f"\nArtifacts: {out_dir}")


if __name__ == "__main__":
    main()
