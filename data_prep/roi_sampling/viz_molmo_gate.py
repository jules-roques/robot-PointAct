"""Stage 3.0 gate: is MolmoPoint actually pointing at the right thing?

Builds, per task, from **training** data and at the policy's replan cadence:

  * a left-camera video with the model's point drawn on every replan frame,
  * the same for the right camera,
  * the interactive point-cloud animation of the resulting sampling, with the Gaussian
    centre(s) marked (``viz_sampling_episode.py --method molmo``).

and logs them to one W&B run tagged ``Stage 3: MolmoPoint anchor`` -- the same stage tag the
trained arms carry, so one filter returns the gate together with the runs it gated. The
``gate`` tag and ``job_type="viz"`` are what distinguish it from them.

This is a gate, not a report: nothing downstream should be trained until a human has looked
at it. What to look for —

  * does the point land on the object the instruction names?
  * on OpenDrawer, does it pick the **correct** drawer on both left- and right-instruction
    episodes? That is the capability the whole approach rests on, and the one an
    open-vocabulary box detector did not have.
  * does the lifted anchor sit on the object in 3D, or has it slipped onto an occluder?
    The 2D overlay cannot show this; the point-cloud animation can.
  * does it jitter between replans? sigma=0.08 m is forgiving, but a centre that jumps
    between two objects is not jitter, it is ambiguity.
  * for PickPlaceCounterToStove, does adding the pan visibly change where the budget goes?

Runs in the **root** env: it reads the finished cache, so it needs plotly/wandb rather
than the model.

Usage:
    python -m data_prep.roi_sampling.viz_molmo_gate \
        --dataset-root robot_data/robocasa365/lerobot_point_lmdb \
        --out-dir $SCRATCH/viz/molmo_gate --wandb
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import av
import lmdb
import numpy as np

from pointact.roi_sampling import molmo_cache

VIEWS = ("left", "right")

#: One colour per pointing query, so "object" and "destination" stay distinguishable in a
#: single frame of PickPlaceCounterToStove.
QUERY_COLORS = ((255, 45, 149), (80, 200, 255))

#: Which episode to show per task, and the arms to render. PickPlaceCounterToStove gets
#: both so the object-only vs object+pan difference is visible side by side.
DEFAULT_TASKS = {
    "OpenDrawer": {"episode": 0, "arms": {"molmo": [0]}},
    "PickPlaceCounterToStove": {"episode": 0, "arms": {"molmo-obj": [0], "molmo-objpan": [0, 1]}},
    "TurnOnMicrowave": {"episode": 0, "arms": {"molmo": [0]}},
}


def read_meta(meta_dir: Path) -> tuple[dict[int, int], dict[int, str]]:
    lengths, tasks = {}, {}
    with open(meta_dir / "episodes.jsonl") as f:
        for line in f:
            r = json.loads(line)
            ep = int(r["episode_index"])
            lengths[ep] = int(r["length"])
            tk = r.get("tasks") or [""]
            tasks[ep] = tk[0] if tk else ""
    return lengths, tasks


def draw_marker(img: np.ndarray, x: float, y: float, color, scale: int) -> None:
    """A crosshair + ring, drawn with numpy so this needs no PIL/cv2 in the root env."""
    h, w = img.shape[:2]
    xi, yi = int(round(x * scale)), int(round(y * scale))
    r, arm, t = 5 * scale, 9 * scale, max(1, scale // 2)
    for dy in range(-t, t + 1):
        yy = yi + dy
        if 0 <= yy < h:
            img[yy, max(0, xi - arm):min(w, xi + arm + 1)] = color
    for dx in range(-t, t + 1):
        xx = xi + dx
        if 0 <= xx < w:
            img[max(0, yi - arm):min(h, yi + arm + 1), xx] = color
    ang = np.linspace(0, 2 * np.pi, 180)
    for a in ang:
        yy, xx = int(round(yi + r * np.sin(a))), int(round(xi + r * np.cos(a)))
        if 0 <= yy < h and 0 <= xx < w:
            img[yy, xx] = color


def render_view_video(
    dataset_dir: Path, ep: int, view: str, key_frames: list[int],
    pixels: dict[int, list[dict]], out_path: Path, chunks_size: int, scale: int, fps: int,
) -> Path | None:
    """One mp4 of the replan frames with each query's point drawn on it."""
    src = (dataset_dir / "videos" / f"chunk-{ep // chunks_size:03d}"
           / f"observation.images.{view}_image" / f"episode_{ep:06d}.mp4")
    if not src.exists():
        print(f"  no video at {src}", file=sys.stderr)
        return None

    wanted = set(key_frames)
    frames = {}
    with av.open(str(src)) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i in wanted:
                frames[i] = f.to_ndarray(format="rgb24")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(out_path), mode="w") as out:
        stream = None
        for f in key_frames:
            img = frames.get(f)
            if img is None:
                continue
            # Nearest-neighbour upscale: these are 256x256 renders and a point on a drawer
            # handle is a couple of pixels; interpolating would blur exactly what is judged.
            img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
            for det in pixels.get(f, []):
                uv = det.get(f"{view}_uv")
                if uv is not None:
                    draw_marker(img, uv[0], uv[1],
                                QUERY_COLORS[det["query_id"] % len(QUERY_COLORS)], scale)
            if stream is None:
                stream = out.add_stream("libx264", rate=fps)
                stream.width, stream.height = img.shape[1], img.shape[0]
                stream.pix_fmt = "yuv420p"
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(packet)
        if stream is not None:
            for packet in stream.encode():
                out.mux(packet)
    return out_path


def load_cache(dataset_dir: Path, dirname: str, ep: int, frames: list[int]) -> dict[int, list[dict]]:
    path = dataset_dir / dirname
    if not path.exists():
        raise SystemExit(f"no anchor cache at {path} -- run build_molmo_cache.py first")
    out = {}
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False)
    with env.begin(buffers=True) as t:
        for f in frames:
            buf = t.get(f"{ep}-{f}".encode("ascii"))
            if buf is None:
                continue
            rec = np.frombuffer(bytes(buf), dtype=molmo_cache.RECORD_DTYPE)
            px = molmo_cache.decode_pixels(rec)
            if px:
                out[f] = px
    env.close()
    return out


def build_cloud_html(
    repo: Path, dataset_dir: Path, ep: int, anchor_ids: list[int], stride: int,
    out_dir: Path, prefix: str, cache_dirname: str, python: str,
) -> Path | None:
    """Shell out to viz_sampling_episode so the animation code stays single-sourced."""
    cmd = [
        python, "-m", "data_prep.roi_sampling.viz_sampling_episode",
        "--dataset-dir", str(dataset_dir), "--episode", str(ep),
        "--method", "molmo", "--stride", str(stride),
        "--out-dir", str(out_dir), "--out-prefix", prefix,
        "--molmo-anchor-dirname", cache_dirname,
        "--molmo-anchor-ids", *[str(i) for i in anchor_ids],
        "--dark", "--color-by", "weight",
    ]
    env = {**os.environ, "PYTHONPATH": f"{repo}:{os.environ.get('PYTHONPATH', '')}"}
    r = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  cloud animation failed: {r.stderr.strip()[-500:]}", file=sys.stderr)
        return None
    hits = sorted(out_dir.glob(f"{prefix}*.html"))
    return hits[-1] if hits else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--cache-dirname", default="points_3views_molmo")
    ap.add_argument("--tasks", nargs="*", default=list(DEFAULT_TASKS))
    ap.add_argument("--episode", type=int, default=None,
                    help="Override the per-task default episode.")
    ap.add_argument("--stride", type=int, default=8, help="Must match the cache's stride.")
    ap.add_argument("--scale", type=int, default=3, help="Nearest-neighbour upscale for the video.")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="pointact-robocasa365")
    ap.add_argument("--wandb-entity", default="diffusion4robots")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    root = args.dataset_root.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name="stage3.0-visu-molmopoint", job_type="viz",
            # Same stage tag as the trained arms in experiments/13_robocasa365/runs/*molmo*,
            # so one W&B filter returns the gate and the runs it gated. "gate" and
            # job_type="viz" are what still tell them apart.
            tags=["Stage 3: MolmoPoint anchor", "molmo", "gate"],
            config={"stride": args.stride, "cache_dirname": args.cache_dirname,
                    "tasks": args.tasks},
        )

    summary = {}
    for task in args.tasks:
        spec = DEFAULT_TASKS.get(task, {"episode": 0, "arms": {"molmo": [0]}})
        ep = args.episode if args.episode is not None else spec["episode"]
        dataset_dir = root / task
        if not dataset_dir.exists():
            print(f"skip {task}: no {dataset_dir}", file=sys.stderr)
            continue

        lengths, tasks_by_ep = read_meta(dataset_dir / "meta")
        info = json.loads((dataset_dir / "meta" / "info.json").read_text())
        chunks_size = int(info.get("chunks_size", 1000))
        key_frames = list(range(0, lengths[ep], args.stride))
        pixels = load_cache(dataset_dir, args.cache_dirname, ep, key_frames)
        hit = len(pixels) / max(1, len(key_frames))
        instruction = tasks_by_ep.get(ep, "")
        print(f"{task} ep{ep}: {len(pixels)}/{len(key_frames)} replan frames pointed "
              f"({hit:.0%})  instruction={instruction!r}")
        summary[task] = {"episode": ep, "instruction": instruction, "point_rate": round(hit, 3)}

        media = {}
        for view in VIEWS:
            p = render_view_video(dataset_dir, ep, view, key_frames, pixels,
                                  args.out_dir / f"{task}_ep{ep}_{view}.mp4",
                                  chunks_size, args.scale, args.fps)
            if p is not None:
                media[f"{task}/{view}_camera"] = p

        html = {}
        for arm, ids in spec["arms"].items():
            p = build_cloud_html(repo, dataset_dir, ep, ids, args.stride, args.out_dir,
                                 f"{task}_ep{ep}_{arm}_", args.cache_dirname, args.python)
            if p is not None:
                html[f"{task}/{arm}_cloud"] = p

        if run is not None:
            import wandb
            log = {k: wandb.Video(str(v), format="mp4") for k, v in media.items()}
            log.update({k: wandb.Html(v.read_text(), inject=False) for k, v in html.items()})
            log[f"{task}/point_rate"] = hit
            run.log(log)
        print(f"  videos: {len(media)}  animations: {len(html)}")

    (args.out_dir / "gate_summary.json").write_text(json.dumps(summary, indent=2))
    if run is not None:
        run.summary.update({f"{k}/point_rate": v["point_rate"] for k, v in summary.items()})
        run.finish()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
