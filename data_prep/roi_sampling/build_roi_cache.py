"""Stage B: build the ROI halo cache with YOLO-World.

Runs in the ROI preproc env (torch + ultralytics, H100). For each stored frame:
  1. load the base-frame point cloud from the points LMDB (same key/order as training),
  2. decode the left/right RGB frame from the episode mp4,
  3. detect drawers with a YOLO-World checkpoint pre-baked for "drawer",
  4. build a 3D anchor per box and keep the drawer the instruction names (3D side test),
  5. reproject the cloud, take the in-box epicenter, size the halo,
  6. write (anchor_xyz, radius, n_in_box) under the identical `ep-frame` key.

The cache stores the **halo**, not per-point flags: the dataloader derives the mask (or
soft weights) from the points it already has, so radius scaling and hard/soft selection
are runtime knobs that need no rebuild. Frames with no reliable detection store a zero
radius, and the dataloader falls back to uniform sampling for them.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import av
import lmdb
import msgpack
import msgpack_numpy
import numpy as np
from tqdm.auto import tqdm

from pointact.roi_sampling.geometry import (
    candidate_anchors,
    halo_from_candidate,
    parse_task_side,
    select_anchor_by_side,
)

msgpack_numpy.patch()

# Cache record: anchor(3) + radius(1) + n_in_box(1), float32. Storing the halo itself
# (not baked per-point flags) keeps radius scaling and hard/soft selection as dataloader
# knobs — they can change without rebuilding this cache.
ROI_RECORD_DTYPE = np.float32


def load_calib(path: Path) -> tuple[list[dict], tuple[int, int]]:
    d = np.load(path, allow_pickle=True)
    cams = [
        {"name": "left", "intrinsic": d["left_K"], "base2cam": d["left_base2cam"]},
        {"name": "right", "intrinsic": d["right_K"], "base2cam": d["right_base2cam"]},
    ]
    hw = tuple(int(x) for x in d["image_hw"])
    return cams, hw


def episode_lengths(meta_dir: Path) -> dict[int, int]:
    """Read {episode_index: length} from meta/episodes.jsonl."""
    lengths = {}
    with open(meta_dir / "episodes.jsonl") as f:
        for line in f:
            row = json.loads(line)
            lengths[int(row["episode_index"])] = int(row["length"])
    return lengths


def episode_sides(meta_dir: Path) -> dict[int, str | None]:
    """Map {episode_index: "left"|"right"|None} from each episode's instruction.

    OpenDrawer instructions are "Open the left drawer." / "Open the right drawer.", so the
    target side is known for every frame — at training and at eval alike.
    """
    sides = {}
    with open(meta_dir / "episodes.jsonl") as f:
        for line in f:
            row = json.loads(line)
            tasks = row.get("tasks") or []
            sides[int(row["episode_index"])] = parse_task_side(tasks[0] if tasks else None)
    return sides


def decode_video(path: Path) -> list[np.ndarray]:
    """Decode an mp4 to a list of RGB uint8 HxWx3 frames (stored-frame orientation)."""
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def detect_boxes(model, images_rgb: list[np.ndarray], conf: float, max_det: int) -> list[np.ndarray]:
    """Run YOLO-World on a batch of RGB frames; return (K,4) xyxy boxes per frame.

    Frames are passed as RGB: on these sim renders RGB gives markedly higher
    open-vocab confidence than the usual BGR convention (verified on OpenDrawer).
    """
    results = model.predict(images_rgb, conf=conf, max_det=max_det, imgsz=256, verbose=False)
    out = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            out.append(np.zeros((0, 4), dtype=np.float32))
        else:
            out.append(r.boxes.xyxy.cpu().numpy().astype(np.float32))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, type=Path,
                    help="OpenDrawer dir with points_3views/, videos/, meta/.")
    ap.add_argument("--calib", required=True, type=Path)
    ap.add_argument("--weights", required=True, type=Path,
                    help="YOLO-World checkpoint pre-baked with the class prompt.")
    ap.add_argument("--out-dirname", default="points_3views_roi")
    ap.add_argument("--points-dirname", default="points_3views")
    ap.add_argument("--episodes", nargs="*", type=int, default=None)
    # "drawer" fires reliably at ~0.1-0.16 conf on these sim renders. Keep several
    # candidates so the instructed side ("left"/"right") can pick the correct drawer.
    ap.add_argument("--conf", type=float, default=0.02)
    ap.add_argument("--max-det", type=int, default=5)
    ap.add_argument("--halo-scale", type=float, default=1.0)
    ap.add_argument("--min-radius", type=float, default=0.02)
    ap.add_argument("--max-radius", type=float, default=0.25)
    ap.add_argument("--min-in-box", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--map-size-gb", type=float, default=8.0)
    args = ap.parse_args()

    from ultralytics import YOLOWorld

    dataset_dir = args.dataset_dir.expanduser().resolve()
    cams, image_hw = load_calib(args.calib)
    lengths = episode_lengths(dataset_dir / "meta")
    sides = episode_sides(dataset_dir / "meta")
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    chunks_size = int(info.get("chunks_size", 1000))

    episodes = args.episodes if args.episodes else sorted(lengths)
    print(f"building ROI cache for {len(episodes)} episodes -> {args.out_dirname}")

    model = YOLOWorld(str(args.weights))

    points_env = lmdb.open(str(dataset_dir / args.points_dirname), readonly=True, lock=False, readahead=False)
    out_dir = dataset_dir / args.out_dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    out_env = lmdb.open(str(out_dir), map_size=int(args.map_size_gb * (1024 ** 3)))

    stats = {"frames": 0, "detected": 0, "roi_points": 0, "total_points": 0}

    def video_path(cam_key: str, ep: int) -> Path:
        chunk = ep // chunks_size
        return dataset_dir / "videos" / f"chunk-{chunk:03d}" / f"observation.images.{cam_key}_image" / f"episode_{ep:06d}.mp4"

    with points_env.begin(buffers=True) as ptxn:
        for ep in tqdm(episodes, desc="episodes", unit="ep"):
            n = lengths[ep]
            side = sides.get(ep)  # "left"/"right" from the episode instruction
            left = decode_video(video_path("left", ep))
            right = decode_video(video_path("right", ep))
            n_use = min(n, len(left), len(right))
            if n_use != n:
                print(f"  ep {ep}: length mismatch cloud={n} left={len(left)} right={len(right)} -> using {n_use}")

            # Detect over the whole episode in batches.
            left_boxes, right_boxes = [], []
            for s in range(0, n_use, args.batch):
                e = min(s + args.batch, n_use)
                left_boxes.extend(detect_boxes(model, left[s:e], args.conf, args.max_det))
                right_boxes.extend(detect_boxes(model, right[s:e], args.conf, args.max_det))

            with out_env.begin(write=True) as wtxn:
                for f in range(n_use):
                    key = f"{ep}-{f}".encode("ascii")
                    buf = ptxn.get(key)
                    if buf is None:
                        continue
                    cloud = msgpack.unpackb(bytes(buf)).astype(np.float32)
                    pts = cloud[:, :3]
                    # One 3D anchor per detected box, then pick the drawer the
                    # instruction names by comparing candidates in the base frame.
                    dets = {"left": left_boxes[f], "right": right_boxes[f]}
                    cands = candidate_anchors(
                        pts, cams, dets, image_hw, min_in_box=args.min_in_box
                    )
                    cand = select_anchor_by_side(cands, side)
                    res = halo_from_candidate(
                        pts, cand,
                        halo_scale=args.halo_scale,
                        min_radius=args.min_radius,
                        max_radius=args.max_radius,
                    )
                    if res.anchor is None:
                        record = np.array([0, 0, 0, 0, res.n_in_box], dtype=ROI_RECORD_DTYPE)
                    else:
                        record = np.array(
                            [*res.anchor.tolist(), res.radius, res.n_in_box],
                            dtype=ROI_RECORD_DTYPE,
                        )
                    wtxn.put(key, record.tobytes())

                    stats["frames"] += 1
                    stats["total_points"] += len(pts)
                    if res.anchor is not None:
                        stats["detected"] += 1
                        stats["roi_points"] += int(res.roi_mask.sum())

    points_env.close()
    out_env.sync()
    out_env.close()

    det_rate = stats["detected"] / max(1, stats["frames"])
    roi_frac = stats["roi_points"] / max(1, stats["total_points"])
    print(f"done: frames={stats['frames']} detected={stats['detected']} "
          f"({det_rate:.1%}) mean_roi_fraction={roi_frac:.1%}")
    # Persist a small summary next to the cache.
    (out_dir.parent / "roi_meta").mkdir(parents=True, exist_ok=True)
    with open(out_dir.parent / "roi_meta" / "build_summary.json", "w") as fh:
        json.dump({**stats, "detect_rate": det_rate, "roi_fraction": roi_frac,
                   "args": {k: str(v) for k, v in vars(args).items()}}, fh, indent=2)


if __name__ == "__main__":
    main()
