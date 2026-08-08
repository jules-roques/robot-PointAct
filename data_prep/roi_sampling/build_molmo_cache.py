"""Build the MolmoPoint anchor cache: where should the point budget go, per frame.

Runs in ``envs/molmo`` (transformers 4.57.1) on an A100/H100. For each strided frame:

  1. decode the left and right agentview frames from the episode mp4s,
  2. ask MolmoPoint to point at the task's target(s), given the episode's own instruction,
  3. pad each returned pixel into a small window,
  4. project the stored base-frame cloud into that camera and take the median of the points
     landing in the window -- the 3D anchor,
  5. fuse the two views per query,
  6. write the anchors under the key the dataloader already uses.

Lifting through the cloud rather than reading the depth map at the pixel is deliberate:
invalid-depth pixels (which both OpenDrawer and PickPlaceCounterToStove log) then reduce
support instead of producing a confident anchor at the wrong depth.

Pointing runs at the policy's replan cadence, not every frame, because that is what eval
can afford -- eval is causal and must point on the current frame. The anchor is held across
the intervening frames and written to every one of them, so the dataloader needs no stride
logic and training sees exactly the signal eval will produce.

Usage:
    uv run --project envs/molmo --no-sync python -m data_prep.roi_sampling.build_molmo_cache \
        --dataset-dir robot_data/robocasa365/lerobot_point_lmdb/OpenDrawer \
        --model-dir $SCRATCH/models/MolmoPoint-8B
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import av
import lmdb
import pyarrow.parquet as pq
import msgpack
import msgpack_numpy
import numpy as np
from tqdm.auto import tqdm

from pointact.roi_sampling import molmo_cache
from pointact.roi_sampling.geometry import candidate_anchors
# Prompts, view list and the fusion rules are shared with the live evaluator rather than
# duplicated: pointact/roi_sampling/molmo_anchors.py explains why that has to be one source.
from pointact.roi_sampling.molmo_anchors import (
    STATIC_VIEWS,
    VIEWS,
    apply_wrist,
    fuse,
    queries_for,
)

msgpack_numpy.patch()


def load_calib(path: Path):
    """(static cams, image_hw, wrist_K, wrist_eef2cam). wrist_* are None on old calibs."""
    d = np.load(path, allow_pickle=True)
    cams = [{"name": v, "intrinsic": d[f"{v}_K"], "base2cam": d[f"{v}_base2cam"]}
            for v in STATIC_VIEWS]
    wk = d["wrist_K"] if "wrist_K" in d.files else None
    we = d["wrist_eef2cam"] if "wrist_eef2cam" in d.files else None
    return cams, tuple(int(x) for x in d["image_hw"]), wk, we


def read_episode_states(dataset_dir: Path, ep: int, chunks_size: int) -> np.ndarray:
    """(T, 16) proprio; [0:3] eef position and [3:7] eef xyzw quaternion, in the base frame."""
    f = (dataset_dir / "data" / f"chunk-{ep // chunks_size:03d}"
         / f"episode_{ep:06d}.parquet")
    return np.stack(pq.read_table(str(f)).column("observation.state").to_pylist()).astype(np.float64)


def wrist_cam_at(states: np.ndarray, t: int, wrist_K, eef2cam) -> dict | None:
    """The wrist camera's base->cam at frame ``t``, from proprio alone.

    ``base2cam(t) = eef2cam @ inv(T_base_eef(t))``. Validated end to end by projecting the
    MuJoCo handle centroid into the wrist video: out of frame while the arm is home, then
    on the handle with depth falling 0.56 -> 0.17 m as the gripper closes on it.
    """
    from scipy.spatial.transform import Rotation as R

    if wrist_K is None or eef2cam is None or t >= len(states):
        return None
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat(states[t, 3:7]).as_matrix()
    T[:3, 3] = states[t, :3]
    return {"name": "wrist", "intrinsic": wrist_K, "base2cam": eef2cam @ np.linalg.inv(T)}


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


def decode_video(path: Path) -> list[np.ndarray]:
    """Decode an mp4 to RGB uint8 HxWx3 frames in stored-frame orientation.

    No vertical flip: the stored frames and the depth the clouds were built from share row
    indexing, so a pixel measured here lines up with the reprojected v directly (verified
    by projecting the ground-truth handle centroid back onto these frames).
    """
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def anchor_from_pixel(cloud_xyz, cam, uv, image_hw, window: int, min_in_window: int):
    """Lift one pixel to a base-frame anchor: pad to a window, take the in-window median."""
    x, y = float(uv[0]), float(uv[1])
    box = np.array([[x - window, y - window, x + window, y + window]])
    cands = candidate_anchors(cloud_xyz, [cam], {cam["name"]: box}, image_hw,
                              min_in_box=min_in_window)
    if not cands:
        return None
    anchor, mask = cands[0]
    return anchor, int(mask.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--calib", type=Path, default=None,
                    help="Defaults to <dataset-dir>/roi_meta/camera_calib.npz")
    ap.add_argument("--out-dirname", default="points_3views_molmo")
    ap.add_argument("--points-dirname", default="points_3views")
    ap.add_argument("--episodes", nargs="*", type=int, default=None)
    ap.add_argument("--episode-range", nargs=2, type=int, default=None,
                    help="[start, end) -- for splitting the build across an array job.")
    # Must match the policy's replan_steps: the cache has to carry the same signal eval
    # produces, and eval re-points once per replan.
    ap.add_argument("--stride", type=int, default=8)
    # 2 (a 5x5 box) beats wider windows on these 256x256 renders: measured against the
    # OpenDrawer GT handle over 272 strided frames, the median anchor error is 3.2 cm at
    # window 2 against 4.4 cm at window 6. A handle is only a few pixels across, so a wide
    # window pulls in the cabinet face behind it and biases the median off the target.
    ap.add_argument("--point-window", type=int, default=2,
                    help="Half-width in pixels of the box a returned point is padded into.")
    # 1, i.e. effectively off. This guard came from the YOLO builder, where "few points in
    # the box" meant an unreliable *detection*. For a single pointed pixel it means only that
    # the cloud is sparse there, which is not evidence against the point: on TurnOnMicrowave
    # median support at window 2 is 7, and min=5 sent 35% of frames to a uniform fallback.
    # n_support is recorded per anchor, so low-confidence lifts stay identifiable downstream.
    ap.add_argument("--min-in-window", type=int, default=1,
                    help="Fewer cloud points than this in the window -> no anchor.")
    ap.add_argument("--agree-dist", type=float, default=0.10,
                    help="Metres; agentviews closer than this are averaged.")
    ap.add_argument("--wrist-accept-dist", type=float, default=0.15,
                    help="Metres; the wrist anchor is adopted only within this of the "
                         "agentview one. Guards the episode start, where the target is out "
                         "of the wrist's frame but the model answers anyway.")
    ap.add_argument("--no-wrist", action="store_true",
                    help="Point on the agentviews only (old two-view behaviour).")
    ap.add_argument("--allow-lone-wrist", action="store_true",
                    help="Accept a wrist anchor with no agentview to corroborate it.")
    # 1 is not a placeholder: the checkpoint's remote code raises on any batch > 1 (see
    # MolmoPointer.point). Kept as a flag so a fixed upstream release can be exploited
    # without touching the builder.
    ap.add_argument("--batch", type=int, default=1,
                    help="(frame, query) requests per forward; >1 fails on this checkpoint.")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--map-size-gb", type=float, default=4.0)
    ap.add_argument("--verify-point-order", action="store_true",
                    help="Probe the checkpoint's point-tuple field order before building.")
    args = ap.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    task = dataset_dir.name

    calib_path = args.calib or dataset_dir / "roi_meta" / "camera_calib.npz"
    if not calib_path.exists():
        raise SystemExit(f"no calibration at {calib_path} -- run dump_camera_calib.py first")
    cams, image_hw, wrist_K, wrist_eef2cam = load_calib(calib_path)
    cam_by_name = {c["name"]: c for c in cams}
    use_wrist = (not args.no_wrist) and wrist_K is not None
    if not args.no_wrist and wrist_K is None:
        raise SystemExit(f"{calib_path} has no wrist calibration -- re-run "
                         f"dump_camera_calib.py, or pass --no-wrist")
    views = VIEWS if use_wrist else STATIC_VIEWS

    lengths, tasks = read_meta(dataset_dir / "meta")
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    chunks_size = int(info.get("chunks_size", 1000))

    episodes = sorted(lengths)
    if args.episode_range:
        lo, hi = args.episode_range
        episodes = [e for e in episodes if lo <= e < hi]
    if args.episodes:
        episodes = [e for e in args.episodes if e in lengths]

    print(f"task={task} episodes={len(episodes)} stride={args.stride} "
          f"queries={queries_for(task, tasks[episodes[0]])}")

    from pointact.roi_sampling.molmo_pointer import MolmoPointer
    pointer = MolmoPointer(str(args.model_dir), max_new_tokens=args.max_new_tokens)
    if args.verify_point_order:
        print(f"point field order: {pointer.verify_point_order()}")

    points_env = lmdb.open(str(dataset_dir / args.points_dirname), readonly=True,
                           lock=False, readahead=False)
    out_dir = dataset_dir / args.out_dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    out_env = lmdb.open(str(out_dir), map_size=int(args.map_size_gb * (1024 ** 3)))

    stats = {"task": task, "episodes": len(episodes), "frames": 0, "pointed_frames": 0,
             "queries": 0, "query_hits": 0, "view_hits": {v: 0 for v in views},
             "agree": 0, "both_views": 0, "forwards": 0,
             "wrist_lifted": 0, "wrist_adopted": 0, "wrist_rejected": 0}
    t0 = time.time()

    def video_path(view: str, ep: int) -> Path:
        return (dataset_dir / "videos" / f"chunk-{ep // chunks_size:03d}"
                / f"observation.images.{view}_image" / f"episode_{ep:06d}.mp4")

    with points_env.begin(buffers=True) as ptxn:
        for ep in tqdm(episodes, desc="episodes", unit="ep"):
            n = lengths[ep]
            frames = {v: decode_video(video_path(v, ep)) for v in views}
            n_use = min(n, *(len(frames[v]) for v in views))
            states = read_episode_states(dataset_dir, ep, chunks_size) if use_wrist else None
            queries = queries_for(task, tasks.get(ep, ""))
            key_frames = list(range(0, n_use, args.stride))

            # One request per (key frame, query); both views ride in the same request, so a
            # frame costs len(queries) forwards rather than 2 x len(queries).
            reqs = [(f, qi) for f in key_frames for qi in range(len(queries))]
            dets: dict[tuple[int, int], list] = {}
            for s in range(0, len(reqs), args.batch):
                chunk = reqs[s:s + args.batch]
                image_sets = [[frames[v][f] for v in views] for f, _ in chunk]
                prompts = [queries[qi] for _, qi in chunk]
                out = pointer.point(image_sets, prompts)
                stats["forwards"] += len(chunk)
                for (f, qi), res in zip(chunk, out):
                    dets[(f, qi)] = res

            # Lift, fuse, and hold each key frame's anchors across its replan window.
            with out_env.begin(write=True) as wtxn:
                for ki, f in enumerate(key_frames):
                    buf = ptxn.get(f"{ep}-{f}".encode("ascii"))
                    cloud_xyz = (msgpack.unpackb(bytes(buf))[:, :3].astype(np.float64)
                                 if buf is not None else None)
                    # The wrist camera's pose is frame-specific; the agentviews' are not.
                    wcam = wrist_cam_at(states, f, wrist_K, wrist_eef2cam) if use_wrist else None
                    cams_now = dict(cam_by_name, **({"wrist": wcam} if wcam else {}))

                    anchors = []
                    for qi in range(len(queries)):
                        per_view, uvs = {}, {}
                        for d in dets.get((f, qi), []):
                            view = views[d.image_num] if d.image_num < len(views) else None
                            if view is None or view in uvs:
                                continue  # first point per view; extras are other instances
                            uvs[view] = (d.x, d.y)
                            stats["view_hits"][view] += 1
                            cam = cams_now.get(view)
                            if cloud_xyz is None or cam is None:
                                continue
                            got = anchor_from_pixel(cloud_xyz, cam, (d.x, d.y),
                                                    image_hw, args.point_window,
                                                    args.min_in_window)
                            if got is not None:
                                per_view[view] = got
                        stats["queries"] += 1
                        if sum(v in per_view for v in STATIC_VIEWS) == 2:
                            stats["both_views"] += 1
                        xyz, agreed = fuse(per_view, args.agree_dist)
                        if "wrist" in per_view:
                            stats["wrist_lifted"] += 1
                        xyz, used_wrist = apply_wrist(
                            xyz, per_view, args.wrist_accept_dist,
                            require_agentview=not args.allow_lone_wrist)
                        if "wrist" in per_view:
                            stats["wrist_adopted" if used_wrist else "wrist_rejected"] += 1
                        # Record the slot even when nothing lifted, with a NaN anchor. The
                        # pixels are the expensive half (0.7 s of an 8B model); the lift is
                        # 1.7 ms of geometry. Dropping the pixels on failure meant the frames
                        # we most need to diagnose were the only ones we could not re-lift
                        # offline -- and it silently biased every window sweep towards frames
                        # that had already succeeded. decode_anchors skips non-finite xyz, so
                        # the dataloader still falls back to uniform for these.
                        if xyz is not None:
                            stats["query_hits"] += 1
                            stats["agree"] += int(agreed)
                        if xyz is None and not uvs:
                            continue  # the model pointed nowhere; nothing to record
                        anchors.append({
                            "xyz": xyz if xyz is not None else np.full(3, np.nan),
                            "query_id": qi,
                            "n_support": sum(n for _, n in per_view.values()),
                            "agree": agreed,
                            **{f"{v}_uv": uvs.get(v) for v in VIEWS},
                        })

                    rec = (molmo_cache.encode_record(anchors) if anchors
                           else molmo_cache.empty_record()).tobytes()
                    has_anchor = any(np.isfinite(a["xyz"]).all() for a in anchors)
                    end = key_frames[ki + 1] if ki + 1 < len(key_frames) else n_use
                    for ff in range(f, end):
                        wtxn.put(f"{ep}-{ff}".encode("ascii"), rec)
                        stats["frames"] += 1
                        stats["pointed_frames"] += int(has_anchor)

    points_env.close()
    out_env.sync()
    out_env.close()

    stats["seconds"] = round(time.time() - t0, 1)
    stats["s_per_forward"] = round(stats["seconds"] / max(1, stats["forwards"]), 3)
    stats["query_hit_rate"] = round(stats["query_hits"] / max(1, stats["queries"]), 4)
    stats["agree_rate"] = round(stats["agree"] / max(1, stats["both_views"]), 4)
    stats["frame_cover"] = round(stats["pointed_frames"] / max(1, stats["frames"]), 4)
    stats["wrist_adopt_rate"] = round(
        stats["wrist_adopted"] / max(1, stats["wrist_lifted"]), 4)
    print(json.dumps(stats, indent=2, default=str))

    meta_out = dataset_dir / "roi_meta"
    meta_out.mkdir(parents=True, exist_ok=True)
    tag = "" if not args.episode_range else f"_{args.episode_range[0]}-{args.episode_range[1]}"
    with open(meta_out / f"molmo_build_summary{tag}.json", "w") as fh:
        json.dump({**stats, "args": {k: str(v) for k, v in vars(args).items()}}, fh,
                  indent=2, default=str)


if __name__ == "__main__":
    main()
