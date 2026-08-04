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
import msgpack
import msgpack_numpy
import numpy as np
from tqdm.auto import tqdm

from pointact.roi_sampling import molmo_cache
from pointact.roi_sampling.geometry import candidate_anchors

msgpack_numpy.patch()

VIEWS = ("left", "right")


def opendrawer_queries(instruction: str) -> list[str]:
    """"Open the left drawer." -> point at that drawer's handle.

    The side matters and is the whole reason a pointing model is used here: an
    open-vocabulary box detector finds every drawer and cannot tell which one the
    instruction means.
    """
    m = re.search(r"\b(left|right)\b", instruction, re.I)
    side = f"{m.group(1).lower()} " if m else ""
    return [f"Point to the handle of the {side}drawer."]


def ppcs_queries(instruction: str) -> list[str]:
    """"Pick the apple from the plate and place it in the pan." -> the apple, then the pan.

    Query 0 is the manipulated object and query 1 the destination; `molmo_anchor_ids`
    selects which become Gaussian centres, so both arms come from this one cache.
    """
    m = re.search(r"pick the (.+?) from the", instruction, re.I)
    obj = m.group(1).strip() if m else "object"
    return [f"Point to the {obj}.", "Point to the pan."]


def tom_queries(instruction: str) -> list[str]:
    return ["Point to the start button on the microwave."]


#: task -> instruction -> pointing queries. Derived from each episode's own instruction
#: rather than hard-coded per task, because PickPlaceCounterToStove varies the object.
TASK_QUERIES = {
    "OpenDrawer": opendrawer_queries,
    "PickPlaceCounterToStove": ppcs_queries,
    "TurnOnMicrowave": tom_queries,
}


def load_calib(path: Path) -> tuple[list[dict], tuple[int, int]]:
    d = np.load(path, allow_pickle=True)
    cams = [{"name": v, "intrinsic": d[f"{v}_K"], "base2cam": d[f"{v}_base2cam"]} for v in VIEWS]
    return cams, tuple(int(x) for x in d["image_hw"])


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


def fuse(per_view: dict[str, tuple[np.ndarray, int]], agree_dist: float):
    """Combine the two views' anchors for one query.

    Agreeing views are averaged; disagreeing ones fall back to the better-supported view,
    since a disagreement means at least one of them lifted through an occluder and the
    midpoint of a right answer and a wrong one is simply a third wrong answer.
    """
    got = [(v, a, n) for v, (a, n) in per_view.items()]
    if not got:
        return None, False
    if len(got) == 1:
        return got[0][1], False
    (_, a0, n0), (_, a1, n1) = got
    if float(np.linalg.norm(a0 - a1)) <= agree_dist:
        return (a0 + a1) / 2.0, True
    return (a0 if n0 >= n1 else a1), False


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
    ap.add_argument("--point-window", type=int, default=6,
                    help="Half-width in pixels of the box a returned point is padded into.")
    ap.add_argument("--min-in-window", type=int, default=8,
                    help="Fewer cloud points than this in the window -> no anchor.")
    ap.add_argument("--agree-dist", type=float, default=0.10,
                    help="Metres; views closer than this are averaged.")
    ap.add_argument("--batch", type=int, default=4,
                    help="(frame, query) requests per forward.")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--map-size-gb", type=float, default=4.0)
    ap.add_argument("--verify-point-order", action="store_true",
                    help="Probe the checkpoint's point-tuple field order before building.")
    args = ap.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    task = dataset_dir.name
    if task not in TASK_QUERIES:
        raise SystemExit(f"no pointing queries defined for task {task!r}; "
                         f"known: {sorted(TASK_QUERIES)}")
    queries_for = TASK_QUERIES[task]

    calib_path = args.calib or dataset_dir / "roi_meta" / "camera_calib.npz"
    if not calib_path.exists():
        raise SystemExit(f"no calibration at {calib_path} -- run dump_camera_calib.py first")
    cams, image_hw = load_calib(calib_path)
    cam_by_name = {c["name"]: c for c in cams}

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
          f"queries={queries_for(tasks[episodes[0]])}")

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
             "queries": 0, "query_hits": 0, "view_hits": {"left": 0, "right": 0},
             "agree": 0, "both_views": 0, "forwards": 0}
    t0 = time.time()

    def video_path(view: str, ep: int) -> Path:
        return (dataset_dir / "videos" / f"chunk-{ep // chunks_size:03d}"
                / f"observation.images.{view}_image" / f"episode_{ep:06d}.mp4")

    with points_env.begin(buffers=True) as ptxn:
        for ep in tqdm(episodes, desc="episodes", unit="ep"):
            n = lengths[ep]
            frames = {v: decode_video(video_path(v, ep)) for v in VIEWS}
            n_use = min(n, *(len(frames[v]) for v in VIEWS))
            queries = queries_for(tasks.get(ep, ""))
            key_frames = list(range(0, n_use, args.stride))

            # One request per (key frame, query); both views ride in the same request, so a
            # frame costs len(queries) forwards rather than 2 x len(queries).
            reqs = [(f, qi) for f in key_frames for qi in range(len(queries))]
            dets: dict[tuple[int, int], list] = {}
            for s in range(0, len(reqs), args.batch):
                chunk = reqs[s:s + args.batch]
                image_sets = [[frames[v][f] for v in VIEWS] for f, _ in chunk]
                prompts = [queries[qi] for _, qi in chunk]
                out = pointer.point(image_sets, prompts)
                stats["forwards"] += 1
                for (f, qi), res in zip(chunk, out):
                    dets[(f, qi)] = res

            # Lift, fuse, and hold each key frame's anchors across its replan window.
            with out_env.begin(write=True) as wtxn:
                for ki, f in enumerate(key_frames):
                    buf = ptxn.get(f"{ep}-{f}".encode("ascii"))
                    cloud_xyz = (msgpack.unpackb(bytes(buf))[:, :3].astype(np.float64)
                                 if buf is not None else None)
                    anchors = []
                    for qi in range(len(queries)):
                        per_view, uvs = {}, {}
                        for d in dets.get((f, qi), []):
                            view = VIEWS[d.image_num] if d.image_num < len(VIEWS) else None
                            if view is None or view in per_view:
                                continue  # first point per view; extras are other instances
                            uvs[view] = (d.x, d.y)
                            stats["view_hits"][view] += 1
                            if cloud_xyz is None:
                                continue
                            got = anchor_from_pixel(cloud_xyz, cam_by_name[view], (d.x, d.y),
                                                    image_hw, args.point_window,
                                                    args.min_in_window)
                            if got is not None:
                                per_view[view] = got
                        stats["queries"] += 1
                        if len(per_view) == 2:
                            stats["both_views"] += 1
                        xyz, agreed = fuse(per_view, args.agree_dist)
                        if xyz is None:
                            continue
                        stats["query_hits"] += 1
                        stats["agree"] += int(agreed)
                        anchors.append({
                            "xyz": xyz, "query_id": qi,
                            "n_support": sum(n for _, n in per_view.values()),
                            "left_uv": uvs.get("left"), "right_uv": uvs.get("right"),
                            "agree": agreed,
                        })

                    rec = (molmo_cache.encode_record(anchors) if anchors
                           else molmo_cache.empty_record()).tobytes()
                    end = key_frames[ki + 1] if ki + 1 < len(key_frames) else n_use
                    for ff in range(f, end):
                        wtxn.put(f"{ep}-{ff}".encode("ascii"), rec)
                        stats["frames"] += 1
                        stats["pointed_frames"] += int(bool(anchors))

    points_env.close()
    out_env.sync()
    out_env.close()

    stats["seconds"] = round(time.time() - t0, 1)
    stats["s_per_forward"] = round(stats["seconds"] / max(1, stats["forwards"]), 3)
    stats["query_hit_rate"] = round(stats["query_hits"] / max(1, stats["queries"]), 4)
    stats["agree_rate"] = round(stats["agree"] / max(1, stats["both_views"]), 4)
    stats["frame_cover"] = round(stats["pointed_frames"] / max(1, stats["frames"]), 4)
    print(json.dumps(stats, indent=2, default=str))

    meta_out = dataset_dir / "roi_meta"
    meta_out.mkdir(parents=True, exist_ok=True)
    tag = "" if not args.episode_range else f"_{args.episode_range[0]}-{args.episode_range[1]}"
    with open(meta_out / f"molmo_build_summary{tag}.json", "w") as fh:
        json.dump({**stats, "args": {k: str(v) for k, v in vars(args).items()}}, fh,
                  indent=2, default=str)


if __name__ == "__main__":
    main()
