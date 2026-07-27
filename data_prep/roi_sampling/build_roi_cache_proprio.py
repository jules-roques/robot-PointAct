"""Build the ROI halo cache from proprioception — no detector, no GPU.

While the gripper is closed it is holding the drawer handle, so the end-effector position
(``observation.state[0:3]``, already in the robot-base frame = the point-cloud frame) *is*
the handle position. That gives an exact per-frame handle location for free:

    f <  grasp_start : handle = eef at grasp_start   (drawer shut, handle where it will be grasped)
    grasp_start <= f <= grasp_end : handle = eef at f (gripper holds the handle as it opens)
    f >  grasp_end   : handle = eef at grasp_end     (released; handle stays where it was left)

Why this replaces the YOLO-World pass for training:
  * coverage 100% (vs 43.3%) — every frame gets an ROI,
  * always the drawer the demonstration actually opens — no wrong-drawer failures,
  * zero drift: the measured handle travel while opening is ~0.32 m, which exceeds the
    halo radius, so a detector-anchored *static* halo slides off the handle late in every
    episode. Tracking removes that entirely.
  * uses only data already in the dataset; no re-acquisition, no extra model.

Limitation: proprioception is a *demonstration* signal, so this is a training-time ROI.
The eval path still needs its own localizer (see the design doc).

Output format is identical to build_roi_cache.py — anchor(3), radius(1), n_in_box(1) as
float32 under the `ep-frame` key — so the dataloader and viz are unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lmdb
import msgpack
import msgpack_numpy
import numpy as np
import pyarrow.parquet as pq
from tqdm.auto import tqdm

msgpack_numpy.patch()

ROI_RECORD_DTYPE = np.float32


def handle_trajectory(states: np.ndarray, actions: np.ndarray) -> np.ndarray | None:
    """Per-frame handle position (N, 3) in the robot-base frame, or None.

    Uses the first contiguous run of closed-gripper frames: before it the handle is where
    it will be grasped, during it the eef tracks the handle, after it the handle stays
    where the gripper let go.
    """
    if states.ndim != 2 or states.shape[1] < 3 or actions.ndim != 2 or actions.shape[1] < 8:
        return None
    closed = actions[:, 7] > 0.5
    idx = np.flatnonzero(closed)
    if len(idx) == 0:
        return None
    g0 = int(idx[0])
    # End of the first contiguous closed run (tolerating 1-frame dropouts).
    g1 = g0
    for k in range(g0 + 1, len(closed)):
        if closed[k]:
            g1 = k
        elif k + 1 < len(closed) and closed[k + 1]:
            continue  # single-frame gap, keep going
        else:
            break

    eef = states[:, 0:3].astype(np.float64)
    handle = np.empty_like(eef)
    handle[:g0] = eef[g0]
    handle[g0 : g1 + 1] = eef[g0 : g1 + 1]
    handle[g1 + 1 :] = eef[g1]
    return handle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--out-dirname", default="points_3views_roi")
    ap.add_argument("--points-dirname", default="points_3views")
    ap.add_argument("--episodes", nargs="*", type=int, default=None)
    # Handle-focused radius. Also scalable at runtime via the dataloader's
    # roi_radius_scale, so this needs no rebuild to retune.
    ap.add_argument("--radius", type=float, default=0.15)
    ap.add_argument("--stats-stride", type=int, default=50,
                    help="Sample every Nth frame to report the realised ROI fraction.")
    ap.add_argument("--map-size-gb", type=float, default=8.0)
    args = ap.parse_args()

    d = args.dataset_dir.expanduser().resolve()
    info = json.loads((d / "meta" / "info.json").read_text())
    chunks = int(info.get("chunks_size", 1000))

    lengths = {}
    with open(d / "meta" / "episodes.jsonl") as f:
        for line in f:
            r = json.loads(line)
            lengths[int(r["episode_index"])] = int(r["length"])

    episodes = args.episodes if args.episodes else sorted(lengths)
    print(f"building proprioceptive ROI cache for {len(episodes)} episodes (radius={args.radius} m)")

    points_env = lmdb.open(str(d / args.points_dirname), readonly=True, lock=False, readahead=False)
    out_dir = d / args.out_dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    out_env = lmdb.open(str(out_dir), map_size=int(args.map_size_gb * (1024 ** 3)))

    n_frames = n_written = n_noanchor = 0
    roi_fracs = []

    with points_env.begin(buffers=True) as ptxn:
        for ep in tqdm(episodes, desc="episodes", unit="ep"):
            p = d / "data" / f"chunk-{ep // chunks:03d}" / f"episode_{ep:06d}.parquet"
            try:
                tb = pq.read_table(p, columns=["observation.state", "action"])
            except Exception:
                n_noanchor += lengths.get(ep, 0)
                continue
            S = np.array([np.asarray(x, np.float32) for x in tb.column("observation.state").to_pylist()])
            A = np.array([np.asarray(x, np.float32) for x in tb.column("action").to_pylist()])
            handle = handle_trajectory(S, A)
            if handle is None:
                n_noanchor += lengths.get(ep, 0)
                continue

            n = min(lengths[ep], len(handle))
            with out_env.begin(write=True) as wtxn:
                for f in range(n):
                    key = f"{ep}-{f}".encode("ascii")
                    rec = np.array([*handle[f].tolist(), args.radius, 0.0], dtype=ROI_RECORD_DTYPE)
                    wtxn.put(key, rec.tobytes())
                    n_frames += 1
                    n_written += 1

                    if f % args.stats_stride == 0:
                        buf = ptxn.get(key)
                        if buf is not None:
                            cloud = msgpack.unpackb(bytes(buf)).astype(np.float32)
                            dist = np.linalg.norm(cloud[:, :3].astype(np.float64) - handle[f], axis=1)
                            roi_fracs.append(float((dist <= args.radius).mean()))

    points_env.close()
    out_env.sync()
    out_env.close()

    frac = float(np.mean(roi_fracs)) if roi_fracs else 0.0
    total = n_written + n_noanchor
    print(f"done: frames={n_written} coverage={n_written / max(1, total):.1%} "
          f"(no-anchor frames={n_noanchor}) mean_roi_fraction={frac:.1%} "
          f"[sampled {len(roi_fracs)} frames]")
    meta = out_dir.parent / "roi_meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "build_summary_proprio.json").write_text(json.dumps({
        "source": "proprioceptive handle tracking (no detector)",
        "frames": n_written, "no_anchor_frames": n_noanchor,
        "coverage": n_written / max(1, total),
        "radius": args.radius, "mean_roi_fraction": frac,
    }, indent=2))


if __name__ == "__main__":
    main()
