"""Compare how many points each voxel grid puts near the end effector.

Reads the throwaway caches written by voxel_probe_jeanzay.slurm alongside the 1 cm cache the
dataset is actually built on, and reports, per grid: cloud size, points within a few radii of
the gripper, points on the target handle, and -- the number that decides whether a finer grid
would help -- how many ROI points the eef-density draw manages to place at each point budget.

    python experiments/13_robocasa365/voxel_probe_report.py

The 1 cm column comes from the real LMDB cache (points_3views), the others from replay caches
(episode npz + points npy). Different readers, same measurement.
"""

import argparse
import json
import os
from pathlib import Path

import lmdb
import msgpack
import msgpack_numpy
import numpy as np
import pandas as pd

msgpack_numpy.patch()

from pointact.roi_sampling.geometry import eef_density_weights
from pointact.roi_sampling.sampling import density_weighted_indices

# Matches the eef arm's configured density (od-eef-*.yaml) so the "drawn" columns describe the
# sampler we actually train with, not a hypothetical one.
SIGMA, FLOOR = 0.08, 0.05
RADII = (0.04, 0.08, 0.16)
BUDGETS = (2048, 4096, 8192)
# From environments.py's label table; the handle is the anchor arm's target.
POINT_LABEL_TARGET_HANDLE = 4


def workspace_mask(xyz: np.ndarray, ws: dict) -> np.ndarray:
    return (
        (xyz[:, 0] > ws["X_BBOX"][0]) & (xyz[:, 0] < ws["X_BBOX"][1])
        & (xyz[:, 1] > ws["Y_BBOX"][0]) & (xyz[:, 1] < ws["Y_BBOX"][1])
        & (xyz[:, 2] > ws["Z_BBOX"][0]) & (xyz[:, 2] < ws["Z_BBOX"][1])
    )


def frame_row(pc: np.ndarray, labels, eef: np.ndarray, grid: str, ep: int, frame: int) -> dict:
    """One frame's occupancy numbers, plus what the eef draw achieves at each budget."""
    d = np.linalg.norm(pc[:, :3] - eef, axis=1)
    row = {"grid": grid, "ep": ep, "frame": frame, "n_total": len(pc)}
    for r in RADII:
        row[f"n_r{r}"] = int((d <= r).sum())
    if labels is not None:
        row["n_handle"] = int((labels == POINT_LABEL_TARGET_HANDLE).sum())

    w = eef_density_weights(pc[:, :3], eef, SIGMA, FLOOR)
    for budget in BUDGETS:
        # Mirrors LeRobotPointCloudDataset.augment_point_cloud's count rule, minus the 0.8-1.0
        # jitter (fixed at 0.9) so the columns are comparable across grids.
        k = min(int(len(pc) * 0.9), budget)
        if k >= len(pc):
            row[f"drawn_n{budget}"] = int((d <= 0.08).sum())
        else:
            idx = density_weighted_indices(len(pc), k, w, np.random.default_rng(1))
            row[f"drawn_n{budget}"] = int((d[idx] <= 0.08).sum())
    return row


def read_lmdb_cache(root: Path, episodes, per_ep: int, ws: dict) -> list[dict]:
    """The 1 cm baseline, from the LMDB the dataset actually trains on."""
    pts = lmdb.open(str(root / "points_3views"), readonly=True, lock=False, readahead=False)
    labs_dir = root / "points_3views_labels"
    labs = lmdb.open(str(labs_dir), readonly=True, lock=False, readahead=False) if labs_dir.exists() else None
    rng = np.random.default_rng(0)

    rows = []
    with pts.begin(buffers=True) as ptxn:
        ltxn = labs.begin(buffers=True) if labs is not None else None
        for ep in episodes:
            pq = next(root.glob(f"data/*/episode_{ep:06d}.parquet"), None)
            if pq is None:
                continue
            df = pd.read_parquet(pq)
            for frame in sorted(rng.choice(len(df), size=min(per_ep, len(df)), replace=False)):
                raw = ptxn.get(f"{ep}-{frame}".encode())
                if raw is None:
                    continue
                pc = msgpack.unpackb(raw).copy().astype(np.float32)
                labels = None
                if ltxn is not None:
                    lraw = ltxn.get(f"{ep}-{frame}".encode())
                    if lraw is not None:
                        labels = np.asarray(msgpack.unpackb(lraw).copy())
                m = workspace_mask(pc, ws)
                pc = pc[m]
                labels = labels[m] if labels is not None else None
                eef = np.asarray(df["observation.state"].iloc[int(frame)][:3], dtype=np.float64)
                rows.append(frame_row(pc, labels, eef, "1cm (cache)", ep, int(frame)))
    return rows


def read_replay_cache(cache_dir: Path, per_ep: int, ws: dict) -> list[dict]:
    """A probe cache: one npz + one concatenated points npy per episode."""
    meta = json.loads((cache_dir / "cache_meta.json").read_text())
    grid = f"{meta['voxel_size'] * 1000:g}mm"
    rng = np.random.default_rng(0)

    rows = []
    for npz_path in sorted((cache_dir / "episodes").glob("episode_*.npz")):
        if npz_path.stem.endswith(("_points", "_point_labels", "_voxel_keys")):
            continue
        ep = int(npz_path.stem.split("_")[-1])
        cache = np.load(npz_path, allow_pickle=False)
        offsets = cache["point_cloud_offsets"]
        states = cache["observation_state"]
        points = np.load(npz_path.with_name(f"{npz_path.stem}_points.npy"), mmap_mode="r")
        lab_path = npz_path.with_name(f"{npz_path.stem}_point_labels.npy")
        all_labels = np.load(lab_path, mmap_mode="r") if lab_path.exists() else None

        n_frames = len(states)
        for frame in sorted(rng.choice(n_frames, size=min(per_ep, n_frames), replace=False)):
            lo, hi = int(offsets[frame]), int(offsets[frame + 1])
            pc = np.asarray(points[lo:hi], dtype=np.float32)
            labels = np.asarray(all_labels[lo:hi]) if all_labels is not None else None
            # replay.py already cropped to the workspace; re-applying is a no-op that keeps
            # this path identical to the LMDB one.
            m = workspace_mask(pc, ws)
            pc = pc[m]
            labels = labels[m] if labels is not None else None
            eef = np.asarray(states[int(frame)][:3], dtype=np.float64)
            rows.append(frame_row(pc, labels, eef, grid, ep, int(frame)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="OpenDrawer")
    parser.add_argument("--lmdb-root", type=Path, default=None,
                        help="1 cm cache; defaults to robot_data/.../<task>.")
    parser.add_argument("--probe-root", type=Path, default=None,
                        help="Parent of the v<size> probe caches; defaults to $SCRATCH/voxel_probe/<task>.")
    parser.add_argument("--episodes", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--per-episode", type=int, default=20)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    lmdb_root = args.lmdb_root or repo / "robot_data/robocasa365/lerobot_point_lmdb" / args.task
    probe_root = args.probe_root or Path(os.environ["SCRATCH"]) / "voxel_probe" / args.task

    ws = json.loads((lmdb_root / "cache_meta.json").read_text())["workspace"]

    rows = read_lmdb_cache(lmdb_root, args.episodes, args.per_episode, ws)
    for cache_dir in sorted(probe_root.glob("v*")):
        if (cache_dir / "cache_meta.json").exists():
            rows += read_replay_cache(cache_dir, args.per_episode, ws)

    t = pd.DataFrame(rows)
    if t.empty:
        raise SystemExit("no frames read -- has the probe finished?")

    cols = ["n_total", *(f"n_r{r}" for r in RADII), *(["n_handle"] if "n_handle" in t else []),
            *(f"drawn_n{b}" for b in BUDGETS)]
    order = ["1cm (cache)", "5mm", "2mm"]
    grids = [g for g in order if g in set(t.grid)] + [g for g in sorted(set(t.grid)) if g not in order]

    pd.set_option("display.width", 220)
    print(f"frames: {len(t)}  episodes: {sorted(t.ep.unique())}\n")
    print("MEDIAN per frame")
    print(t.groupby("grid")[cols].median().reindex(grids).round(0).astype("Int64").to_string())
    print("\nMEAN per frame")
    print(t.groupby("grid")[cols].mean().reindex(grids).round(1).to_string())

    base = t[t.grid == "1cm (cache)"][cols].median()
    print("\nRATIO to the 1 cm grid (medians) -- >1 means the finer grid resolves more")
    print((t.groupby("grid")[cols].median().reindex(grids) / base).round(2).to_string())

    print("\nSaturation: ROI points drawn at 4096, as a share of those available within 8 cm")
    share = (t.groupby("grid")["drawn_n4096"].median() / t.groupby("grid")["n_r0.08"].median())
    print(share.reindex(grids).apply(lambda v: f"  {v:.0%}").to_string())


if __name__ == "__main__":
    main()
