"""Stage 8's x-axis: how many points the oracle draw puts near the handle, per grid.

Reads the point LMDBs directly (no parquet, no GPU) and runs each frame through exactly what
the dataloader does for a stage-8 arm -- workspace crop, re-voxelize onto the arm's grid
(pointact.data.robot.voxel), then the oracle Gaussian draw at the budget -- and reports the
handle neighbourhood before and after the draw. The same frames are used for every grid, so
the columns differ only in resolution.

    python experiments/13_robocasa365/probe_roi_resolution.py \
        --fine-root $SCRATCH/datasets/robot_data/robocasa365/lerobot_point_lmdb_g2.5mm/OpenDrawer

The headline column is `drawn<=4cm`: the ROI points the policy actually receives. Stage 7 at
1 cm puts ~96-100% of what exists there into the draw at 4096-8192, so a finer grid should
multiply this number roughly by the handle column of the voxel probe (71 -> 216 -> 730 at
1 cm -> 5 mm -> 2 mm). If it does not -- if the floor's background share eats the budget at
fine grids -- the arm is not testing what it claims to.
"""

import argparse
import os
import random
import sys
from pathlib import Path

import lmdb
import msgpack
import msgpack_numpy
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from pointact.data.robot.voxel import voxel_downsample  # noqa: E402
from pointact.roi_sampling.geom_gt import load_episode_map, load_targets  # noqa: E402
from pointact.roi_sampling.geometry import eef_density_weights  # noqa: E402
from pointact.roi_sampling.sampling import density_weighted_indices  # noqa: E402
from probe_cloud_size import crop  # noqa: E402

msgpack_numpy.patch()
RADII = (0.02, 0.04, 0.08)
SIGMA, FLOOR = 0.08, 0.05


def open_txn(root: Path):
    env = lmdb.open(str(root / "points_3views"), readonly=True, lock=False,
                    readahead=False, meminit=False)
    return env.begin()


def main() -> None:
    default_1cm = os.path.expandvars(
        "$SCRATCH/datasets/robot_data/robocasa365/lerobot_point_lmdb/OpenDrawer")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coarse-root", type=Path, default=Path(default_1cm))
    parser.add_argument("--fine-root", type=Path, default=None)
    parser.add_argument("--fine-grids", type=float, nargs="+",
                        default=[0.01 / np.sqrt(2), 0.005, 0.0025])
    parser.add_argument("--budget", type=int, default=8192)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # The oracle anchor is keyed by (episode, frame) of the COARSE dataset; the fine one is a
    # re-render of the same episodes, which is checked below rather than assumed.
    lookup = load_targets(args.coarse_root / "roi_meta/target_positions.npz", ["handle"],
                          load_episode_map(args.coarse_root))
    arms = [("1cm", args.coarse_root, 0.01)]
    if args.fine_root is not None:
        arms += [(f"{g * 1000:.3g}mm", args.fine_root, g) for g in args.fine_grids]
    txns = {root: open_txn(root) for _, root, _ in arms}

    coarse = txns[args.coarse_root]
    keys = [k for k in coarse.cursor().iternext(values=False)]
    random.Random(args.seed).shuffle(keys)
    rng = np.random.default_rng(args.seed)

    stats = {name: {"cloud": [], "drawn_share8": [],
                    **{f"avail{r}": [] for r in RADII}, **{f"drawn{r}": [] for r in RADII}}
             for name, _, _ in arms}
    used = 0
    for key in keys:
        ep, fr = map(int, bytes(key).decode().split("-"))
        anchor = lookup("handle", ep, fr)
        if anchor is None:
            continue
        clouds = {}
        for name, root, grid in arms:
            raw = txns[root].get(bytes(key))
            if raw is None:
                break
            cloud = crop(msgpack.unpackb(raw).astype(np.float32))
            # Onto the cache's own grid this is a no-op, so every arm can take the same path.
            clouds[name] = voxel_downsample(cloud, grid)
        if len(clouds) != len(arms):
            continue
        for name, cloud in clouds.items():
            d = np.linalg.norm(cloud[:, :3] - anchor, axis=1)
            m = min(int(len(cloud) * rng.uniform(0.8, 1.0)), args.budget)
            if len(cloud) > m:
                w = eef_density_weights(cloud[:, :3], anchor, SIGMA, FLOOR)
                idx = density_weighted_indices(len(cloud), m, w, rng)
            else:
                idx = np.arange(len(cloud))
            s = stats[name]
            s["cloud"].append(len(cloud))
            s["drawn_share8"].append((d[idx] <= 0.08).mean())
            for r in RADII:
                s[f"avail{r}"].append((d <= r).sum())
                s[f"drawn{r}"].append((d[idx] <= r).sum())
        used += 1
        if used == args.frames:
            break

    print(f"{used} OpenDrawer frames, oracle draw at {args.budget}, sigma {SIGMA} floor {FLOOR}; "
          "medians per frame")
    head = f"{'grid':>8} {'cloud':>7} " + " ".join(
        f"{'avail<=' + str(int(r * 100)) + 'cm':>11} {'drawn<=' + str(int(r * 100)) + 'cm':>11}"
        for r in RADII) + f" {'share<=8cm':>11}"
    print(head)
    for name, _, _ in arms:
        s = stats[name]
        med = lambda k: int(np.median(s[k]))  # noqa: E731
        print(f"{name:>8} {med('cloud'):>7} " + " ".join(
            f"{med(f'avail{r}'):>11} {med(f'drawn{r}'):>11}" for r in RADII)
            + f" {100 * np.median(s['drawn_share8']):>10.0f}%")


if __name__ == "__main__":
    main()
