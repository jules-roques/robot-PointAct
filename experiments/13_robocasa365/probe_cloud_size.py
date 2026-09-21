"""Measure a task's point-cloud size straight from the point LMDB.

Why this exists as a separate tool from ``smoke_stage5.py``: that one goes through the
LeRobot dataset, so it needs the per-episode parquet files. When the $SCRATCH purge ate
those (2026-09-21) the LMDBs survived intact, and the cloud size is exactly the number
needed to set the no-sampler arm's ``per_device_train_batch_size`` and its ``max_npoints``
cap. This reads the LMDB directly and needs no parquet, no metadata and no GPU.

It reproduces what the model actually sees before sampling: the LMDB is already voxelised
at the grid the directory name implies (``lerobot_point_lmdb`` = 1 cm), and the only other
thing ``__getitem__`` does before drawing a budget is the workspace crop, which is applied
here with the same box.

    python experiments/13_robocasa365/probe_cloud_size.py \
        --lmdb robot_data/robocasa365/lerobot_point_lmdb/CloseBlenderLid/points_3views \
        --frames 400

Keys are ``"{episode}-{frame}"``; the probe samples keys rather than walking all of them,
because these LMDBs are 60-97 GB and a full pass is minutes of IO for a median that has
converged after a few hundred draws.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import lmdb
import msgpack
import msgpack_numpy
import numpy as np

msgpack_numpy.patch()

#: The same box `_base.yaml` sets, and the only transform between the LMDB and the sampler.
WORKSPACE = {"X_BBOX": (-0.8, 0.8), "Y_BBOX": (-0.8, 0.8), "Z_BBOX": (0.0, 1.0)}


def crop(cloud: np.ndarray) -> np.ndarray:
    m = (
        (cloud[:, 0] > WORKSPACE["X_BBOX"][0]) & (cloud[:, 0] < WORKSPACE["X_BBOX"][1])
        & (cloud[:, 1] > WORKSPACE["Y_BBOX"][0]) & (cloud[:, 1] < WORKSPACE["Y_BBOX"][1])
        & (cloud[:, 2] > WORKSPACE["Z_BBOX"][0]) & (cloud[:, 2] < WORKSPACE["Z_BBOX"][1])
    )
    return cloud[m]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lmdb", type=Path, required=True, help="points_3views directory")
    ap.add_argument("--frames", type=int, default=400, help="Keys to sample.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = lmdb.open(str(args.lmdb), readonly=True, lock=False, readahead=False,
                    meminit=False, max_readers=2048)
    with env.begin(write=False) as txn:
        total = txn.stat()["entries"]
        # Collect the key space once. cursor.iternext(values=False) is cheap -- it never
        # touches the 60-97 GB of values, only the key pages.
        keys = [k for k in txn.cursor().iternext(values=False)]
        rng = random.Random(args.seed)
        picked = keys if len(keys) <= args.frames else rng.sample(keys, args.frames)

        raw, cropped = [], []
        for k in picked:
            cloud = msgpack.unpackb(txn.get(k)).copy().astype(np.float32)
            raw.append(len(cloud))
            cropped.append(len(crop(cloud)))
    env.close()

    raw_a, crop_a = np.asarray(raw), np.asarray(cropped)
    print(f"lmdb   : {args.lmdb}")
    print(f"entries: {total} keys, sampled {len(picked)}")
    for name, a in (("raw", raw_a), ("workspace-cropped", crop_a)):
        print(f"  {name:18s} median {int(np.median(a)):6d}  mean {a.mean():8.1f}  "
              f"p95 {int(np.percentile(a, 95)):6d}  min {a.min():6d}  max {a.max():6d}")

    # What the two numbers the no-sampler arm needs actually are.
    cap = int(np.percentile(crop_a, 100))
    print(f"\n  a max_npoints cap must exceed the largest CROPPED cloud ({cap}) or it binds")
    print(f"  and the arm silently becomes a large uniform draw.")


if __name__ == "__main__":
    main()
