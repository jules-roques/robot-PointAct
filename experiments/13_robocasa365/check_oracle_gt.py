"""Pre-flight: confirm every oracle arm's geometry ground truth exists, under the key its
yaml names.

A missing key does not fail at submit time. It fails per-sample, deep inside the dataloader
worker pool, with a traceback that names the pickle machinery rather than the file that is
actually missing -- after the job has been queued and scheduled. Reading the npzs here costs
seconds.

    python experiments/13_robocasa365/check_oracle_gt.py \
        --lmdb-root $SCRATCH/robot_data/robocasa365/lerobot_point_lmdb
"""

import argparse
import os
import sys

import numpy as np

#: (task, geom key the run yaml asks for) for the stage-5 five-task grid. Extend when an arm
#: targets a new geom -- the point of this script is that the pairing is explicit somewhere.
DEFAULT_TARGETS = [
    ("OpenDrawer", "handle"),
    ("TurnOnMicrowave", "start_button"),
    ("PickPlaceCounterToStove", "obj"),
    ("CloseBlenderLid", "lid"),
    ("CoffeeSetupMug", "obj"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lmdb-root", required=True,
                        help="Directory holding one subdirectory per task.")
    parser.add_argument("--target", action="append", metavar="TASK:KEY", default=None,
                        help="Override the default task:key pairs. Repeatable.")
    args = parser.parse_args()

    targets = DEFAULT_TARGETS
    if args.target:
        targets = [tuple(t.split(":", 1)) for t in args.target]

    bad = 0
    for task, key in targets:
        path = os.path.join(args.lmdb_root, task, "roi_meta", "target_positions.npz")
        if not os.path.exists(path):
            print("%-26s MISSING FILE %s" % (task, path))
            bad += 1
            continue
        data = np.load(path)
        keys = list(data.keys())
        if key not in keys:
            print("%-26s MISSING KEY %r; has %s" % (task, key, keys[:8]))
            bad += 1
            continue
        arr = data[key]
        finite = np.isfinite(arr).all(axis=-1) if arr.ndim > 1 else np.isfinite(arr)
        print("%-26s ok  key=%-13s shape=%-14s finite=%d/%d  keys=%s"
              % (task, key, arr.shape, finite.sum(), finite.size, keys[:6]))

    print("BAD=%d" % bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
