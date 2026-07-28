"""Write ground-truth per-point labels into an LMDB keyed to an EXISTING converted dataset.

`replay.py --labels-only` caches labels per *source* episode. Two things have to be reconciled:

1. Episode numbering. The converted dataset indexes episodes by their rank among successfully
   replayed episodes, so it differs from the source numbering whenever an episode failed.
2. Point identity. Replaying is not bit-deterministic across machines, so a frame's replayed
   cloud can have one or two points more or less than the stored one. Labels therefore cannot
   be attached by array position; they are joined on the packed voxel key that both clouds
   derive from the same voxel grid.

Frames whose join leaves points unmatched are reported, and the run aborts if the rate exceeds
--max-unmatched-frac, since silently misaligned labels would poison the oracle experiment in a
way training curves would not reveal.

    python data_prep/robocasa365_to_lerobot/export_point_labels.py \
        --cache-dir  .../replay_cache_labels \
        --dataset-dir .../lerobot_point_lmdb/OpenDrawer
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lmdb
import msgpack
import msgpack_numpy
import numpy as np
from tqdm.auto import tqdm

from data_prep.robocasa365_to_lerobot.voxel_keys import voxel_keys_for_points

msgpack_numpy.patch()

NUM_POINT_LABELS = 5
LABEL_NAMES = ["background", "robot", "target_other", "target_door", "target_handle"]


def cached_episode_indices(cache_dir: Path) -> list[int]:
    files = sorted((cache_dir / "episodes").glob("episode_*.npz"))
    return [int(path.stem.split("_")[-1]) for path in files]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--point-cloud-dirname", default="points_3views")
    parser.add_argument("--label-dirname", default=None,
                        help="Defaults to '<point-cloud-dirname>_labels'.")
    parser.add_argument("--lmdb-map-size", type=int, default=int(1024**4))
    parser.add_argument("--lmdb-commit-every", type=int, default=200)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--max-unmatched-frac", type=float, default=1e-3,
                        help="Abort if more than this fraction of points fail to join.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cache_dir = args.cache_dir.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    label_dirname = args.label_dirname or f"{args.point_cloud_dirname}_labels"

    points_env = lmdb.open(
        str(dataset_dir / args.point_cloud_dirname), readonly=True, lock=False, readahead=False
    )
    points_txn = points_env.begin(buffers=True)

    source_indices = cached_episode_indices(cache_dir)
    if not source_indices:
        raise SystemExit(f"No cached episodes under {cache_dir / 'episodes'}")
    print(f"{len(source_indices)} cached episodes -> dataset indices 0..{len(source_indices) - 1}")

    label_env = lmdb.open(str(dataset_dir / label_dirname), map_size=int(args.lmdb_map_size))

    totals = np.zeros(NUM_POINT_LABELS, dtype=np.int64)
    n_frames = 0
    n_points_total = 0
    n_unmatched_total = 0
    worst_frame = (0.0, None)
    txn = label_env.begin(write=True)
    pending = 0
    try:
        for dataset_index, source_index in enumerate(
            tqdm(source_indices, desc="episodes", unit="ep", dynamic_ncols=True)
        ):
            stem = f"episode_{source_index:06d}"
            with np.load(cache_dir / "episodes" / f"{stem}.npz", allow_pickle=True) as data:
                offsets = data["point_cloud_offsets"].astype(np.int64)
            labels = np.load(cache_dir / "episodes" / f"{stem}_point_labels.npy", mmap_mode="r")
            keys = np.load(cache_dir / "episodes" / f"{stem}_voxel_keys.npy", mmap_mode="r")

            for frame_index in range(len(offsets) - 1):
                key = f"{dataset_index}-{frame_index}".encode("ascii")
                buf = points_txn.get(key)
                if buf is None:
                    raise SystemExit(
                        f"source episode {source_index} -> dataset episode {dataset_index}: "
                        f"frame {frame_index} missing from the points LMDB. The cached episodes "
                        f"do not correspond to this dataset."
                    )
                ref_cloud = msgpack.unpackb(buf)
                ref_keys = voxel_keys_for_points(
                    np.asarray(ref_cloud[:, :3], dtype=np.float32), args.voxel_size
                )

                lo, hi = offsets[frame_index], offsets[frame_index + 1]
                src_labels = np.asarray(labels[lo:hi], dtype=np.uint8)
                src_keys = np.asarray(keys[lo:hi], dtype=np.int32)

                # Join replayed labels onto the stored cloud by voxel identity. Both key
                # arrays are sorted (voxel_downsample emits points in voxel order), so
                # searchsorted is an exact merge; unmatched points keep BACKGROUND.
                pos = np.searchsorted(src_keys, ref_keys)
                pos = np.clip(pos, 0, len(src_keys) - 1)
                matched = src_keys[pos] == ref_keys
                frame_labels = np.where(matched, src_labels[pos], 0).astype(np.uint8)

                n_unmatched = int((~matched).sum())
                n_unmatched_total += n_unmatched
                n_points_total += len(ref_keys)
                frac = n_unmatched / max(len(ref_keys), 1)
                if frac > worst_frame[0]:
                    worst_frame = (frac, (source_index, frame_index))

                txn.put(key, msgpack.packb(frame_labels))
                totals += np.bincount(frame_labels, minlength=NUM_POINT_LABELS)
                n_frames += 1

                pending += 1
                if args.lmdb_commit_every > 0 and pending >= args.lmdb_commit_every:
                    txn.commit()
                    txn = label_env.begin(write=True)
                    pending = 0
        txn.commit()
        txn = None
    finally:
        if txn is not None:
            txn.abort()
        label_env.close()
        points_env.close()

    total_points = int(totals.sum())
    unmatched_frac = n_unmatched_total / max(n_points_total, 1)
    print(f"\nwrote {n_frames} frames / {total_points} points to {dataset_dir / label_dirname}")
    print(f"  unmatched points: {n_unmatched_total}/{n_points_total} ({unmatched_frac:.2e}); "
          f"worst frame {worst_frame[1]} at {worst_frame[0]:.2e}")
    if unmatched_frac > args.max_unmatched_frac:
        raise SystemExit(
            f"Unmatched fraction {unmatched_frac:.2e} exceeds --max-unmatched-frac "
            f"{args.max_unmatched_frac:.2e}; labels are not trustworthy."
        )
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name:14s}: {totals[i]:10d}  ({totals[i] / max(total_points, 1):.4f})")
    roi = int(totals[3] + totals[4])
    print(
        f"\ndoor+handle = {roi / max(total_points, 1):.4f} of points; a uniform 4096-point "
        f"budget lands ~{roi / max(total_points, 1) * 4096:.0f} of them"
    )


if __name__ == "__main__":
    main()
