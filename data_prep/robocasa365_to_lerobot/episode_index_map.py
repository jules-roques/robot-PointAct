"""Recover which source episode each converted episode came from.

``convert.py`` numbers its output episodes densely, by position among the replay cache files
it found, and drops any episode whose replay failed. So OpenDrawer's 514 source episodes
became 496 converted ones under a *renumbering* -- converted episode 3 is source episode 11 --
and the correspondence is printed at conversion time but never written down. Anything that
joins converted data (point clouds, anchor caches, labels) to something derived from the
source episodes (a simulator replay, an initial-state dir) needs it back.

TurnOnMicrowave lost no episodes and its map is the identity; do not assume that, check it.

The join key is the **whole action sequence**. Both datasets store it and ``replay.py`` only
reorders each action's components without changing their values, so the converted array is
bit-identical to the reordered source prefix. Three weaker keys all fail here and are worth
naming, because each looks like it should work:

* *A prefix of the actions* -- even 20 of them. Many demonstrations open with the same
  scripted approach, so short prefixes are shared by dozens of episodes and a forward scan
  silently locks onto the wrong one.
* *Episode length.* The converted length is the replayed length, which differs from the
  source length whenever the episode ended early.
* *The instruction.* The source's ``ep_meta.json`` lang disagrees with the instruction the
  replay recorded for about a third of OpenDrawer episodes. The converted one is the one the
  labels and the pointing queries were built from, so it is the authoritative one -- but it
  makes the source-side string useless as a key, and useless as a check.

The map is verified, not trusted: every converted episode's length must equal the
``replay_length`` its mapped source episode recorded in ``replay_summary.jsonl``. On
OpenDrawer that check passes 496/496 for the derived map and 4/496 for the identity.

    python -m data_prep.robocasa365_to_lerobot.episode_index_map \
        --source-dir $SCRATCH/.../OpenDrawer/20250816/lerobot \
        --dataset-dir $SCRATCH/.../lerobot_point_lmdb/OpenDrawer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm

#: Where the map is cached, relative to the converted dataset.
MAP_NAME = "source_episode_map.json"


def reorder_source_actions(actions: np.ndarray) -> np.ndarray:
    """Source (T, 12) axis-angle actions -> the (T, 13) xyzw form the converted data stores.

    Duplicated from ``replay.py:reorder_source_action_to_env`` rather than imported: that
    module pulls in MuJoCo, and the whole point of this one is to run without a simulator.
    Keep the two in step.
    """
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 12:
        raise ValueError(f"Expected source actions (T, 12), got {a.shape}")
    q = R.from_rotvec(a[:, 8:11]).as_quat().astype(np.float32)
    return np.concatenate(
        [a[:, 5:8], q, a[:, 11:12], a[:, 0:4], a[:, 4:5]], axis=1).astype(np.float32)


def episode_actions(dataset_dir: Path, episode: int, chunks_size: int) -> np.ndarray:
    p = (dataset_dir / "data" / f"chunk-{episode // chunks_size:03d}"
         / f"episode_{episode:06d}.parquet")
    t = pq.read_table(p, columns=["action"])
    return np.array([np.asarray(x, dtype=np.float32) for x in t.column("action").to_pylist()])


def read_chunks_size(dataset_dir: Path) -> int:
    return int(json.loads((dataset_dir / "meta" / "info.json").read_text())
               .get("chunks_size", 1000))


def count_episodes(dataset_dir: Path) -> int:
    with open(dataset_dir / "meta" / "episodes.jsonl") as f:
        return sum(1 for _ in f)


def build_map(source_dir: Path, dataset_dir: Path) -> dict[int, int]:
    """{converted episode -> source episode}, by exact whole-sequence match.

    Every source episode is compared against every converted one rather than scanned in
    order, because conversion order is not something this can afford to assume. A converted
    episode matching zero or several source episodes is a hard error: a partial map would
    join some rows correctly and others silently wrong, which is worse than no map.

    The converted array carries one extra terminal no-op action when the episode ended early
    (``replay.py`` appends it so the frame arrays stay aligned), so the comparison drops the
    last row.
    """
    src_chunks = read_chunks_size(source_dir)
    dst_chunks = read_chunks_size(dataset_dir)
    n_src = count_episodes(source_dir)
    n_dst = count_episodes(dataset_dir)

    source = [reorder_source_actions(episode_actions(source_dir, e, src_chunks))
              for e in tqdm(range(n_src), desc="reading source", unit="ep")]

    mapping: dict[int, int] = {}
    for dst in tqdm(range(n_dst), desc="matching", unit="ep"):
        want = episode_actions(dataset_dir, dst, dst_chunks)
        n = max(1, len(want) - 1)
        cands = [j for j, s in enumerate(source)
                 if len(s) >= n and np.array_equal(s[:n], want[:n])]
        if len(cands) != 1:
            raise SystemExit(
                f"converted episode {dst} matched {len(cands)} source episodes ({cands[:5]}); "
                f"the datasets do not correspond, or the action reordering has drifted")
        mapping[dst] = cands[0]
    return mapping


def verify_map(dataset_dir: Path, mapping: dict[int, int]) -> tuple[int, int]:
    """(agreements, total) between converted episode length and the recorded replay length.

    Independent of the key the map was built from, so it is a real check rather than a
    restatement. Anything short of total agreement means the map is wrong.
    """
    summary = {}
    with open(dataset_dir / "replay_summary.jsonl") as f:
        for line in f:
            r = json.loads(line)
            summary[int(r["episode_index"])] = int(r["replay_length"])
    lengths = {}
    with open(dataset_dir / "meta" / "episodes.jsonl") as f:
        for line in f:
            r = json.loads(line)
            lengths[int(r["episode_index"])] = int(r["length"])
    ok = sum(1 for dst, src in mapping.items() if summary.get(src) == lengths[dst])
    return ok, len(mapping)


def load_map(dataset_dir: Path) -> dict[int, int] | None:
    """The cached map, or None. Keys come back as ints."""
    p = dataset_dir / "meta" / MAP_NAME
    if not p.exists():
        return None
    return {int(k): int(v) for k, v in json.loads(p.read_text())["converted_to_source"].items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True, type=Path)
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    src = args.source_dir.expanduser().resolve()
    dst = args.dataset_dir.expanduser().resolve()
    mapping = build_map(src, dst)

    ok, total = verify_map(dst, mapping)
    if ok != total:
        raise SystemExit(f"map failed verification: {ok}/{total} converted episodes have a "
                         f"length matching their mapped source episode's replay_length")

    identity = all(k == v for k, v in mapping.items())
    dropped = sorted(set(range(count_episodes(src))) - set(mapping.values()))
    out = args.out or dst / "meta" / MAP_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "source_dir": str(src),
        "identity": identity,
        "n_converted": len(mapping),
        "verified_length_agreement": f"{ok}/{total}",
        "dropped_source_episodes": dropped,
        "converted_to_source": {str(k): v for k, v in mapping.items()},
    }, indent=2))
    print(f"\n{dst.name}: {len(mapping)} converted episodes, identity={identity}, "
          f"{len(dropped)} source episodes dropped, length check {ok}/{total}")
    if dropped:
        print(f"  dropped: {dropped}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
