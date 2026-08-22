"""Attribute a run's cost between the dataloader and the GPU, at a given point budget.

The Stage 6 no-sampler arm trains on the whole ~22K-point cloud instead of a 4096-point
draw, and its measured 1.45 s/step raised the obvious question: is that the network, or is
it the CPU shipping 5x more points per sample?

The answer is not obvious from the config, because the budget is applied LATE. Every arm --
4096, 8192, unsampled -- reads the full cloud out of LMDB, unpacks it and crops it to the
workspace; only then does augment_point_cloud() draw the subset (data_3d.py:488). So the
arms differ in the *tail* of __getitem__, not in the bulk of it, and the marginal cost of
the unsampled arm is much smaller than the 5x point ratio suggests.

This measures rather than argues it. Two numbers come out:

  * per-stage milliseconds inside __getitem__, at each budget, so the tail is visible;
  * the throughput a real multi-worker DataLoader sustains, in samples/s, which is the only
    figure comparable to the `train_samples_per_second` HF prints. If the loader's rate
    comfortably exceeds training's, the dataloader is not what is setting the pace.

Note the per-process arithmetic when comparing: training runs `dataloader_num_workers`
workers per GPU process, so a 4-GPU run at 88 samples/s needs 22 samples/s out of each
process's worker pool, not 88.

    python experiments/13_robocasa365/profile_dataloader.py \
        experiments/13_robocasa365/runs/s6-od-nosampler-s0.yaml --npoints 4096 32768
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pointact.constants import OBS_POINTS  # noqa: E402
from pointact.data.robot.multi_data import load_single_lerobot_dataset  # noqa: E402
from pointact.data.schema import LerobotConfig  # noqa: E402
from pointact.train.run_config import resolve_run_config  # noqa: E402


class Timer:
    """Accumulates wall time per named stage."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, name: str, seconds: float) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + seconds
        self.counts[name] = self.counts.get(name, 0) + 1

    def ms(self, name: str) -> float:
        return 1000.0 * self.totals[name] / max(1, self.counts[name])


def stage_profile(ds, idxs: list[int], timer: Timer) -> list[int]:
    """Re-walk __getitem__ stage by stage, timing each.

    Deliberately a copy of the dataset's own sequence rather than a hook into it: the point
    is to see the split, and a copy that drifts from data_3d.py is caught by the end-to-end
    row below, which calls the real __getitem__ and must agree with the sum.
    """
    # One untimed pass over EVERY sampled frame, not a prefix of them. Without it the first
    # budget measured absorbs the cold page-cache misses and reports ~27 ms for a read that
    # costs 0.15 ms warm -- putting the whole cost on the one stage that does not scale with
    # the budget. A 16-frame prefix was not enough and left exactly that artefact in run
    # 1271201: its n4096 row reads 26.88 ms against 0.14 ms for the two warm budgets below it,
    # and its stage sum (31 ms) contradicts its own end-to-end figure (3.49 ms).
    for i in idxs:
        item = ds.hf_dataset[i]
        ds.load_point_cloud(int(item["episode_index"]), int(item["frame_index"]))

    kept = []
    for i in idxs:
        t0 = time.perf_counter()
        item = ds.hf_dataset[i]
        t1 = time.perf_counter()
        ep, fr = int(item["episode_index"]), int(item["frame_index"])

        # Not optional scaffolding: augment_point_cloud rotates item[ACTION], which is (T, D)
        # only after the chunk query -- the raw row holds a single action and the rotation
        # raises IndexError on it. So this is part of the sequence, and it is timed.
        item, _ = ds.query_action_chunk(item, i, ep, ds.delta_indices)
        t2 = time.perf_counter()

        cloud = ds.load_point_cloud(ep, fr)
        t3 = time.perf_counter()
        raw_n = len(cloud)

        cloud, _ = ds.filter_point_cloud_by_workspace(cloud, None)
        t4 = time.perf_counter()
        cropped_n = len(cloud)

        cloud = ds.augment_point_cloud(cloud, item)
        t5 = time.perf_counter()
        kept.append(len(cloud))

        timer.add("hf_dataset[i]", t1 - t0)
        timer.add("query_action_chunk", t2 - t1)
        timer.add("load_point_cloud (lmdb+msgpack)", t3 - t2)
        timer.add("workspace crop", t4 - t3)
        timer.add("augment (draw+colour+rot)", t5 - t4)
        timer.add("_raw_n", raw_n / 1e9)          # smuggled through as a mean
        timer.add("_cropped_n", cropped_n / 1e9)
    return kept


def collate_points(examples: list[dict]) -> dict:
    """The point half of pointact.data.collators.DataCollator: cat, never pad."""
    points = [e[OBS_POINTS] for e in examples]
    return {
        "points": torch.cat(points, dim=0),
        "npoints_in_batch": torch.LongTensor([len(p) for p in points]),
    }


def loader_throughput(ds, batch_size: int, workers: int, batches: int, warmup: int) -> tuple:
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=workers,
        collate_fn=collate_points, drop_last=True,
        persistent_workers=False, prefetch_factor=2 if workers else None,
    )
    it = iter(loader)
    # Warm-up batches are discarded: the first ones pay worker fork, lmdb open and page-cache
    # misses, none of which recur over a 50K-step run.
    for _ in range(warmup):
        next(it)
    started = time.perf_counter()
    n_samples, n_points = 0, 0
    for _ in range(batches):
        batch = next(it)
        n_samples += int(batch["npoints_in_batch"].numel())
        n_points += int(batch["points"].shape[0])
    elapsed = time.perf_counter() - started
    return n_samples / elapsed, n_points / n_samples, elapsed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_config", type=Path)
    ap.add_argument("--npoints", nargs="+", type=int, default=None,
                    help="Budgets to compare. Default: the config's own, alone.")
    ap.add_argument("--samples", type=int, default=64, help="Frames for the per-stage pass.")
    ap.add_argument("--batches", type=int, default=20, help="Timed batches per budget.")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--workers", type=int, default=None,
                    help="Default: the config's dataloader_num_workers (per-GPU-process).")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Default: the config's per_device_train_batch_size.")
    args = ap.parse_args()

    meta, data, train = resolve_run_config(args.run_config)
    cfg = dict(data["lerobot_datasets"][0])
    workers = args.workers if args.workers is not None else train.get("dataloader_num_workers", 8)
    batch_size = args.batch_size or train.get("per_device_train_batch_size", 16)
    budgets = args.npoints or [cfg.get("max_npoints")]

    print(f"run={train.get('run_name')} arm={meta.get('sampling')} "
          f"config budget={cfg.get('max_npoints')}")
    print(f"loader: batch {batch_size} x {workers} workers  (per GPU process)")
    print(f"cpus visible: {os.cpu_count()}  affinity: {len(os.sched_getaffinity(0))}\n")

    ds = load_single_lerobot_dataset(0, [LerobotConfig(**cfg)], chunk_size=train["chunk_size"])
    rng = np.random.default_rng(0)
    idxs = rng.choice(ds.num_frames, size=min(args.samples, ds.num_frames),
                      replace=False).tolist()

    rows = []
    for budget in budgets:
        ds.max_npoints = int(budget)
        timer = Timer()
        kept = stage_profile(ds, idxs, timer)

        # The real __getitem__, for the stages the walk above leaves out (action chunk,
        # centring, normalisation, text-context lookup, post_process) and as a check that the
        # walk did not drift from it.
        started = time.perf_counter()
        for i in idxs:
            ds[i]
        end_to_end = 1000.0 * (time.perf_counter() - started) / len(idxs)

        rate, mean_points, elapsed = loader_throughput(
            ds, batch_size, workers, args.batches, args.warmup)

        print(f"--- max_npoints = {budget} " + "-" * 40)
        print(f"  cloud: {1e9 * timer.totals['_raw_n'] / timer.counts['_raw_n']:,.0f} raw -> "
              f"{1e9 * timer.totals['_cropped_n'] / timer.counts['_cropped_n']:,.0f} cropped -> "
              f"{np.median(kept):,.0f} kept (median)")
        for name in ("hf_dataset[i]", "query_action_chunk",
                     "load_point_cloud (lmdb+msgpack)", "workspace crop",
                     "augment (draw+colour+rot)"):
            print(f"  {name:>34s}: {timer.ms(name):7.2f} ms")
        walked = sum(timer.ms(n) for n in timer.totals if not n.startswith("_"))
        print(f"  {'(sum of the above)':>34s}: {walked:7.2f} ms")
        print(f"  {'full __getitem__, 1 worker':>34s}: {end_to_end:7.2f} ms  "
              f"-> {1000 / end_to_end:.1f} samples/s single-threaded")
        print(f"  {'DataLoader, ' + str(workers) + ' workers':>34s}: "
              f"{rate:7.1f} samples/s  ({args.batches} batches in {elapsed:.1f}s, "
              f"{mean_points:,.0f} points/sample)\n")
        rows.append((budget, end_to_end, rate))

    print("=" * 66)
    print(f"{'budget':>10s} | {'ms/sample (1 proc)':>19s} | {'samples/s (pool)':>17s}")
    print("-" * 66)
    for budget, ms, rate in rows:
        print(f"{budget:>10,d} | {ms:>19.2f} | {rate:>17.1f}")
    print()
    print("Compare the right-hand column against train_samples_per_second / n_gpus: that is")
    print("the rate ONE process's worker pool has to sustain. Well above it means the GPU is")
    print("the constraint and the point budget is buying network time, not loader time.")


if __name__ == "__main__":
    main()
