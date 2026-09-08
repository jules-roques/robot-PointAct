"""What one training step costs, decomposed, as a function of the point budget.

The headline this feeds is "point policies are expensive, and here is where the money goes".
That needs three things the existing tools do not give together:

  * a **decomposition** -- data loading vs host-to-device vs forward vs backward vs optimiser
    -- so "expensive" can be attributed rather than asserted;
  * a **spread**, not just a mean, because a single s/step number invites the reader to
    compare two budgets whose distributions overlap;
  * **per-iteration** timing, so neither of the above is an average smeared over startup.

`pilot_throughput.py` answers a different question and answers it better: it reports the
end-to-end marginal rate a real 4-GPU `accelerate` run sustains, which is the number to quote
for "how long will this run take". It gets there by differencing two step counts, so it
yields one figure per configuration and no distribution. This script is the microscope, not
the clock: one process, one GPU, explicit `torch.cuda.synchronize()` around each timed
region, mean and standard deviation over N repetitions after a discarded warm-up. **Report
both**, and expect the sum here to sit below pilot_throughput's s/step -- this loop has no
gradient accumulation, no DDP all-reduce and no checkpoint writes.

Two properties of this pipeline the decomposition makes visible, and which are the actual
reason the naive "more points = proportionally slower" intuition is wrong:

  * **The budget is applied late.** Every arm reads the whole cloud out of LMDB and crops it
    to the workspace; only then is the subset drawn (`data_3d.py:488`). So the loader cost
    barely moves with the budget -- the arms differ in the tail of `__getitem__`, not the bulk.
  * **The network is sublinear in points.** Only PTv3's first stages see more tokens; the deep
    stages are volume-limited by the 1 cm voxel grid, and attention is windowed at
    `patch_size`. 4.6x the points has measured 1.46x the step time.

    python experiments/13_robocasa365/profile_point_cost.py \
        experiments/13_robocasa365/runs/s7-od-uniform-n4096-s0.yaml \
        --npoints 1024 2048 4096 8192 16384 --reps 30 --out cost.json

The model is built once and reused across budgets -- the architecture does not depend on the
point count, only the token count flowing through it does. PTv3 init weights are deliberately
NOT loaded: they change no shape and no timing, and loading 549 MB per invocation would be
the slowest thing here.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# Load the dataset inline instead of through MultiLeRobotDataset's worker Pool.
#
# Two reasons, one of which cost a 1h45 walltime. (a) The pool exists to parallelise across
# SEVERAL datasets; this profiler always has one, so it buys nothing and adds a pickle
# round-trip of the built dataset back to the parent. (b) Unlike scripts/train.py, this script
# runs as a plain single process rather than under `accelerate launch`, and forking the pool
# after the model has been constructed in the parent hangs -- job 1876820 sat at
# "load 1 lerobot datasets with 8 processes ..." until SLURM killed it, having produced no
# output at all. Setting this to 1 takes the `num_processes <= 1` branch, which never forks.
os.environ.setdefault("DATASET_NUM_PROCESSES", "1")
# Nothing here tokenizes in a loop, and the fork warning it emits is noise in a timing log.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pointact.train.script_utils import parse_training_args  # noqa: E402


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Samples:
    """Timed durations for one named region, in milliseconds."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.ms: list[float] = []

    def add(self, seconds: float) -> None:
        self.ms.append(1000.0 * seconds)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.ms) if self.ms else float("nan")

    @property
    def std(self) -> float:
        # Sample standard deviation: these are N draws from the step-time distribution, not
        # the whole population of steps a run will take.
        return statistics.stdev(self.ms) if len(self.ms) > 1 else 0.0

    def as_dict(self) -> dict:
        return {"mean_ms": self.mean, "std_ms": self.std, "n": len(self.ms),
                "min_ms": min(self.ms) if self.ms else None,
                "max_ms": max(self.ms) if self.ms else None}


def move_to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def profile_budget(model, optimizer, dataset, collator, budget: int, *, batch_size: int,
                   workers: int, reps: int, warmup: int, device: torch.device,
                   forward_only: bool) -> dict:
    """Time one budget end to end, region by region."""
    for ds in dataset.lerobot_dataset._datasets:
        ds.max_npoints = int(budget)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
        collate_fn=collator, drop_last=True,
        persistent_workers=False, prefetch_factor=2 if workers else None,
    )
    it = iter(loader)

    regions = {name: Samples(name) for name in
               ("dataload", "to_device", "forward", "backward", "optimizer", "step_total")}
    points_per_sample: list[float] = []

    torch.cuda.reset_peak_memory_stats(device) if torch.cuda.is_available() else None

    total = warmup + reps
    for i in range(total):
        timed = i >= warmup  # warm-up pays worker fork, lmdb open, page cache and autotuning

        t0 = time.perf_counter()
        batch = next(it)
        t1 = time.perf_counter()

        if "npoints_in_batch" in batch:
            n = batch["npoints_in_batch"]
            points_per_sample.append(float(n.float().mean()))

        batch = move_to_device(batch, device)
        _sync()
        t2 = time.perf_counter()

        if forward_only:
            with torch.no_grad():
                out = model(**batch)
            _sync()
            t3 = time.perf_counter()
            t4 = t5 = t3
        else:
            out = model(**batch)
            loss = out["loss"] if isinstance(out, dict) else out.loss
            _sync()
            t3 = time.perf_counter()

            loss.backward()
            _sync()
            t4 = time.perf_counter()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            _sync()
            t5 = time.perf_counter()

        if timed:
            regions["dataload"].add(t1 - t0)
            regions["to_device"].add(t2 - t1)
            regions["forward"].add(t3 - t2)
            regions["backward"].add(t4 - t3)
            regions["optimizer"].add(t5 - t4)
            regions["step_total"].add(t5 - t0)

    del it, loader
    peak_gb = (torch.cuda.max_memory_allocated(device) / 1024**3
               if torch.cuda.is_available() else 0.0)
    return {
        "budget": int(budget),
        "batch_size": batch_size,
        "points_per_sample": statistics.fmean(points_per_sample) if points_per_sample else None,
        "peak_gpu_gb": peak_gb,
        "regions": {k: v.as_dict() for k, v in regions.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_config", type=Path)
    ap.add_argument("--npoints", nargs="+", type=int, required=True,
                    help="Budgets to sweep. Ascending order reads best in the table.")
    ap.add_argument("--reps", type=int, default=30,
                    help="Timed iterations per budget. 10 is enough for a mean but thin for "
                         "a standard deviation, which is half of what this reports.")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Default: the config's per_device_train_batch_size. Hold it FIXED "
                         "across budgets or the columns are not comparable.")
    ap.add_argument("--workers", type=int, default=None,
                    help="Default: the config's dataloader_num_workers (per GPU process).")
    ap.add_argument("--forward-only", action="store_true",
                    help="Skip backward and the optimiser. The forward column is measured "
                         "either way; this only makes the sweep cheaper.")
    ap.add_argument("--out", type=Path, default=None, help="Write the full result as JSON.")
    args = ap.parse_args()

    # parse_training_args reads sys.argv, exactly as scripts/train.py does, so the run yaml is
    # interpreted by the same code path the real run uses -- no second opinion about what the
    # config means.
    sys.argv = [sys.argv[0], str(args.run_config)]
    training_args = parse_training_args()

    from train_registry import resolve_recipe  # noqa: E402  (needs scripts/ on sys.path)
    from train import _compute_dtype, build_model, load_processor, _import_object  # noqa: E402
    from pointact.train.train_utils import configure_processor  # noqa: E402
    from pointact.train.text_context import ensure_text_context  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recipe = resolve_recipe(training_args.model_class)
    compute_dtype = _compute_dtype(training_args)

    # Progress markers, because every step below is slow enough that silence is ambiguous:
    # the first version of this script hung for 1h45 inside create_data_module and the log
    # gave no way to tell that from "still loading the model".
    print("[1/3] building model ...", flush=True)
    model = build_model(recipe, training_args, compute_dtype)
    processor = load_processor(recipe, training_args)
    ensure_text_context(training_args)

    print("[2/3] building data module ...", flush=True)
    create_data_module = _import_object(recipe.data_module_fn)
    data_module = create_data_module(processor=processor, args=training_args)
    configure_processor(processor, data_module["train_dataset"], training_args)
    print("[3/3] ready; timing", flush=True)

    model.config.use_cache = False
    model.to(device)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=training_args.learning_rate)

    batch_size = args.batch_size or training_args.per_device_train_batch_size
    workers = args.workers if args.workers is not None else training_args.dataloader_num_workers

    print(f"run={training_args.run_name}  device={device}  dtype={compute_dtype}")
    print(f"batch={batch_size} (per device, held fixed)  workers={workers}  "
          f"reps={args.reps} after {args.warmup} warm-up")
    print(f"trainable params: {sum(p.numel() for p in trainable) / 1e6:.1f}M\n")

    rows = []
    for budget in args.npoints:
        row = profile_budget(
            model, optimizer, data_module["train_dataset"], data_module["data_collator"],
            budget, batch_size=batch_size, workers=workers, reps=args.reps,
            warmup=args.warmup, device=device, forward_only=args.forward_only,
        )
        rows.append(row)
        r = row["regions"]
        print(f"--- budget {budget:,} " + "-" * 46)
        print(f"  points/sample actually fed: {row['points_per_sample']:,.0f}   "
              f"peak GPU {row['peak_gpu_gb']:.1f} GB")
        for name in ("dataload", "to_device", "forward", "backward", "optimizer", "step_total"):
            s = r[name]
            print(f"  {name:>12s}: {s['mean_ms']:9.2f} +/- {s['std_ms']:7.2f} ms"
                  f"   [{s['min_ms']:.1f}, {s['max_ms']:.1f}]")
        print()

    base = next((r for r in rows if r["budget"] == 4096), rows[0])
    base_step = base["regions"]["step_total"]["mean_ms"]
    base_load = base["regions"]["dataload"]["mean_ms"]

    print("=" * 78)
    print(f"{'budget':>8s} | {'points fed':>10s} | {'dataload ms':>18s} | "
          f"{'step ms':>18s} | {'vs ' + str(base['budget']):>7s}")
    print("-" * 78)
    for row in rows:
        r = row["regions"]
        print(f"{row['budget']:>8,d} | {row['points_per_sample']:>10,.0f} | "
              f"{r['dataload']['mean_ms']:>10.2f} +/- {r['dataload']['std_ms']:<5.2f} | "
              f"{r['step_total']['mean_ms']:>10.2f} +/- {r['step_total']['std_ms']:<5.2f} | "
              f"{r['step_total']['mean_ms'] / base_step:>6.2f}x")
    print()
    print(f"Read the last column against the point ratio: {base['budget']} is the reference. "
          f"A ratio well below the\npoint ratio is the sublinearity -- the deep PTv3 stages "
          f"are limited by occupied VOXELS, not by\npoints, and the 1 cm grid caps those. "
          f"Data loading (reference {base_load:.2f} ms) barely moves\nbecause the budget is "
          f"applied after the full cloud has already been read and cropped.")

    if args.out:
        args.out.write_text(json.dumps({
            "run": training_args.run_name,
            "batch_size": batch_size,
            "workers": workers,
            "reps": args.reps,
            "warmup": args.warmup,
            "forward_only": args.forward_only,
            "rows": rows,
        }, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
