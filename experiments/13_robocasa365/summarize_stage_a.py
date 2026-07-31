"""Assemble the stage-A point-count x sampling table and print it for the gate email.

Stage B trains at a single point count, and which one is a judgement call the user makes --
not something this pipeline should decide. So stage A ends with one summary job that collects
every per-trial result file, prints this table, and lets SLURM mail it (see gate_stage_a.slurm).

Reads the `per_trial_seed*_n*.json` files written by run_robocasa365_client.py. Multiple seeds
for the same (run, checkpoint) are pooled: each is a disjoint scene stream, so they pool as
independent trials rather than being averaged.

    python experiments/13_robocasa365/summarize_stage_a.py \
        --exprs-dir $SCRATCH/PointAct_exprs/robocasa365/ablation
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

#: Run directory names look like od-eef-n4096-s0.
RUN_RE = re.compile(r"^(?P<task>[a-z]+)-(?P<sampling>uniform|eef|anchor)-n(?P<npoints>\d+)-s(?P<seed>\d+)$")


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval.

    Not the textbook Wald interval (p +/- z*sqrt(p(1-p)/n)): that under-covers badly at the
    sample sizes here and can produce bounds outside [0, 1].
    """
    if trials == 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def collect(exprs_dir: Path) -> dict:
    """(sampling, npoints, ckpt_step) -> pooled successes/trials across seeds."""
    pooled = defaultdict(lambda: {"successes": 0, "trials": 0, "seeds": set()})

    for run_dir in sorted(exprs_dir.iterdir()):
        match = RUN_RE.match(run_dir.name)
        if not match or not (run_dir / "results").is_dir():
            continue
        for ckpt_dir in sorted((run_dir / "results").iterdir()):
            step = ckpt_dir.name.removeprefix("checkpoint-")
            for result_file in sorted(ckpt_dir.glob("per_trial_seed*_n*.json")):
                record = json.loads(result_file.read_text())
                key = (match["sampling"], int(match["npoints"]), step)
                pooled[key]["successes"] += record["successes"]
                pooled[key]["trials"] += record["num_trials"]
                pooled[key]["seeds"].add(record["seed"])
    return pooled


def render(pooled: dict, final_step: str) -> str:
    """Point-count x sampling table at the final checkpoint, plus the full curve below."""
    lines = []
    npoints_values = sorted({key[1] for key in pooled})
    samplings = [s for s in ("uniform", "eef", "anchor") if any(k[0] == s for k in pooled)]

    if not pooled:
        return "No stage-A results found yet."

    lines.append(f"Stage A -- OpenDrawer, success rate at checkpoint {final_step} [Wilson 95% CI]")
    lines.append("")
    header = f"{'points':>8s} | " + " | ".join(f"{s:^26s}" for s in samplings)
    lines.append(header)
    lines.append("-" * len(header))
    for npoints in npoints_values:
        cells = []
        for sampling in samplings:
            entry = pooled.get((sampling, npoints, final_step))
            if entry is None:
                cells.append(f"{'--':^26s}")
                continue
            rate = entry["successes"] / max(entry["trials"], 1)
            low, high = wilson_interval(entry["successes"], entry["trials"])
            cells.append(f"{rate:6.1%} [{low:.1%},{high:.1%}] n={entry['trials']:<3d}".center(26))
        lines.append(f"{npoints:8d} | " + " | ".join(cells))

    lines.append("")
    lines.append("Training-duration curve (success rate by checkpoint):")
    steps = sorted({key[2] for key in pooled}, key=lambda s: int(s) if s.isdigit() else 1 << 30)
    lines.append(f"{'arm':>20s} | " + " | ".join(f"{s:>8s}" for s in steps))
    for npoints in npoints_values:
        for sampling in samplings:
            cells = []
            for step in steps:
                entry = pooled.get((sampling, npoints, step))
                cells.append(
                    f"{entry['successes'] / max(entry['trials'], 1):8.1%}" if entry else f"{'--':>8s}"
                )
            lines.append(f"{sampling:>12s}/n{npoints:<6d} | " + " | ".join(cells))

    lines.append("")
    lines.append("Pick the point count for stage B, then regenerate its run yamls:")
    lines.append("  python experiments/13_robocasa365/runs/generate_stage_b.py --npoints <N>")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exprs-dir", type=Path, required=True)
    parser.add_argument(
        "--final-step",
        default="50000",
        help="Checkpoint the headline table is taken at (the others form the curve).",
    )
    parser.add_argument("--out", type=Path, default=None, help="Also write the table here.")
    args = parser.parse_args()

    table = render(collect(args.exprs_dir), args.final_step)
    print(table)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
