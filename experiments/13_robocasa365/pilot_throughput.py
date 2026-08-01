"""Measure training throughput for {vlm, text_cache} x {2048, 4096, 8192} points.

The ablation budget rests on two assumptions that were estimated, not measured: that dropping
the VLM buys ~3x, and that cost scales linearly in the point count. Both set the size of a
~380 H100-hour grid, and this costs well under one GPU-hour to check.

Method: each configuration is timed at two step counts and the *marginal* rate is reported,

    steps/s = (n_long - n_short) / (runtime_long - runtime_short)

which cancels process startup, model loading, dataloader warm-up and CUDA autotuning. The
aggregate rate HF prints would fold all of that into a short run and flatter the cheap
configurations.

    bash experiments/13_robocasa365/pilot_throughput.sh          # via the SLURM wrapper
    python experiments/13_robocasa365/pilot_throughput.py --help
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import tempfile
from pathlib import Path

import yaml



def build_pilot_config(base: Path, out_dir: Path, run_root: Path, context: str, npoints: int,
                       steps: int) -> Path:
    """A run yaml for one pilot point: no checkpoints, no W&B, a handful of steps."""
    document = {
        "extends": str(base.resolve()),
        "meta": {
            "task": "OpenDrawer",
            "sampling": "uniform",
            "npoints": npoints,
            "context": context,
            "seed": 0,
            "stage": "pilot",
        },
        "train": {
            "max_steps": steps,
            "context_source": context,
            "save_strategy": "no",
            "report_to": [],
            "logging_steps": max(1, steps // 4),
            # On $SCRATCH, never the node's /tmp: train.py always writes a final checkpoint,
            # and a vlm-arm checkpoint is ~7GB, which fills /tmp and kills the run at the very
            # end -- after all the timed work is done.
            "output_dir": str(run_root / f"{context}-n{npoints}-s{steps}"),
        },
        "data": {"lerobot_datasets": [{"max_npoints": npoints}]},
    }
    if context == "vlm":
        # The live-VLM arm must decode images and tokenise a prompt, which is precisely the
        # cost being measured. Clearing text_context_file re-enables both.
        document["data"]["lerobot_datasets"][0]["text_context_file"] = None

    path = out_dir / f"pilot-{context}-n{npoints}-s{steps}.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return path


def first_error(blob: str, context: int = 12) -> str:
    """The child process's own traceback, not torchrun's wrapper.

    A failing rank makes accelerate print a ChildFailedError banner that says only "exitcode
    1"; tailing the output shows that banner and buries the actual exception hundreds of lines
    up. This finds the real one.
    """
    lines = blob.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("Traceback (most recent call")]
    bounds = list(zip(starts, starts[1:] + [len(lines)]))
    for start, stop in bounds:
        block = lines[start:stop]
        # Skip torchrun's own traceback -- it only ever reports "exitcode 1".
        if any("distributed/launcher" in line or "CalledProcessError" in line for line in block):
            continue
        # A traceback ends at its first unindented line: the exception itself.
        end = next(
            (start + i for i, line in enumerate(block) if i and line and not line[0].isspace()),
            stop - 1,
        )
        return "\n".join(lines[start : end + 1])
    return "\n".join(lines[-context:]) or "(no output)"


def run_one(config: Path, gpus: int, per_device_batch: int, accum: int, repo: Path,
            log_dir: Path) -> dict:
    """Launch one pilot run; return its wall-clock time.

    Timed end to end rather than scraped from HF's train_* metrics: those are logged through
    the callback stack and simply do not appear on stdout under `report_to: []`, which is how
    an earlier version of this script recorded six successful runs as failures. Wall clock is
    also the honest quantity -- the marginal subtraction across two step counts removes
    startup, model loading and the final checkpoint write, whatever HF chooses to print.
    """
    command = ["accelerate", "launch"]
    command += ["--multi_gpu", f"--num_processes={gpus}", "--num_machines=1", "--machine_rank=0"] \
        if gpus > 1 else ["--num_processes=1"]
    command += [
        "scripts/train.py", str(config),
        "--gradient-accumulation-steps", str(accum),
        "--per-device-train-batch-size", str(per_device_batch),
    ]

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{config.stem}.log"
    print(f"    $ train.py {config.name}  (log: {log_path})", flush=True)

    started = time.monotonic()
    with log_path.open("w") as log:
        # Child output goes to a file rather than a pipe we discard: swallowing it is what made
        # the previous failure undiagnosable from the job log.
        result = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, text=True)
    elapsed = time.monotonic() - started

    if result.returncode != 0:
        blob = log_path.read_text(errors="replace")
        raise RuntimeError(f"{config.name} exited {result.returncode}:\n{first_error(blob)}")
    return {"runtime": elapsed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path,
                        default=Path("experiments/13_robocasa365/runs/_base.yaml"))
    parser.add_argument("--contexts", nargs="+", default=["text_cache", "vlm"])
    parser.add_argument("--npoints", nargs="+", type=int, default=[2048, 4096, 8192])
    # Wide enough that the difference clears run-to-run startup noise. At 20/60 the
    # cheapest cell (n2048) produced a *negative* delta -- ~20s of node-to-node
    # variation swamped 40 steps of a sub-0.5s/step configuration.
    parser.add_argument("--short-steps", type=int, default=40)
    parser.add_argument("--long-steps", type=int, default=160)
    parser.add_argument("--gpus", type=int, default=int(os.environ.get("SLURM_GPUS_ON_NODE", 4)))
    parser.add_argument("--per-device-batch", type=int, default=32)
    parser.add_argument("--effective-batch", type=int, default=128)
    parser.add_argument("--out", type=Path, default=Path("pilot_throughput.json"))
    args = parser.parse_args()

    repo = Path.cwd()
    accum = args.effective_batch // (args.gpus * args.per_device_batch)
    if accum * args.gpus * args.per_device_batch != args.effective_batch:
        sys.exit(f"{args.gpus} x {args.per_device_batch} does not divide {args.effective_batch}")
    print(f"effective batch {args.effective_batch} = {args.gpus} GPU x "
          f"{args.per_device_batch} x accum {accum}\n")

    results = {}
    log_dir = args.out.parent / "pilot_logs"
    run_root = args.out.parent / "pilot_runs"
    with tempfile.TemporaryDirectory(prefix="pilot-") as tmp:
        tmp_dir = Path(tmp)
        for context in args.contexts:
            for npoints in args.npoints:
                label = f"{context}/n{npoints}"
                print(f"[{label}]", flush=True)
                try:
                    timings = {}
                    for steps in (args.short_steps, args.long_steps):
                        config = build_pilot_config(
                            args.base, tmp_dir, run_root, context, npoints, steps
                        )
                        timings[steps] = run_one(
                            config, args.gpus, args.per_device_batch, accum, repo, log_dir
                        )["runtime"]
                    delta_steps = args.long_steps - args.short_steps
                    delta_time = timings[args.long_steps] - timings[args.short_steps]
                    if delta_time <= 0:
                        raise RuntimeError(f"non-positive time delta ({delta_time:.2f}s)")
                    results[label] = {
                        "steps_per_second": delta_steps / delta_time,
                        "seconds_per_step": delta_time / delta_steps,
                        "runtimes": timings,
                    }
                    print(f"    -> {results[label]['seconds_per_step']:.3f} s/step\n", flush=True)
                except Exception as exc:  # noqa: BLE001 - one bad cell must not lose the rest
                    print(f"    !! FAILED: {exc}\n", flush=True)
                    results[label] = {"error": str(exc)}
                finally:
                    # Reclaim the checkpoints: 12 runs at up to ~7GB each otherwise.
                    shutil.rmtree(run_root, ignore_errors=True)

    print(report(results, args))
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


def report(results: dict, args) -> str:
    """Seconds per step, projected 50K-step wall clock, and the two ratios that matter."""
    lines = ["", "=" * 74, "Throughput pilot (marginal rate, startup and warm-up excluded)", "=" * 74]
    lines.append(f"{'config':>18s} | {'s/step':>8s} | {'50K steps':>10s} | {'GPU-hours':>10s}")
    lines.append("-" * 74)

    hours = {}
    for label, entry in results.items():
        if "error" in entry:
            lines.append(f"{label:>18s} | {'FAILED':>8s} | {'--':>10s} | {'--':>10s}")
            continue
        wall_h = entry["seconds_per_step"] * 50_000 / 3600
        hours[label] = wall_h
        lines.append(
            f"{label:>18s} | {entry['seconds_per_step']:8.3f} | {wall_h:9.1f}h | "
            f"{wall_h * args.gpus:9.1f}h"
        )

    lines.append("")
    for npoints in args.npoints:
        vlm, cached = hours.get(f"vlm/n{npoints}"), hours.get(f"text_cache/n{npoints}")
        if vlm and cached:
            lines.append(f"  dropping the VLM at n={npoints}: {vlm / cached:.1f}x faster "
                         f"(the budget assumed 3x)")

    base = hours.get("text_cache/n4096")
    if base:
        scaling = [
            f"n{n}={hours[f'text_cache/n{n}'] / base:.2f}x"
            for n in args.npoints if f"text_cache/n{n}" in hours
        ]
        lines.append(f"  point-count scaling vs n4096 (text_cache): {', '.join(scaling)} "
                     f"(the budget assumed linear: 0.5x / 1x / 2x)")

    stage_a = sum(
        hours[f"text_cache/n{n}"] * args.gpus * 3  # three sampling arms per point count
        for n in args.npoints if f"text_cache/n{n}" in hours
    )
    if stage_a:
        lines.append("")
        lines.append(f"  => stage A (9 runs) projects to {stage_a:.0f} H100-hours "
                     f"(the plan budgeted 272)")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
