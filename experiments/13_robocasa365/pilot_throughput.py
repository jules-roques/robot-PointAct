"""Measure training throughput for {vlm, text_cache} x {2048, 4096, 8192} points.

The ablation budget rests on two assumptions that were estimated, not measured: that dropping
the VLM buys ~3x, and that cost scales linearly in the point count. Both set the size of a
~380 H100-hour grid, and this costs well under one GPU-hour to check.

Method: each configuration is timed at two step counts and the *marginal* rate is reported,

    steps/s = (n_long - n_short) / (runtime_long - runtime_short)

which cancels dataloader warm-up and CUDA autotuning. The aggregate rate HF prints would fold
all of that into a short run and flatter the cheap configurations.

`runtime` here is HF's `train_runtime` -- its own timer around the training loop -- not the
process wall clock, which is what this script used until 2026-09-01. Wall clock also carries
model loading and the final checkpoint write, and those are NOT stable enough to subtract: two
runs of the same configuration on the same H100 node differed by ~20 s, a third of the 60-step
window being measured. That made a strictly-more-expensive configuration finish its short run
FASTER than a cheaper one, and reported a 1.4x cost ratio as 2.0x. Wall clock is kept as a
fallback and printed alongside; a gap between the two columns means startup noise, and the
train_runtime column is the one to read.

    bash experiments/13_robocasa365/pilot_throughput.sh          # via the SLURM wrapper
    python experiments/13_robocasa365/pilot_throughput.py --help
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tempfile
from pathlib import Path

import yaml



def build_pilot_config(base: Path, out_dir: Path, run_root: Path, context: str, npoints: int,
                       steps: int, point_ca: bool = False) -> Path:
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
            "output_dir": str(run_root / f"{context}-n{npoints}-ca{int(point_ca)}-s{steps}"),
            # The point branch of every CABlock. Off in every arm so far, so the
            # cross-attention weights the checkpoints carry are trained by the 17 action
            # tokens alone; turning it on routes the whole cloud through those same
            # q/kv/proj and adds a norm + 4x MLP per block.
            "ptv3_apply_point_ca": bool(point_ca),
        },
        "data": {"lerobot_datasets": [{"max_npoints": npoints}]},
    }
    if context == "vlm":
        # The live-VLM arm must decode images and tokenise a prompt, which is precisely the
        # cost being measured. Clearing text_context_file re-enables both.
        document["data"]["lerobot_datasets"][0]["text_context_file"] = None

    path = out_dir / f"pilot-{context}-n{npoints}-ca{int(point_ca)}-s{steps}.yaml"
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


TRAIN_RUNTIME = re.compile(r"train_runtime[\"']?\s*:\s*[\"']?([0-9.]+)")


def run_one(config: Path, gpus: int, per_device_batch: int, accum: int, repo: Path,
            log_dir: Path) -> dict:
    """Launch one pilot run; return its wall-clock time AND HF's own training-loop time.

    Both, because the marginal subtraction across two step counts only cancels startup if
    startup is stable, and measured on H100 it is not: two cells of the SAME configuration
    differed by ~20 s of model loading and CUDA autotuning, which is a third of the 60-step
    difference being measured. That noise once made a strictly-more-expensive configuration
    look CHEAPER at 20 steps than a cheaper one, and inflated a 1.4x ratio to 2.0x.

    `train_runtime` is HF's own timer around the training loop, so it excludes startup, model
    loading and the final checkpoint write by construction rather than by subtraction. It does
    reach the log file under `report_to: []` -- an earlier version of this script looked for it
    on stdout, did not find it, and concluded it was unavailable. Prefer it; keep wall clock as
    the fallback and report both so a disagreement is visible rather than silent.
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

    blob = log_path.read_text(errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"{config.name} exited {result.returncode}:\n{first_error(blob)}")
    hits = TRAIN_RUNTIME.findall(blob)
    return {"runtime": elapsed, "train_runtime": float(hits[-1]) if hits else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path,
                        default=Path("experiments/13_robocasa365/runs/_base.yaml"))
    parser.add_argument("--contexts", nargs="+", default=["text_cache", "vlm"])
    parser.add_argument("--npoints", nargs="+", type=int, default=[2048, 4096, 8192])
    parser.add_argument("--point-ca", nargs="+", default=["false"],
                        choices=["false", "true"],
                        help="values of ptv3_apply_point_ca to sweep (default: false only)")
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
        cas = [c == "true" for c in args.point_ca]
        for context in args.contexts:
            for npoints in args.npoints:
              for point_ca in cas:
                # Keep the historical label shape when the CA axis is not being swept, so a
                # single-setting run stays comparable with the earlier pilot JSONs.
                label = f"{context}/n{npoints}"
                if len(cas) > 1:
                    label += f"/ca={str(point_ca).lower()}"
                print(f"[{label}]", flush=True)
                try:
                    timings = {}
                    for steps in (args.short_steps, args.long_steps):
                        config = build_pilot_config(
                            args.base, tmp_dir, run_root, context, npoints, steps, point_ca
                        )
                        timings[steps] = run_one(
                            config, args.gpus, args.per_device_batch, accum, repo, log_dir
                        )
                    delta_steps = args.long_steps - args.short_steps
                    wall = {k: v["runtime"] for k, v in timings.items()}
                    loop = {k: v["train_runtime"] for k, v in timings.items()}
                    delta_wall = wall[args.long_steps] - wall[args.short_steps]
                    have_loop = all(v is not None for v in loop.values())
                    delta_time = (loop[args.long_steps] - loop[args.short_steps]
                                  if have_loop else delta_wall)
                    source = "train_runtime" if have_loop else "wall_clock"
                    if delta_time <= 0:
                        raise RuntimeError(f"non-positive time delta ({delta_time:.2f}s)")
                    results[label] = {
                        "steps_per_second": delta_steps / delta_time,
                        "seconds_per_step": delta_time / delta_steps,
                        "source": source,
                        "seconds_per_step_wall": delta_wall / delta_steps,
                        "runtimes": wall,
                        "train_runtimes": loop,
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
    lines.append(f"{'config':>26s} | {'s/step':>8s} | {'wall':>8s} | {'50K steps':>10s} | {'GPU-h':>8s}")
    lines.append("-" * 74)

    hours = {}
    for label, entry in results.items():
        if "error" in entry:
            lines.append(f"{label:>26s} | {'FAILED':>8s} | {'--':>8s} | {'--':>10s} | {'--':>8s}")
            continue
        wall_h = entry["seconds_per_step"] * 50_000 / 3600
        hours[label] = wall_h
        w = entry.get("seconds_per_step_wall", entry["seconds_per_step"])
        lines.append(
            f"{label:>26s} | {entry['seconds_per_step']:8.3f} | {w:8.3f} | {wall_h:9.1f}h | "
            f"{wall_h * args.gpus:7.1f}h"
        )
        # A large gap means startup noise is comparable to the quantity being measured; the
        # train_runtime column is the one to trust, but the reader should know it happened.
        if abs(w - entry["seconds_per_step"]) > 0.15 * entry["seconds_per_step"]:
            lines.append(f"{'':>26s}   ^ wall-clock marginal disagrees by "
                         f"{100 * (w / entry['seconds_per_step'] - 1):+.0f}% "
                         f"(startup noise; using {entry.get('source', 'train_runtime')})")

    lines.append("")
    for npoints in args.npoints:
        vlm, cached = hours.get(f"vlm/n{npoints}"), hours.get(f"text_cache/n{npoints}")
        if vlm and cached:
            lines.append(f"  dropping the VLM at n={npoints}: {vlm / cached:.1f}x faster "
                         f"(the budget assumed 3x)")

    for npoints in args.npoints:
        for context in args.contexts:
            off = hours.get(f"{context}/n{npoints}/ca=false")
            on = hours.get(f"{context}/n{npoints}/ca=true")
            if off and on:
                lines.append(f"  ptv3_apply_point_ca at {context}/n={npoints}: "
                             f"{on / off:.2f}x  (+{100 * (on / off - 1):.0f}% wall clock)")

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
