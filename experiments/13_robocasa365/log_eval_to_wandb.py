"""Log a run's eval results into that run's OWN W&B run, under an `eval/` section.

Everything for one arm lives in one W&B run: training curves, the sampling animation, and the
eval results. Possible because each run pins its id in output_dir/wandb_run_id.txt, so
`wandb.init(id=..., resume="allow")` reopens it rather than creating a second run that has to
be cross-referenced. W&B derives panel sections from the key prefix, so `eval/*` lands in a
section called "eval".

This runs after the eval arrays finish rather than from inside them: the arrays run up to four
tasks at once per arm, and concurrent writers cannot share a run -- least of all an offline
one, which is mandatory on Jean Zay. The jobs write files; this owns the W&B side. Re-running
is safe: results are re-read and re-logged from scratch.

Needs outbound network (a Jean Zay login node, not a compute node) and the training run must
already be synced, or there is nothing to resume into.

    python experiments/13_robocasa365/log_eval_to_wandb.py \
        $SCRATCH/PointAct_exprs/robocasa365/ablation/od-eef-n4096-s0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import wandb

sys.path.insert(0, str(Path(__file__).parent))
from summarize_stage_a import wilson_interval  # noqa: E402


def load_results(run_dir: Path) -> list[dict]:
    """One pooled record per evaluated checkpoint, ordered by step.

    Several seeds at the same checkpoint are pooled rather than averaged: each reruns a
    disjoint scene stream, so they combine as independent trials.
    """
    by_step: dict[int, dict] = {}
    for ckpt_dir in sorted((run_dir / "results").glob("checkpoint-*")):
        try:
            # "final-48750" also occurs; take the trailing integer so it sorts with the rest.
            step = int(ckpt_dir.name.removeprefix("checkpoint-").rsplit("-", 1)[-1])
        except ValueError:
            # e.g. checkpoint-50000-viz, written by a rollout-only pass. Its trials must never
            # be pooled into the real result -- a handful of viz episodes would shift the
            # headline number. Only its figures are picked up, by rollout_media below.
            continue
        entry = by_step.setdefault(
            step, {"ckpt_step": step, "successes": 0, "trials": 0, "dir": ckpt_dir}
        )
        for result_file in sorted(ckpt_dir.glob("per_trial_seed*_n*.json")):
            record = json.loads(result_file.read_text())
            entry["successes"] += record["successes"]
            entry["trials"] += record["num_trials"]
    return [by_step[step] for step in sorted(by_step) if by_step[step]["trials"]]


def rollout_media(ckpt_dir: Path) -> dict:
    """The success/failure rollout animations for this checkpoint, if any.

    Looks in the checkpoint's own directory and in a `-viz` sibling, so figures produced by a
    separate rollout-only pass are picked up without their trials polluting the pooled rate.
    """
    search = [ckpt_dir, ckpt_dir.with_name(ckpt_dir.name + "-viz")]
    media = {}
    for outcome in ("success", "failure"):
        matches = sorted(m for d in search if d.is_dir() for m in d.glob(f"rollout_{outcome}_*.html"))
        if matches:
            media[f"eval/rollout_{outcome}"] = wandb.Html(
                matches[0].read_text(encoding="utf-8"), inject=False
            )
    return media


def log_run(run_dir: Path, project: str, entity: str) -> bool:
    run_id_file = run_dir / "wandb_run_id.txt"
    if not run_id_file.exists():
        print(f"  skip {run_dir.name}: no wandb_run_id.txt")
        return False
    results = load_results(run_dir)
    if not results:
        print(f"  skip {run_dir.name}: no eval results under results/")
        return False

    run = wandb.init(
        project=project, entity=entity, id=run_id_file.read_text().strip(), resume="allow",
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
    )
    # Plot against the checkpoint the number came from, not W&B's own monotonic step: array
    # tasks finish in no particular order, and the training run's step axis already means
    # something else.
    wandb.define_metric("eval/ckpt_step")
    wandb.define_metric("eval/*", step_metric="eval/ckpt_step")

    for entry in results:
        low, high = wilson_interval(entry["successes"], entry["trials"])
        rate = entry["successes"] / max(entry["trials"], 1)
        payload = {
            "eval/ckpt_step": entry["ckpt_step"],
            # Also emitted on the training axis. Eval rows otherwise carry no
            # train/global_step, so any panel using the workspace default x-axis reports
            # "no data" -- and semantically the eval of checkpoint N *is* the policy at
            # global step N, so this puts it exactly where it belongs against the loss curve.
            "train/global_step": entry["ckpt_step"],
            "eval/success_rate": rate,
            "eval/successes": entry["successes"],
            "eval/trials": entry["trials"],
            "eval/wilson_lo": low,
            "eval/wilson_hi": high,
        }
        payload.update(rollout_media(entry["dir"]))
        run.log(payload)
        print(f"  {run_dir.name} @{entry['ckpt_step']}: "
              f"{entry['successes']}/{entry['trials']} = {rate:.1%} [{low:.1%}, {high:.1%}]")

    # Summary columns so the runs table sorts by final success directly. update() rather than
    # item assignment: assigning on a resumed run did not persist.
    final = results[-1]
    summary = {
        "eval/final_success_rate": final["successes"] / max(final["trials"], 1),
        "eval/final_ckpt_step": final["ckpt_step"],
        "eval/final_trials": final["trials"],
    }
    summary.update(duration_summary(run, run_dir))
    run.summary.update(summary)
    run.finish()
    return True


def duration_summary(run, run_dir: Path) -> dict:
    """Training cost, in the units anyone actually asks about.

    HF records train_runtime in seconds and nothing else; wall-clock hours, GPU-hours and the
    epoch equivalent are what the budget and the steps-vs-epochs question are argued in.
    """
    runtime = run.summary.get("train_runtime")
    if not runtime:
        return {}
    out = {"duration/train_runtime_h": runtime / 3600.0}

    args_file = run_dir / "training_args.json"
    if args_file.exists():
        args = json.loads(args_file.read_text())
        steps = args.get("max_steps") or 0
        world = args.get("world_size") or 4
        if steps:
            out["duration/seconds_per_step"] = runtime / steps
            out["duration/gpu_hours"] = runtime / 3600.0 * world
        # Frames come from the dataset the run actually used, so the epoch equivalent is real
        # rather than assumed -- 50K steps is ~51 epochs on OpenDrawer but ~88 on TurnOnMicrowave.
        info = run_dir / "data_config.yaml"
        if info.exists() and steps:
            import yaml
            entry = yaml.safe_load(info.read_text())["lerobot_datasets"][0]
            meta = Path(entry["root"]) / entry["repo_id"] / "meta" / "info.json"
            candidates = [meta, Path(os.path.expandvars("$SCRATCH/datasets")) / meta]
            for path in candidates:
                if path.exists():
                    frames = json.loads(path.read_text())["total_frames"]
                    eff = args.get("effective_batch") or 128
                    out["duration/steps_per_epoch"] = frames / eff
                    out["duration/epochs_equivalent"] = steps / (frames / eff)
                    break
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "pointact-robocasa365"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "diffusion4robots"))
    args = parser.parse_args()

    if os.environ.get("WANDB_MODE") == "offline":
        sys.exit("WANDB_MODE=offline: this needs the API. Run it from a login node.")
    # Keep wandb's scratch off $WORK, whose inode quota is the tight one on Jean Zay.
    os.environ.setdefault("WANDB_DIR", os.path.expandvars("$SCRATCH/wandb-attach"))
    Path(os.environ["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)

    done = sum(log_run(d, args.project, args.entity) for d in args.run_dirs)
    print(f"\nlogged {done}/{len(args.run_dirs)} run(s)")


if __name__ == "__main__":
    main()
