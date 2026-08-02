"""Replay a run's eval results into a single W&B run, as a success-vs-checkpoint curve.

Why a separate script rather than logging from the eval job: the eval jobs run as an array,
one per checkpoint, so up to four of them are alive at once -- and concurrent writers cannot
share one W&B run. Both clusters also force WANDB_MODE=offline (mandatory on Jean Zay, whose
compute nodes have no outbound network), and two processes cannot write one offline run and be
merged at sync time; W&B's "shared mode" needs the online service.

So the eval jobs write files, and this owns the run. It is idempotent: rerun it whenever new
results land and it resumes the same run id and re-logs the whole curve.

The eval run carries the same `group` as its training run, so the W&B workspace shows the
success curve and the training loss side by side when grouped by arm. It stays a *separate*
run, which also keeps the A100's system metrics out of the H100 training run's.

    python experiments/13_robocasa365/log_eval_to_wandb.py \
        --run-dir $SCRATCH/PointAct_exprs/robocasa365/ablation/od-eef-n4096-s0
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import wandb

sys.path.insert(0, str(Path(__file__).parent))
from summarize_stage_a import wilson_interval  # noqa: E402


def load_results(run_dir: Path) -> list[dict]:
    """One pooled record per evaluated checkpoint, ordered by step.

    Several seeds for the same checkpoint are pooled rather than averaged: each is a disjoint
    scene stream, so they combine as independent trials.
    """
    by_step: dict[int, dict] = {}
    for ckpt_dir in sorted((run_dir / "results").glob("checkpoint-*")):
        raw_step = ckpt_dir.name.removeprefix("checkpoint-")
        # "final-48750" also appears; take the trailing integer so it sorts with the rest.
        step = int(raw_step.rsplit("-", 1)[-1])
        entry = by_step.setdefault(step, {"ckpt_step": step, "successes": 0, "trials": 0, "seeds": []})
        for result_file in sorted(ckpt_dir.glob("per_trial_seed*_n*.json")):
            record = json.loads(result_file.read_text())
            entry["successes"] += record["successes"]
            entry["trials"] += record["num_trials"]
            entry["seeds"].append(record["seed"])
    return [by_step[step] for step in sorted(by_step)]


def rollout_html(run_dir: Path, step: int) -> dict[str, "wandb.Html"]:
    """The two point-cloud rollout animations captured for this checkpoint, if any."""
    media = {}
    for outcome in ("success", "failure"):
        for path in sorted((run_dir / "results").glob(f"checkpoint-*{step}/rollout_{outcome}_*.html")):
            media[f"eval/rollout_{outcome}"] = wandb.Html(path.read_text(), inject=False)
            break  # one per outcome per checkpoint
    return media


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="A training run's output_dir.")
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "pointact-robocasa365"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "diffusion4robots"))
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    results = load_results(run_dir)
    if not results:
        print(f"no eval results under {run_dir}/results -- nothing to log")
        return

    training_args = json.loads((run_dir / "training_args.json").read_text())
    config = {key: value for key, value in training_args.items() if key.startswith("exp_")}
    config["run"] = run_dir.name

    group = "/".join(
        str(config[key]) for key in ("exp_task", "exp_sampling", "exp_npoints") if config.get(key)
    )

    # Deterministic id derived from the training run's, so reruns resume rather than fork.
    run_id = "eval-" + hashlib.sha1(str(run_dir).encode()).hexdigest()[:12]
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        id=run_id,
        resume="allow",
        name=f"{run_dir.name}-eval",
        group=group,
        job_type="eval",
        tags=[str(v) for v in config.values() if isinstance(v, (str, int))],
        config=config,
    )

    # Success is plotted against the checkpoint it came from, not against wandb's own step
    # counter -- results arrive out of order as array tasks finish.
    wandb.define_metric("ckpt_step")
    wandb.define_metric("eval/*", step_metric="ckpt_step")

    for entry in results:
        low, high = wilson_interval(entry["successes"], entry["trials"])
        payload = {
            "ckpt_step": entry["ckpt_step"],
            "eval/success_rate": entry["successes"] / max(entry["trials"], 1),
            "eval/successes": entry["successes"],
            "eval/trials": entry["trials"],
            "eval/wilson_lo": low,
            "eval/wilson_hi": high,
        }
        payload.update(rollout_html(run_dir, entry["ckpt_step"]))
        wandb.log(payload)
        print(
            f"  step {entry['ckpt_step']:>6}: "
            f"{entry['successes']}/{entry['trials']} = "
            f"{entry['successes'] / max(entry['trials'], 1):.1%} [{low:.1%}, {high:.1%}]"
        )

    # Summary columns so the runs table can be sorted by final success directly.
    final = results[-1]
    run.summary["eval/final_success_rate"] = final["successes"] / max(final["trials"], 1)
    run.summary["eval/final_ckpt_step"] = final["ckpt_step"]
    run.summary["eval/final_trials"] = final["trials"]
    run.finish()


if __name__ == "__main__":
    main()
