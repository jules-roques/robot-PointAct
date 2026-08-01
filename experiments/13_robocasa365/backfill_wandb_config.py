"""Push a run's ablation coordinates into its existing W&B run config.

W&B config is written once at wandb.init(), so runs already in flight cannot pick up renamed
or added columns. The stage-A runs were launched with the coordinates under `exp_*` names,
which are not what you reach for in the group-by dropdown -- and `sampling_strategy` there
resolves to `train_sampling_strategy` (the dataloader's sampler, always "random"), not the
ablation arm. This rewrites them in place via the public API.

Values come from the run's own run_config.resolved.yaml, so this cannot invent a coordinate
that disagrees with what trained. Safe to re-run.

    python experiments/13_robocasa365/backfill_wandb_config.py \
        $SCRATCH/PointAct_exprs/robocasa365/ablation/od-*
"""

import argparse
import os
import sys
from pathlib import Path

import wandb
import yaml

#: meta key in the run yaml -> W&B config column. Matches META_TO_EXP_FIELD in
#: pointact/train/run_config.py; keep the two in step.
META_TO_COLUMN = {
    "task": "task_name",
    "sampling": "sampling_strategy",
    "npoints": "cloud_size",
    "seed": "arm_seed",
    "stage": "stage",
    "context": "context_source",
}
STALE_COLUMNS = ("exp_task", "exp_sampling", "exp_npoints", "exp_context", "exp_seed", "exp_stage")


def backfill(run_dir: Path, project: str, entity: str, drop_stale: bool) -> bool:
    id_file, resolved = run_dir / "wandb_run_id.txt", run_dir / "run_config.resolved.yaml"
    if not id_file.exists() or not resolved.exists():
        print(f"  skip {run_dir.name}: missing wandb_run_id.txt or run_config.resolved.yaml")
        return False

    meta = yaml.safe_load(resolved.read_text()).get("meta") or {}
    updates = {col: meta[key] for key, col in META_TO_COLUMN.items() if meta.get(key) is not None}
    if not updates:
        print(f"  skip {run_dir.name}: no meta block")
        return False

    run = wandb.Api().run(f"{entity}/{project}/{id_file.read_text().strip()}")
    run.config.update(updates)
    if drop_stale:
        for key in STALE_COLUMNS:
            run.config.pop(key, None)
    run.update()
    print(f"  {run_dir.name}: " + ", ".join(f"{k}={v}" for k, v in sorted(updates.items())))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "pointact-robocasa365"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "diffusion4robots"))
    parser.add_argument("--keep-stale", action="store_true",
                        help="Leave the old exp_* columns in place instead of removing them.")
    args = parser.parse_args()

    if os.environ.get("WANDB_MODE") == "offline":
        sys.exit("WANDB_MODE=offline: this needs the API. Run it from a login node.")

    dirs = [d for d in args.run_dirs if d.is_dir()]
    done = sum(backfill(d, args.project, args.entity, not args.keep_stale) for d in dirs)
    print(f"\nupdated {done}/{len(dirs)} run(s)")


if __name__ == "__main__":
    main()
