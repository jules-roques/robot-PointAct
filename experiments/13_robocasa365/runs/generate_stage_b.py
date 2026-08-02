"""Regenerate the stage-B run yamls at the point count chosen after stage A.

Stage B trains the two new tasks at a single point count, picked by hand from the stage-A
table (see summarize_stage_a.py and gate_stage_a.slurm). The checked-in stage-B files use 4096
as a placeholder so the grid is complete and reviewable before stage A finishes.

    python experiments/13_robocasa365/runs/generate_stage_b.py --npoints 8192
"""

import argparse
from pathlib import Path

TASKS = {
    "PickPlaceCounterToStove": "ppcs",
    "TurnOnMicrowave": "tom",
}

SAMPLING_BLOCK = {
    "uniform": "",
    "eef": """      eef_sampling: true
      eef_sampling_sigma: 0.08
      eef_sampling_floor: 0.05
""",
}

HEAD = {
    "uniform": "Uniform point subsample: the baseline draw.",
    "eef": (
        "EEF-density sampling: weight-proportional draw under a Gaussian-with-floor\n"
        "# density centred on the frame's own end-effector position. No cache needed."
    ),
}

TEMPLATE = """# {task} / {sampling} / {npoints} points -- stage B.
# {head}
#
# Point count chosen from the stage-A sweep on OpenDrawer; regenerate with
# generate_stage_b.py --npoints <N> if that choice changes.
extends: _base.yaml

meta:
  task: {task}
  sampling: {sampling}
  npoints: {npoints}
  context: text_cache
  seed: {seed}
  stage: "Stage 2: Task transfer"

data:
  lerobot_datasets:
    - repo_id: {task}
      state_action_norm_file: robot_data/robocasa365/lerobot_point_lmdb/{task}/robot_state_action_stats/rot6d.json
      text_context_file: text_context/qwen2.5-vl-3b.pt
      max_npoints: {npoints}
{sampling_block}"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npoints", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(__file__).parent
    # Stage-B files are regenerated wholesale, so drop the previous point count's copies
    # rather than leaving two contradictory grids side by side.
    for stale in out_dir.glob("*.yaml"):
        if stale.name.startswith(("ppcs-", "tom-")):
            stale.unlink()

    for task, abbrev in TASKS.items():
        for sampling in ("uniform", "eef"):
            name = f"{abbrev}-{sampling}-n{args.npoints}-s{args.seed}"
            (out_dir / f"{name}.yaml").write_text(
                TEMPLATE.format(
                    task=task,
                    sampling=sampling,
                    npoints=args.npoints,
                    seed=args.seed,
                    head=HEAD[sampling],
                    sampling_block=SAMPLING_BLOCK[sampling],
                )
            )
            print(f"wrote {name}.yaml")


if __name__ == "__main__":
    main()
