"""Generate the stage-5 run yamls: five tasks x four sampling arms, one point count.

Stage 5 is the wide grid the earlier stages were narrowing towards. It fixes everything the
ablations settled -- 8192 points, no images, cached text context -- and varies only the task
and the sampler, so the four arms are comparable within a task and the five tasks are
comparable within an arm.

    python experiments/13_robocasa365/runs/generate_stage5.py            # first three tasks
    python experiments/13_robocasa365/runs/generate_stage5.py --all      # once the new data lands

Why these coordinates:

* **8192 points.** Stage 1 measured the point-count axis on OpenDrawer and the oracle arm was
  the one that moved: 71% at 4096 against 84% at 8192, while uniform and eef were flat. The
  budget is only worth spending if the sampler knows where to spend it, which is exactly what
  this grid is about.
* **30K steps, final checkpoint only.** Stage 1-3 curves at 10/20/30/40/50K show every arm
  effectively converged by 30K on every task; the remaining 20K bought noise. Trading it for
  rollouts buys precision where it is actually short -- 500 trials is a Wilson half-width of
  ~4.4pp near 50%, against ~10pp at the 100 this campaign has been reporting.
* **Four arms.** uniform (baseline), eef (the deployable prior), oracle (the upper bound on
  any sampler), molmo-bestview (the upper bound on the *pointer*). The two privileged arms
  now share one ground-truth definition -- see pointact/roi_sampling/geom_gt.py.
* **One pointing query.** Stage 3 ran PickPlaceCounterToStove with the object alone and with
  object+pan; the second query was clearly worse (26% vs 45%), so it is dropped here.

Intermediate checkpoints are still written every 5K: they cost nothing, they let a requeued
job resume, and they leave the option of adding a curve later without retraining.
"""

import argparse
from pathlib import Path

#: task -> filename abbreviation. The first three have data; the last two are downloaded and
#: converted by the stage-5 data-prep chain and are only emitted under --all.
TASKS = {
    "OpenDrawer": "od",
    "TurnOnMicrowave": "tom",
    "PickPlaceCounterToStove": "ppcs",
}
NEW_TASKS = {
    "CloseBlenderLid": "blender",
    "CoffeeSetupMug": "mug",
}

STAGE = "Stage 5: Five tasks x four samplers"

HEAD = {
    "uniform": "Uniform point subsample: the baseline draw, and the control every other arm\n"
               "# in this grid is read against.",
    "eef": "EEF-density sampling: a Gaussian-with-floor density centred on the frame's own\n"
           "# end-effector position. No cache and no privileged information -- the deployable arm.",
    "oracle": "GT-oracle sampling: the same density centred on where the simulator says the\n"
              "# target actually is. Privileged; an upper bound on what any sampler could buy,\n"
              "# not a deployable policy.",
    "molmo": "MolmoPoint best-view anchor: the frozen pointer answers in all three views, each\n"
             "# answer is lifted to 3D, and the one nearest the ground truth is kept.\n"
             "# Privileged in its SELECTION only -- the anchor itself is a real detection, so\n"
             "# this bounds the pointer rather than the sampler. Against the eef arm it says\n"
             "# whether pointing can match a gripper prior; against the oracle arm, how much\n"
             "# of the remaining gap is the detector.",
}

BLOCK = {
    "uniform": "",
    "eef": """      eef_sampling: true
      eef_sampling_sigma: 0.08
      eef_sampling_floor: 0.05
""",
    "oracle": """      oracle_sampling: true
      # Geometry, not rendered segmentation labels: a label centroid averages only the
      # VISIBLE surface, so it moves with the camera and is undefined once the target is
      # occluded. geom_gt.ORACLE_TARGET picks the set for this task.
      oracle_gt: geom
      oracle_gt_npz: roi_meta/target_positions.npz
      oracle_gt_set: {gt_set}
      oracle_sampling_sigma: 0.08
      oracle_sampling_floor: 0.05
""",
    "molmo": """      molmo_sampling: true
      molmo_anchor_dirname: points_3views_molmo_bestgt
      # Query 0 = the manipulated object (the control being pressed, on TurnOnMicrowave).
      # The destination query stage 3 also tried is dropped: it lost, clearly.
      molmo_anchor_ids: [0]
      molmo_view_select: closest_gt
      molmo_sampling_sigma: 0.08
      molmo_sampling_floor: 0.05
      # A frame where no view lifted a detection gets the same density centred on the
      # gripper instead of a uniform draw. Never missing, and a strong sampler in its own
      # right -- on PickPlaceCounterToStove uniform scores 4% where eef scores 51%, so the
      # fallback decides those frames more than the anchor does.
      molmo_fallback: eef
""",
}

TEMPLATE = """# {task} / {sampling} / {npoints} points, {steps_k}K steps -- stage 5.
# {head}
extends: _base.yaml

meta:
  task: {task}
  sampling: {sampling}
  npoints: {npoints}
  context: text_cache
  seed: {seed}
  stage: "{stage}"

train:
  max_steps: {steps}
  # Both of these exist to avoid colliding with stage 1, which trained OpenDrawer at this
  # same point count. run_name is derived from `meta` alone, so s5-od-uniform-n8192-s0 and
  # the stage-1 od-uniform-n8192-s0 would resolve to ONE deterministic output_dir -- and the
  # trainer resumes from whatever checkpoints it finds there, so a 30K stage-5 run would
  # start from a finished 50K stage-1 checkpoint and report success. Stated explicitly here
  # rather than hoping the coordinates differ.
  run_name: {name}
  output_base: $SCRATCH/PointAct_exprs/robocasa365/stage5

data:
  lerobot_datasets:
    - repo_id: {task}
      state_action_norm_file: robot_data/robocasa365/lerobot_point_lmdb/{task}/robot_state_action_stats/rot6d.json
      text_context_file: text_context/qwen2.5-vl-3b.pt
      max_npoints: {npoints}
{block}"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npoints", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all", action="store_true",
                        help="Include CloseBlenderLid and CoffeeSetupMug, whose data-prep "
                             "chain has to have finished first.")
    args = parser.parse_args()

    # Imported here so --help works outside the pointact env.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from pointact.roi_sampling.geom_gt import ORACLE_TARGET

    tasks = dict(TASKS)
    if args.all:
        tasks.update(NEW_TASKS)

    out_dir = Path(__file__).parent
    # Regenerated wholesale, so a previous point count's copies do not sit alongside this
    # one's claiming to be the same grid.
    for stale in out_dir.glob("s5-*.yaml"):
        stale.unlink()

    written = []
    for task, abbrev in tasks.items():
        for sampling in ("uniform", "eef", "oracle", "molmo"):
            name = f"s5-{abbrev}-{sampling}-n{args.npoints}-s{args.seed}"
            block = BLOCK[sampling].format(gt_set=ORACLE_TARGET[task])
            (out_dir / f"{name}.yaml").write_text(TEMPLATE.format(
                task=task, sampling=sampling, npoints=args.npoints,
                steps=args.steps, steps_k=args.steps // 1000, seed=args.seed,
                stage=STAGE, head=HEAD[sampling], block=block, name=name,
            ))
            written.append(name)

    print(f"wrote {len(written)} run configs to {out_dir}:")
    for name in written:
        print(f"  {name}")


if __name__ == "__main__":
    main()
