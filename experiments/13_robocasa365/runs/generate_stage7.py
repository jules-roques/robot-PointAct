"""Generate the stage-7 run yamls: one task, three samplers, six point budgets.

Stage 7 re-derives the point-budget axis that stage 1 measured, on an encoder whose
pretraining granularity matches what we feed it. Stage 1 initialised PTv3 from **Concerto**,
which pretrains on a 2 cm grid, while PointAct voxelises at 1 cm and hardcodes
``grid_size=0.01`` into the encoder -- so every stage-1 number carries a factor-of-two
granularity mismatch between pretraining and inference. **Utonia** pretrains at a hardcoded
``grid_size=0.01`` (its per-domain ``RandomScale`` is what moves each dataset onto that one
canonical unit), which is exactly the unit PointAct feeds, so it is the encoder this axis
should have been measured on. ``runs/_base.yaml`` has carried ``ptv3_backend: utonia`` since
2026-08-17; stage 1 predates that.

    python experiments/13_robocasa365/runs/generate_stage7.py

Why these coordinates:

* **OpenDrawer only.** One task, deliberately. Stage 5 already showed the sampler ordering is
  task-dependent (eef wins on four tasks and loses 8.4pp on CoffeeSetupMug), so a point-budget
  curve pooled over tasks would average two different shapes. Extend to more tasks once the
  shape on one is known.
* **Six budgets, 512 -> 16384.** The interesting region is the LOW end. At 1 cm the whole
  OpenDrawer cloud is only ~18-22K points and the eef draw already takes 97% of everything
  within one sigma of the gripper at 4096, so 4096 -> 8192 buys about eleven ROI points and
  the axis is expected to be flat up there (measured 2026-08-03, see the README's voxel-probe
  table). 512 is where a uniform draw should visibly break while a targeted one holds -- the
  knee that makes this a curve rather than a line.
* **16384 is the convergence end, not a "big budget".** It keeps ~73% of the median cloud, so
  the three samplers can differ only in which ~27% they discard and must converge there by
  construction. That is the point of including it: it brackets the axis between "sampling
  decides everything" and "sampling cannot matter".
* **Three samplers.** uniform (baseline), eef (the deployable prior -- no cache, no privileged
  information), oracle (the same Gaussian centred on the simulator's own target position; an
  upper bound on what any sampler could buy, not a policy). MolmoPoint arms are deliberately
  absent: they did not beat the eef prior and were dropped from new grids on 2026-09-07.
* **50K steps, checkpoints every 5K.** Back to the stage-1 duration schedule -- stage 5 cut to
  30K to buy rollouts, but the duration axis is half of what this stage reports, and a cosine
  schedule annealed to 30K is not a prefix of a 50K one, so the two cannot share a curve.

**The full-cloud arm is not generated here.** ``od-none-s0`` (stage 6) is already trained to
50K on this exact recipe -- Utonia, cached text context, effective batch 128, 1 cm cloud with
no budget -- with checkpoints every 2500. It is reused as the no-sampler end of the axis
rather than retrained; see the stage-7 section of the README for how it is folded into the
eval and the table.
"""

import argparse
from pathlib import Path

TASK = "OpenDrawer"
ABBREV = "od"

STAGE = "Stage 7: Point budget x sampler (Utonia)"

#: The budget axis. 16384 is ~73% of the median 1 cm cloud, so it is the convergence end.
BUDGETS = (512, 1024, 2048, 4096, 8192, 16384)

#: Above this budget the per-device batch drops so the arm fits in 80 GB. `effective_batch`
#: is held at 128 either way -- gradient accumulation absorbs the change -- and effective
#: batch is the quantity the arms have to share to be comparable. Stage 6 needed 16 at the
#: unsampled (~22K) budget and measured 32 as an OOM there.
LARGE_BUDGET_BATCH = {16384: 16}

HEAD = {
    "uniform": "Uniform point subsample: the baseline draw, and the control the other two arms\n"
               "# in this grid are read against.",
    "eef": "EEF-density sampling: a Gaussian-with-floor density centred on the frame's own\n"
           "# end-effector position. No cache and no privileged information -- the deployable arm.",
    "oracle": "GT-oracle sampling: the same density centred on where the simulator says the\n"
              "# target actually is. Privileged; an upper bound on what any sampler could buy,\n"
              "# not a deployable policy.",
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
}

TEMPLATE = """# {task} / {sampling} / {npoints} points, {steps_k}K steps -- stage 7.
# {head}
#
# Stage 7 re-runs the point-budget axis on Utonia. Stage 1 measured it on Concerto, which
# pretrains at a 2 cm grid against the 1 cm this pipeline voxelises and feeds, so its
# numbers are not the ones to quote for "how many points does a point policy need?".
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
  # run_name is derived from `meta` alone, so without these two the stage-1 arms at 2048 /
  # 4096 / 8192 would resolve to the SAME deterministic output_dir as their stage-7
  # counterparts -- and the trainer resumes from whatever checkpoints it finds there, so a
  # fresh Utonia arm would silently continue a finished Concerto run and report success.
  # Stated explicitly rather than hoping the coordinates differ.
  run_name: {name}
  output_base: $SCRATCH/PointAct_exprs/robocasa365/stage7
{batch_block}
data:
  lerobot_datasets:
    - repo_id: {task}
      state_action_norm_file: robot_data/robocasa365/lerobot_point_lmdb/{task}/robot_state_action_stats/rot6d.json
      text_context_file: text_context/qwen2.5-vl-3b.pt
      max_npoints: {npoints}
{block}"""

BATCH_BLOCK = """
  # ~{tokens} stage-0 tokens per sample at this budget; 32 does not fit in 80 GB. effective_batch
  # stays 128 (gradient accumulation absorbs the change), which is what makes this arm
  # comparable to the rest of the grid. If it still OOMs, resubmit at 8 -- the run
  # auto-resumes from output_dir.
  per_device_train_batch_size: {batch}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    args = parser.parse_args()

    # Imported here so --help works outside the pointact env.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from pointact.roi_sampling.geom_gt import ORACLE_TARGET

    out_dir = Path(__file__).parent
    # Regenerated wholesale, so a previous budget list's copies do not sit alongside this
    # one's claiming to be the same grid.
    for stale in out_dir.glob("s7-*.yaml"):
        stale.unlink()

    written = []
    for npoints in args.budgets:
        for sampling in ("uniform", "eef", "oracle"):
            name = f"s7-{ABBREV}-{sampling}-n{npoints}-s{args.seed}"
            batch = LARGE_BUDGET_BATCH.get(npoints)
            batch_block = "" if batch is None else BATCH_BLOCK.format(
                batch=batch, tokens=f"{npoints // 1000}K")
            (out_dir / f"{name}.yaml").write_text(TEMPLATE.format(
                task=TASK, sampling=sampling, npoints=npoints,
                steps=args.steps, steps_k=args.steps // 1000, seed=args.seed,
                stage=STAGE, head=HEAD[sampling], name=name, batch_block=batch_block,
                block=BLOCK[sampling].format(gt_set=ORACLE_TARGET[TASK]),
            ))
            written.append(name)

    print(f"wrote {len(written)} run configs to {out_dir}:")
    for name in written:
        print(f"  {name}")
    print("\nthe no-sampler end of the axis is the existing od-none-s0 (stage 6), reused --")
    print("see the Stage 7 section of experiments/13_robocasa365/README.md")


if __name__ == "__main__":
    main()
