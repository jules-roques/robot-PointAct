"""Generate the stage-8 run yamls: does more ROI resolution help once the budget is saturated?

Stage 7 left OpenDrawer here (30K, n=500 pooled): the oracle arm is flat at ~76% from 2048
points up, eef from 4096, and the whole unsampled cloud (75.7%) ties the oracle crop. The point
budget stopped paying because the ROI ran out of points, not because the policy stopped
wanting them: at 1 cm the oracle Gaussian already takes 96% of the points within 4 cm of the
handle at 4096 and 100% at 8192. The only way to put MORE points on the handle is a finer
grid. Stage 8 asks whether that helps.

    python experiments/13_robocasa365/runs/generate_stage8.py [--samplers oracle eef]

**How resolution is changed: scale the cloud, keep grid_size at 0.01.** Utonia's network
voxel is a hardcoded 1 cm, and its own transform changes granularity with a RandomScale in
front of it, so an arm at scale s sees one network voxel = 0.01/s metres. Order is
scale -> voxelize -> sample, never sample first: a later voxel merge would shrink the draw
below the budget by a grid-dependent amount, and the budget is what this stage holds fixed.
In code that is two knobs that travel with the checkpoint into eval:

* data ``point_voxel_size = 0.01/s``: the dataset re-voxelizes the cache to that metric grid
  before sampling (it may only coarsen, so the fine arms read one 2.5 mm render);
* train ``ptv3_coord_scale = s``: coordinates are multiplied by s at the PTv3 input.

Everything upstream of the encoder stays metric, so sigma = 8 cm is 8 cm at every s (8s cm in
network units) -- the Gaussian scales with the cloud without being touched.

**Two families, and the second is what makes the first readable.**

* *fine* -- s in {sqrt 2, 2, 2 sqrt 2}, each RENDERED DIRECTLY at its own grid (7.1 / 5 /
  3.5 mm): new points in the ROI. s = 1 is stage 7's ``s7-od-oracle-n8192-s0``. Until
  2026-10-02 all three re-voxelized one 2.5 mm render instead; that is raw -> 2.5 mm -> grid in
  training against raw -> grid in eval (run_server voxelizes the live cloud once), a
  train/eval gap only the fine arms had, so it confounded fine vs control. With a render per
  grid, point_voxel_size equals the cache's own grid and the dataset re-voxelization is a
  no-op: training and eval both voxelize raw camera points once. The top arm was s = 4
  (2.5 mm) until 2026-10-02, before anything trained: 2 sqrt 2 keeps the axis on even sqrt 2
  steps, and s = 4 was the arm furthest off Utonia's single pretraining granularity (coarsest
  level 4 cm, 1 cm context points 4 voxels apart) for little expected gain -- at 8192 points
  with sigma 8 cm / floor 0.05 the sampler's share of the ROI caps it near ~1K points, which
  the 5 mm grid already roughly fills (estimate, not measured).
* *control* -- s in {2, 2 sqrt 2} on the existing 1 cm data: the same rescale, NO new points. Scaling
  has side effects of its own -- the coarsest encoder level covers 16/s cm, and context points
  1 cm apart sit s voxels apart, off Utonia's convention. Fine wins and control does not ->
  resolution helped. Both lose -> the rescale costs more than the detail buys (the 5 mm result
  again). Both win -> it was the rescale, not the points. Without the control, stage 1 bis
  showed, you cannot tell these apart.

**Held fixed:** OpenDrawer, 8192 points, 30K steps cosine, sigma 8 cm / floor 0.05, Utonia
recipe from _base.yaml. per_device_train_batch_size drops to 16 for s > 1 (effective batch
stays 128): rescaled clouds pool less per stride-2 level, so the deep stages carry more tokens.
"""

import argparse
import math
from pathlib import Path

STAGE = "Stage 8: ROI resolution at fixed budget (Utonia scale)"
NPOINTS = 8192
def render_root(grid: float) -> str:
    """The dataset root convert.slurm writes for VOXEL_SIZE=grid ('_g%gmm' % (grid * 1000))."""
    return f"robot_data/robocasa365/lerobot_point_lmdb_g{grid * 1000:g}mm"


# (scale, metric point grid, dataset root or None for _base's 1 cm root). Fine arms read the
# render made AT their grid: replay/convert with VOXEL_SIZE=<grid>, the same float as here.
ARMS = [
    (math.sqrt(2), 0.01 / math.sqrt(2), render_root(0.01 / math.sqrt(2))),
    (2.0, 0.005, render_root(0.005)),
    (2 * math.sqrt(2), 0.01 / (2 * math.sqrt(2)), render_root(0.01 / (2 * math.sqrt(2)))),
    # Re-added 2026-10-03 at the user's request, as the far end of the axis: s = 4 puts the
    # cloud furthest off Utonia's single pretraining granularity (coarsest level 4 cm), and the
    # question is whether the policy collapses there. Its 2.5 mm render was made at this grid,
    # so it has no train/eval voxelization gap either.
    (4.0, 0.0025, render_root(0.0025)),
    (2.0, 0.01, None),
    (2 * math.sqrt(2), 0.01, None),
]

BLOCK = {
    "eef": """      eef_sampling: true
      eef_sampling_sigma: 0.08
      eef_sampling_floor: 0.05
""",
    "oracle": """      oracle_sampling: true
      oracle_gt: geom
      oracle_gt_npz: roi_meta/target_positions.npz
      oracle_gt_set: handle
      oracle_sampling_sigma: 0.08
      oracle_sampling_floor: 0.05
""",
}

TEMPLATE = """# OpenDrawer / {sampling} / {npoints} points / scale {scale:.4g} on a {grid_mm:.3g} mm grid -- stage 8.
# {family}. Network voxel = 0.01 / {scale:.4g} = {net_mm:.3g} mm; coarsest encoder level
# covers {rf_cm:.3g} cm. See generate_stage8.py for the design.
extends: _base.yaml

meta:
  task: OpenDrawer
  sampling: {sampling}
  npoints: {npoints}
  context: text_cache
  seed: {seed}
  stage: "{stage}"
  scale: {scale}
  grid: {grid}

train:
  max_steps: {steps}
  run_name: {name}
  output_base: $SCRATCH/PointAct_exprs/robocasa365/stage8
  per_device_train_batch_size: 16
  ptv3_coord_scale: {scale}

data:
  lerobot_datasets:
    - repo_id: OpenDrawer
{root_line}      state_action_norm_file: robot_data/robocasa365/lerobot_point_lmdb/OpenDrawer/robot_state_action_stats/rot6d.json
      text_context_file: text_context/qwen2.5-vl-3b.pt
      max_npoints: {npoints}
      point_voxel_size: {grid}
{block}"""


def arm_name(sampling: str, scale: float, grid: float, seed: int) -> str:
    tokens = [f"x{scale:.3g}"]
    if abs(grid - 0.01) > 1e-12:
        tokens.append(f"g{grid * 1000:.3g}mm")
    return f"s8-od-{sampling}-n{NPOINTS}-{'-'.join(tokens)}-s{seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samplers", nargs="+", default=["oracle"], choices=list(BLOCK))
    args = parser.parse_args()

    out_dir = Path(__file__).parent
    for stale in out_dir.glob("s8-*.yaml"):
        stale.unlink()

    written = []
    for sampling in args.samplers:
        for scale, grid, root in ARMS:
            name = arm_name(sampling, scale, grid, args.seed)
            family = ("fine: new ROI points from the 2.5 mm render" if root
                      else "control: same rescale on the 1 cm data, no new points")
            (out_dir / f"{name}.yaml").write_text(TEMPLATE.format(
                sampling=sampling, npoints=NPOINTS, scale=scale, grid=grid,
                grid_mm=grid * 1000, net_mm=10 / scale, rf_cm=16 / scale, family=family,
                seed=args.seed, stage=STAGE, steps=args.steps, name=name,
                root_line=f"      root: {root}\n" if root else "",
                block=BLOCK[sampling],
            ))
            written.append(name)

    print(f"wrote {len(written)} run configs to {out_dir}:")
    for name in written:
        print(f"  {name}")


if __name__ == "__main__":
    main()
