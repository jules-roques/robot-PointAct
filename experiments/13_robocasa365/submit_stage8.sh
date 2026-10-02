#!/bin/bash
# Submit the stage-8 trainings: OpenDrawer x oracle x 8192 points at coordinate scale
# s in {sqrt2, 2, 2sqrt2} on the 2.5 mm render (fine) and s in {2, 2sqrt2} on the 1 cm data (control).
# See runs/generate_stage8.py for the design; s = 1 is stage 7's s7-od-oracle-n8192-s0.
#
#   DRY_RUN=1 bash experiments/13_robocasa365/submit_stage8.sh   # print what it would submit
#   bash experiments/13_robocasa365/submit_stage8.sh             # all five arms
#   S8_FAMILY=control bash .../submit_stage8.sh                  # only the 1 cm controls
#
# The fine arms need the 2.5 mm dataset, built on V100 + CPU (no H100) by:
#   sbatch --array=0-7 --export=ALL,VOXEL_SIZE=0.0025,REPO=$PWD data_prep/robocasa365_to_lerobot/replay.slurm
#   sbatch --export=ALL,VOXEL_SIZE=0.0025,REPO=$PWD data_prep/robocasa365_to_lerobot/convert.slurm
# then link the 1 cm root's text_context/, roi_meta/ and robot_state_action_stats/ into it, and build
# meta/source_episode_map.json (python -m data_prep.robocasa365_to_lerobot.episode_index_map
# --source-dir <src>/lerobot --dataset-dir <root>/OpenDrawer; the geom oracle needs it). Same 514 episodes, identity
# map -- check meta/source_episode_map.json). This script refuses a fine arm until it exists.
#
# Anything already queued under the same job name is skipped. Smoke new arms first:
#   sbatch --constraint=h100 --qos=qos_gpu_h100-dev --time=00:40:00 \
#          --export=ALL,RUN_CONFIG=experiments/13_robocasa365/runs/<arm>.yaml,SMOKE_STEPS=30 \
#          experiments/13_robocasa365/train_jeanzay.slurm
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
RUNS_DIR=experiments/13_robocasa365/runs
TRAIN_SLURM=experiments/13_robocasa365/train_jeanzay.slurm
TRAIN_EXTRA=${TRAIN_EXTRA:---qos=qos_gpu_h100-t3 --time=20:00:00 --constraint=h100}
FINE_ROOT="robot_data/robocasa365/lerobot_point_lmdb_g2.5mm/OpenDrawer"

submit() {
    if [ -n "${DRY_RUN:-}" ]; then
        echo "  would run: sbatch $*" >&2
        echo "dryrun_$RANDOM"
        return
    fi
    sbatch --parsable "$@"
}

CONTROL=(s8-od-oracle-n8192-x2-s0 s8-od-oracle-n8192-x2.83-s0)
FINE=(s8-od-oracle-n8192-x1.41-g7.07mm-s0 s8-od-oracle-n8192-x2-g5mm-s0
      s8-od-oracle-n8192-x2.83-g3.54mm-s0)
case "${S8_FAMILY:-all}" in
    control) DEFAULT_RUNS=("${CONTROL[@]}") ;;
    fine)    DEFAULT_RUNS=("${FINE[@]}") ;;
    all)     DEFAULT_RUNS=("${CONTROL[@]}" "${FINE[@]}") ;;
    *) echo "S8_FAMILY must be control|fine|all" >&2; exit 1 ;;
esac
read -r -a RUN_LIST <<< "${RUNS:-${DEFAULT_RUNS[*]}}"

QUEUED=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)
for run in "${RUN_LIST[@]}"; do
    config="$RUNS_DIR/$run.yaml"
    [ -f "$config" ] || { echo "no such run config: $config" >&2; exit 1; }
    if grep -q "lerobot_point_lmdb_g2.5mm" "$config"; then
        for need in points_3views cache_meta.json text_context roi_meta/target_positions.npz meta/source_episode_map.json; do
            if [ ! -e "$FINE_ROOT/$need" ]; then
                echo "refusing $run: $FINE_ROOT/$need missing (build the 2.5 mm dataset first)" >&2
                exit 1
            fi
        done
    fi
    if grep -qxF "$run" <<< "$QUEUED"; then
        echo "skipped $run (already queued or running)"
        continue
    fi
    echo "submitted $run -> $(submit --job-name="$run" $TRAIN_EXTRA \
        --export=ALL,RUN_CONFIG="$config" "$TRAIN_SLURM")"
done

cat <<'NEXT'

--- when the trainings are done ------------------------------------------------
Eval needs nothing stage-specific: run_server reads ptv3_coord_scale (model) and
point_voxel_size (live-cloud grid) off each checkpoint. Look for
"voxel_size=... coord_scale=..." in the server log of the smoke eval to confirm.

   ARMS="s8-od-oracle-n8192-x2-s0 s8-od-oracle-n8192-x2.83-s0 s8-od-oracle-n8192-x1.41-g7.07mm-s0 s8-od-oracle-n8192-x2-g5mm-s0 s8-od-oracle-n8192-x2.83-g3.54mm-s0"
   sbatch --job-name=eval-s8-od-curve \
     --export=ALL,EXPRS_DIR=$SCRATCH/PointAct_exprs/robocasa365/stage8,\
EVAL_STEPS="5000 10000 15000 20000 25000 30000",EVAL_SEEDS="7",NUM_TRIALS=100,RUNS="$ARMS" \
     experiments/13_robocasa365/eval_task_jeanzay.slurm
   sbatch --job-name=eval-s8-od-headline \
     --export=ALL,EXPRS_DIR=$SCRATCH/PointAct_exprs/robocasa365/stage8,\
EVAL_STEPS="30000",EVAL_SEEDS="11 13 17 19",NUM_TRIALS=100,RUNS="$ARMS" \
     experiments/13_robocasa365/eval_task_jeanzay.slurm

Same seeds as stage 7 (7 + 11/13/17/19 -> n=500), so s = 1 comes from s7-od-oracle-n8192-s0.
NEXT
