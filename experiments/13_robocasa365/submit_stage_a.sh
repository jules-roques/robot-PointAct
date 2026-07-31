#!/bin/bash
# Submit all nine stage-A runs plus the gate that mails their summary.
#
#   bash experiments/13_robocasa365/submit_stage_a.sh            # submit
#   DRY_RUN=1 bash experiments/13_robocasa365/submit_stage_a.sh  # print what it would submit
#
# Each run is one allocation (the longest, 8192 points, is ~12h and fits comfortably in the
# 48h CLEPS / 100h Jean Zay limits). train_jeanzay_dev.slurm and its 2h chaining remain
# available for when the long queues stall.
#
# Eval arrays are submitted per run with --dependency=afterany, not afterok: a run that hits
# its walltime after writing checkpoint-50000 should still be evaluated. The array tasks whose
# checkpoints are missing exit 0 harmlessly.
set -euo pipefail

REPO="$HOME/code/robot-PointAct"
cd "$REPO"

RUNS_DIR=experiments/13_robocasa365/runs
TRAIN_SLURM=${TRAIN_SLURM:-experiments/13_robocasa365/train.slurm}
EVAL_SLURM=${EVAL_SLURM:-experiments/13_robocasa365/eval_grid.slurm}

submit() {
    if [ -n "${DRY_RUN:-}" ]; then
        echo "  would run: sbatch $*"
        echo "dryrun_$RANDOM"
        return
    fi
    sbatch --parsable "$@"
}

eval_job_ids=()
for config in "$RUNS_DIR"/od-*.yaml; do
    run=$(basename "$config" .yaml)
    echo "submitting $run"

    train_id=$(submit --job-name="rc365-$run" --export=ALL,RUN_CONFIG="$config" "$TRAIN_SLURM")
    echo "  train: $train_id"

    eval_id=$(submit --job-name="rc365-eval-$run" \
                     --dependency=afterany:"$train_id" \
                     --export=ALL,RUN="$run" "$EVAL_SLURM")
    echo "  eval : $eval_id"
    eval_job_ids+=("$eval_id")
done

# The gate waits on every eval array. afterok here (unlike above): a summary assembled from a
# crashed eval would quietly under-report an arm, and choosing the stage-B point count off
# that is exactly the mistake this gate exists to prevent.
dependency=$(IFS=:; echo "${eval_job_ids[*]}")
gate_id=$(submit --dependency=afterok:"$dependency" experiments/13_robocasa365/gate_stage_a.slurm)
echo
echo "gate: $gate_id (mails the point-count table when all ${#eval_job_ids[@]} eval arrays finish)"
echo "then: python $RUNS_DIR/generate_stage_b.py --npoints <your choice>"
