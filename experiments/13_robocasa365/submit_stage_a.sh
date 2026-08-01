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

# Resolve the checkout from this script's own location, not $HOME: the ablation lives in a
# worktree, and the repo sits at a different path on each cluster ($HOME/code on CLEPS,
# $WORK/code on Jean Zay). Hardcoding either makes the run-config glob below silently match
# nothing and submit an empty grid.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
echo "repo: $REPO"

RUNS_DIR=experiments/13_robocasa365/runs

# Cluster is detected from $DSDIR, which only Jean Zay defines -- the same test the rest of the
# experiment scripts use. Jean Zay and CLEPS are separate SLURM controllers, so the #SBATCH
# account/partition/QoS directives live in per-cluster files rather than being parameterised.
if [ -n "${DSDIR:-}" ]; then
    SUFFIX="_jeanzay"
    # H100 t3 caps at 20h, comfortably above the longest measured run (7.6h at 8192 points),
    # and turns around faster than t4.
    TRAIN_EXTRA=${TRAIN_EXTRA:---qos=qos_gpu_h100-t3 --time=20:00:00 --constraint=h100}
else
    SUFFIX=""
    TRAIN_EXTRA=${TRAIN_EXTRA:-}
fi
TRAIN_SLURM=${TRAIN_SLURM:-experiments/13_robocasa365/train${SUFFIX}.slurm}
EVAL_SLURM=${EVAL_SLURM:-experiments/13_robocasa365/eval_grid${SUFFIX}.slurm}
GATE_SLURM=${GATE_SLURM:-experiments/13_robocasa365/gate_stage_a${SUFFIX}.slurm}
echo "cluster files: $TRAIN_SLURM / $EVAL_SLURM / $GATE_SLURM"

submit() {
    if [ -n "${DRY_RUN:-}" ]; then
        # To stderr: stdout is captured as the job id by the caller.
        echo "  would run: sbatch $*" >&2
        echo "dryrun_$RANDOM"
        return
    fi
    sbatch --parsable "$@"
}

eval_job_ids=()
for config in "$RUNS_DIR"/od-*.yaml; do
    run=$(basename "$config" .yaml)
    echo "submitting $run"

    train_id=$(submit --job-name="$run" $TRAIN_EXTRA \
                      --export=ALL,RUN_CONFIG="$config" "$TRAIN_SLURM")
    echo "  train: $train_id"

    eval_id=$(submit --job-name="eval-$run" \
                     --dependency=afterany:"$train_id" \
                     --export=ALL,RUN="$run" "$EVAL_SLURM")
    echo "  eval : $eval_id"
    eval_job_ids+=("$eval_id")
done

# The gate waits on every eval array. afterok here (unlike above): a summary assembled from a
# crashed eval would quietly under-report an arm, and choosing the stage-B point count off
# that is exactly the mistake this gate exists to prevent.
dependency=$(IFS=:; echo "${eval_job_ids[*]}")
gate_id=$(submit --dependency=afterok:"$dependency" "$GATE_SLURM")
echo
echo "gate: $gate_id (mails the point-count table when all ${#eval_job_ids[@]} eval arrays finish)"
echo "then: python $RUNS_DIR/generate_stage_b.py --npoints <your choice>"
