#!/bin/bash
# Submit a set of ablation runs, each followed by its eval array.
#
#   bash experiments/13_robocasa365/submit_stage_a.sh            # the nine stage-1 arms + gate
#   DRY_RUN=1 bash experiments/13_robocasa365/submit_stage_a.sh  # print what it would submit
#
# The stage-0 (two-camera-view) arms are a separate submission: they are ~3x slower per step,
# so they need the longer QoS, and the point-count gate does not apply to them.
#
#   RUNS="od-uniform-n4096-vlm-s0 od-eef-n4096-vlm-s0 od-anchor-n4096-vlm-s0" \
#   SUBMIT_GATE=0 \
#   TRAIN_EXTRA="--qos=qos_gpu_h100-t4 --time=30:00:00 --constraint=h100" \
#   bash experiments/13_robocasa365/submit_stage_a.sh
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

# Named explicitly rather than globbed. `od-*.yaml` used to be the whole grid, but the
# directory now also holds the stage-0 two-camera-view arms -- a glob would have silently
# grown this submission by three 21-hour runs.
DEFAULT_RUNS=(od-uniform-n2048-s0 od-eef-n2048-s0 od-anchor-n2048-s0
              od-uniform-n4096-s0 od-eef-n4096-s0 od-anchor-n4096-s0
              od-uniform-n8192-s0 od-eef-n8192-s0 od-anchor-n8192-s0)
read -r -a RUN_LIST <<< "${RUNS:-${DEFAULT_RUNS[*]}}"

eval_job_ids=()
for run in "${RUN_LIST[@]}"; do
    config="$RUNS_DIR/$run.yaml"
    [ -f "$config" ] || { echo "no such run config: $config" >&2; exit 1; }
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
# crashed eval would quietly under-report an arm, and choosing the point count off that is
# exactly the mistake this gate exists to prevent. Off for submissions that are not the
# point-count sweep -- there is nothing for them to gate.
if [ "${SUBMIT_GATE:-1}" = "1" ]; then
    dependency=$(IFS=:; echo "${eval_job_ids[*]}")
    gate_id=$(submit --dependency=afterok:"$dependency" "$GATE_SLURM")
    echo
    echo "gate: $gate_id (mails the point-count table when all ${#eval_job_ids[@]} eval arrays finish)"
    echo "then: python $RUNS_DIR/generate_stage_b.py --npoints <your choice>"
fi

# W&B is offline on Jean Zay compute nodes; nothing pushes it automatically, by choice.
if [ -n "${DSDIR:-}" ]; then
    echo
    echo "W&B curves stay local until you push them. From a Jean Zay LOGIN node:"
    echo "  bash experiments/13_robocasa365/wandb_sync_jeanzay.sh"
fi
