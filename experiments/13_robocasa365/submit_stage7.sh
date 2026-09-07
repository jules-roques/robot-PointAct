#!/bin/bash
# Submit the stage-7 trainings: OpenDrawer x {uniform, eef, oracle} x six point budgets.
#
#   DRY_RUN=1 bash experiments/13_robocasa365/submit_stage7.sh   # print what it would submit
#   bash experiments/13_robocasa365/submit_stage7.sh             # all 18 arms
#   RUNS="s7-od-eef-n512-s0 s7-od-oracle-n512-s0" bash .../submit_stage7.sh   # a subset
#
# Unlike submit_stage_a.sh this submits TRAINING ONLY. Evaluation is no longer one array per
# checkpoint: it is a single packed node-job per task (eval_task_jeanzay.slurm), which runs the
# whole RUNS x EVAL_STEPS x EVAL_SEEDS grid at CONCURRENCY pairs at a time and is idempotent,
# so it is submitted once by hand when the trainings are done rather than chained per arm. The
# commands are printed at the end.
#
# Smoke first. Every one of these is a ~3-11 h allocation, and a config error costs the whole
# queue round-trip; the dev QoS starts in seconds:
#
#   sbatch --constraint=h100 --qos=qos_gpu_h100-dev --time=00:40:00 \
#          --export=ALL,RUN_CONFIG=experiments/13_robocasa365/runs/s7-od-oracle-n512-s0.yaml,SMOKE_STEPS=30 \
#          experiments/13_robocasa365/train_jeanzay.slurm
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
echo "repo: $REPO"

RUNS_DIR=experiments/13_robocasa365/runs

# Cluster is detected from $DSDIR, which only Jean Zay defines -- the same test the rest of the
# experiment scripts use.
if [ -n "${DSDIR:-}" ]; then
    SUFFIX="_jeanzay"
    # t3 caps at 20 h. The longest arm here is 16384 points at ~11 h, so t3 fits and turns
    # around faster than t4, which would ask the scheduler to reserve five times what is used.
    TRAIN_EXTRA=${TRAIN_EXTRA:---qos=qos_gpu_h100-t3 --time=20:00:00 --constraint=h100}
else
    SUFFIX=""
    TRAIN_EXTRA=${TRAIN_EXTRA:-}
fi
TRAIN_SLURM=${TRAIN_SLURM:-experiments/13_robocasa365/train${SUFFIX}.slurm}
echo "cluster file: $TRAIN_SLURM"

submit() {
    if [ -n "${DRY_RUN:-}" ]; then
        echo "  would run: sbatch $*" >&2
        echo "dryrun_$RANDOM"
        return
    fi
    sbatch --parsable "$@"
}

# Named explicitly rather than globbed, for the reason submit_stage_a.sh learned the hard way:
# a glob silently grows the submission when a neighbouring stage drops a file in runs/.
# Ordered cheapest-first so the low-budget arms -- the ones the curve's knee depends on --
# clear the queue before the expensive tail.
DEFAULT_RUNS=(
    s7-od-uniform-n512-s0    s7-od-eef-n512-s0    s7-od-oracle-n512-s0
    s7-od-uniform-n1024-s0   s7-od-eef-n1024-s0   s7-od-oracle-n1024-s0
    s7-od-uniform-n2048-s0   s7-od-eef-n2048-s0   s7-od-oracle-n2048-s0
    s7-od-uniform-n4096-s0   s7-od-eef-n4096-s0   s7-od-oracle-n4096-s0
    s7-od-uniform-n8192-s0   s7-od-eef-n8192-s0   s7-od-oracle-n8192-s0
    s7-od-uniform-n16384-s0  s7-od-eef-n16384-s0  s7-od-oracle-n16384-s0
)
read -r -a RUN_LIST <<< "${RUNS:-${DEFAULT_RUNS[*]}}"

for run in "${RUN_LIST[@]}"; do
    config="$RUNS_DIR/$run.yaml"
    [ -f "$config" ] || { echo "no such run config: $config" >&2; exit 1; }
    train_id=$(submit --job-name="$run" $TRAIN_EXTRA \
                      --export=ALL,RUN_CONFIG="$config" "$TRAIN_SLURM")
    echo "submitted $run -> $train_id"
done

cat <<'NEXT'

--- when the trainings are done ------------------------------------------------

Eval is two packed node-jobs. Both are idempotent (a pair whose final JSON exists is
skipped), so resubmit the identical command to finish a grid that ran out of walltime.

1. The duration curve: 100 trials at every 5K checkpoint, one seed.

   sbatch --export=ALL,EXPRS_DIR=$SCRATCH/PointAct_exprs/robocasa365/stage7,\
EVAL_STEPS="5000 10000 15000 20000 25000 30000 35000 40000 45000 50000",\
EVAL_SEEDS="7",NUM_TRIALS=100,\
RUNS="$(echo s7-od-{uniform,eef,oracle}-n{512,1024,2048,4096,8192,16384}-s0)" \
     experiments/13_robocasa365/eval_task_jeanzay.slurm

2. The headline table: four more seeds at 50K only, pooled with seed 7 above to n=500.

   sbatch --export=ALL,EXPRS_DIR=$SCRATCH/PointAct_exprs/robocasa365/stage7,\
EVAL_STEPS="50000",EVAL_SEEDS="11 13 17 19",NUM_TRIALS=100,\
RUNS="<the same 18>" \
     experiments/13_robocasa365/eval_task_jeanzay.slurm

3. The no-sampler end of the axis lives in a DIFFERENT output tree (it is the reused stage-6
   arm), so it needs its own submission with EXPRS_DIR pointing at `ablation`:

   sbatch --export=ALL,EXPRS_DIR=$SCRATCH/PointAct_exprs/robocasa365/ablation,\
EVAL_STEPS="5000 10000 15000 20000 25000 30000 35000 40000 45000 50000",\
EVAL_SEEDS="7",NUM_TRIALS=100,RUNS="od-none-s0" \
     experiments/13_robocasa365/eval_task_jeanzay.slurm

Rehearse any new grid with SAVE_SUFFIX=-smoke first: both pooling consumers skip a results
directory whose trailing "-" token is not all digits, so smoke output is invisible to every
success rate. SAVE_SUFFIX=-2 would be pooled as step 2.

W&B is offline on Jean Zay compute nodes. From a LOGIN node:
   bash experiments/13_robocasa365/wandb_sync_jeanzay.sh
   python experiments/13_robocasa365/log_eval_to_wandb.py --run-dir <output_dir>
NEXT
