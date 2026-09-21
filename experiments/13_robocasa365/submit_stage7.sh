#!/bin/bash
# Submit the stage-7 trainings: {OpenDrawer, CloseBlenderLid, PickPlaceCounterToStove}
# x {uniform, eef, oracle} x six point budgets, plus one no-sampler arm per task = 57.
#
#   DRY_RUN=1 bash experiments/13_robocasa365/submit_stage7.sh   # print what it would submit
#   bash experiments/13_robocasa365/submit_stage7.sh             # all 57 arms
#   S7_TASKS="blender ppcs" bash .../submit_stage7.sh            # only the two new tasks
#   RUNS="s7-od-eef-n512-s0 s7-od-oracle-n512-s0" bash .../submit_stage7.sh   # a subset
#
# Anything already queued or running under the same job name is SKIPPED, so re-running this
# after adding a task submits only what is missing. Always DRY_RUN=1 first: at 57 arms a
# full accidental resubmission is several hundred H100-hours.
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

# Built from the axes rather than listed one by one, because the grid is now 3 tasks x
# (3 samplers x 6 budgets + 1 no-sampler) = 57 arms and a hand-kept list of 57 names is a
# transcription bug waiting to happen. Still NOT a glob over runs/*.yaml, for the reason
# submit_stage_a.sh learned the hard way: a glob silently grows the submission when a
# neighbouring stage drops a file in runs/.
#
# Ordered cheapest-first ACROSS tasks -- every task's 512 arm goes in before any task's 1024
# -- so the low-budget arms, which is where the curve's knee lives, all clear the queue
# before the expensive tail. If the budget runs out partway, what survives is a complete
# low-end curve on three tasks rather than one finished task and two empty ones.
S7_TASKS=${S7_TASKS:-"od blender ppcs"}
S7_BUDGETS=${S7_BUDGETS:-"512 1024 2048 4096 8192 16384"}

DEFAULT_RUNS=()
for budget in $S7_BUDGETS; do
    for task in $S7_TASKS; do
        for sampling in uniform eef oracle; do
            DEFAULT_RUNS+=("s7-${task}-${sampling}-n${budget}-s0")
        done
    done
done
# The no-sampler arms last: they are the most expensive per step and the least likely to
# change the story, so they are the right thing to lose if the budget is cut short.
for task in $S7_TASKS; do
    DEFAULT_RUNS+=("s7-${task}-none-s0")
done

read -r -a RUN_LIST <<< "${RUNS:-${DEFAULT_RUNS[*]}}"

# The grid is submitted in waves now (OpenDrawer 2026-09-10, the other two 2026-09-21), so a
# rerun of this script would otherwise queue a second copy of everything already waiting.
# Training auto-resumes from output_dir, so a duplicate is not corrupting -- but two jobs
# writing one output_dir is a real way to lose a run, and it wastes a 4-GPU allocation.
QUEUED=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)

skipped=0
for run in "${RUN_LIST[@]}"; do
    config="$RUNS_DIR/$run.yaml"
    [ -f "$config" ] || { echo "no such run config: $config" >&2; exit 1; }
    if grep -qxF "$run" <<< "$QUEUED"; then
        echo "skipped $run (already queued or running)"
        skipped=$((skipped + 1))
        continue
    fi
    train_id=$(submit --job-name="$run" $TRAIN_EXTRA \
                      --export=ALL,RUN_CONFIG="$config" "$TRAIN_SLURM")
    echo "submitted $run -> $train_id"
done
# `[ ... ] && echo` as the last statement would exit 1 under `set -e` whenever nothing was
# skipped, which is the normal case -- so this is an if, not a one-liner.
if [ "$skipped" -gt 0 ]; then
    echo "($skipped already in the queue, left alone)"
fi

cat <<'NEXT'

--- when the trainings are done ------------------------------------------------

Eval is two packed node-jobs. Both are idempotent (a pair whose final JSON exists is
skipped), so resubmit the identical command to finish a grid that ran out of walltime.

All 57 arms live under the same output tree, so one EXPRS_DIR covers the grid -- but
eval_task_jeanzay.slurm REFUSES a mixed-task RUNS list (it checks, and exits), because
one task per job is what keeps the walltime estimate meaningful. So this is six
submissions: two per task, not two in total.

1. The duration curve: 100 trials at every 5K checkpoint, one seed. Once per task.

   for T in od blender ppcs; do
     ARMS="$(echo s7-$T-{uniform,eef,oracle}-n{512,1024,2048,4096,8192,16384}-s0) s7-$T-none-s0"
     sbatch --job-name="eval-s7-$T-curve" \
       --export=ALL,EXPRS_DIR=$SCRATCH/PointAct_exprs/robocasa365/stage7,\
EVAL_STEPS="5000 10000 15000 20000 25000 30000",\
EVAL_SEEDS="7",NUM_TRIALS=100,RUNS="$ARMS" \
       experiments/13_robocasa365/eval_task_jeanzay.slurm
   done

2. The headline table: four more seeds at 30K only, pooled with seed 7 above to n=500.

   for T in od blender ppcs; do
     ARMS="$(echo s7-$T-{uniform,eef,oracle}-n{512,1024,2048,4096,8192,16384}-s0) s7-$T-none-s0"
     sbatch --job-name="eval-s7-$T-headline" \
       --export=ALL,EXPRS_DIR=$SCRATCH/PointAct_exprs/robocasa365/stage7,\
EVAL_STEPS="30000",EVAL_SEEDS="11 13 17 19",NUM_TRIALS=100,RUNS="$ARMS" \
       experiments/13_robocasa365/eval_task_jeanzay.slurm
   done

Never pool across tasks when reading these back. summarize_stage_a.py has a cross-task
pooling bug on record, and stage 5 showed the sampler ordering differs BY TASK -- a curve
averaged over the three would hide the one effect this grid was extended to find.

Read the intermediate checkpoints as "where a 30K run was at step N", not as "a policy
trained for N steps" -- under cosine-to-30K the LR is still ~97% of peak at 5K and ~50% at
15K, so the early points understate their own budget. Only the 30K column is annealed.

Rehearse any new grid with SAVE_SUFFIX=-smoke first: both pooling consumers skip a results
directory whose trailing "-" token is not all digits, so smoke output is invisible to every
success rate. SAVE_SUFFIX=-2 would be pooled as step 2.

W&B is offline on Jean Zay compute nodes. From a LOGIN node:
   bash experiments/13_robocasa365/wandb_sync_jeanzay.sh
   python experiments/13_robocasa365/log_eval_to_wandb.py --run-dir <output_dir>
NEXT
