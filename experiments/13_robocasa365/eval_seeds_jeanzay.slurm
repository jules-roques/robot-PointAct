#!/bin/bash
# Evaluate ONE checkpoint at N seeds and pool them -- the stage-5 counterpart to
# eval_grid_jeanzay.slurm, which arrays over checkpoints instead.
#
#   sbatch --export=ALL,RUN=s5-tom-eef-n8192-s0 experiments/13_robocasa365/eval_seeds_jeanzay.slurm
#
# Stage 5 buys precision rather than a curve: every arm is scored once, at its final
# checkpoint, over 500 rollouts. At n=100 the Wilson half-width near 50% is ~10pp, which is
# wider than most of the differences this campaign has been reporting; at n=500 it is ~4.4pp.
#
# 500 rollouts is five SEEDS of 100, not one 500-trial run, for two reasons. The seeded scene
# stream is deterministic, so re-running a seed replays the same kitchens and buys nothing --
# see eval_robocasa365.sh:25. And five independent array elements finish in the wall-clock of
# one, where a single 500-trial job would want 5x the walltime and queue behind everything.
# The results land in per_trial_seed<S>_n100.json, one file per seed, which is exactly the
# glob summarize_stage_a.py and log_eval_to_wandb.py already pool over.
#
# Seeds are fixed and stated so a rerun cannot silently choose different ones. 7 is the
# historical default and is kept so stage-5 numbers can be compared against earlier stages at
# that seed alone if that is ever wanted.
#
# A100 (rgx@a100, gpu_p5, qos_gpu_a100-t3): PTv3 uses FlashAttention so V100 is out, and
# RoboCasa365 does not render correctly on H100.
#
#SBATCH --job-name=evalseeds
#SBATCH --account=rgx@a100
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_a100-t3
# 6 h, not the grid's 2 h: the molmo arms carry a third process and ran 2h20-4h10 per 100
# trials in stage 3. The guard below refuses to start a molmo arm with less than 200 min left
# rather than letting it die at trial 80 -- a killed run's partial dump does not match the
# pooling glob, so it contributes nothing at all.
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --output=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%A_%a.out
#SBATCH --error=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%A_%a.out

set -euo pipefail

module load ffmpeg/6.1.1

REPO="${SLURM_SUBMIT_DIR:-$WORK/code/robot-PointAct}"
cd "$REPO"

RUN="${RUN:?set RUN=<run name, e.g. s5-tom-eef-n8192-s0> via --export}"
EXPRS_DIR="${EXPRS_DIR:-$SCRATCH/PointAct_exprs/robocasa365/stage5}"
CKPT_DIR="$EXPRS_DIR/$RUN"

CKPT_STEP="${CKPT_STEP:-30000}"
read -r -a SEEDS <<< "${EVAL_SEEDS:-7 11 13 17 19}"
export SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
export NUM_TRIALS="${NUM_TRIALS:-100}"

# Rollout animations from one seed only. They cost buffering and render time, and five copies
# of the same arm's behaviour is four more than anyone looks at.
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
    export VIZ_ROLLOUTS="${VIZ_ROLLOUTS:-1}"
fi

if [ ! -d "$CKPT_DIR/checkpoint-$CKPT_STEP" ]; then
    echo "checkpoint-$CKPT_STEP not written yet under $CKPT_DIR; nothing to do."
    exit 0
fi

# Task and the text-context cache both come from the run's own archived data config, so an
# eval can never silently disagree with the training it measures.
DATA_CONFIG="$CKPT_DIR/data_config.yaml"
read -r ENV_NAME TEXT_CTX <<EOF
$($WORK/code/robot-PointAct/.venv/bin/python -c "
import yaml
ds = yaml.safe_load(open('$DATA_CONFIG'))['lerobot_datasets'][0]
cache = ds.get('text_context_file')
print(ds['repo_id'],
      f\"$REPO/{ds['root']}/{ds['repo_id']}/{cache}\" if cache else '')")
EOF

echo "host=$(hostname) run=$RUN ckpt=$CKPT_STEP env=$ENV_NAME seed=$SEED trials=$NUM_TRIALS"
nvidia-smi -L

if grep -qiE '^\s*molmo_sampling:\s*true' "$DATA_CONFIG"; then
    # Minutes of walltime left, from the epoch timestamp Slurm exports. Fails OPEN (-1, no
    # check), because this guard exists to catch a too-short --time and must never itself be
    # why a correctly-submitted job dies.
    LEFT_MIN=-1
    if [ -n "${SLURM_JOB_END_TIME:-}" ]; then
        LEFT_MIN=$(( (SLURM_JOB_END_TIME - $(date +%s)) / 60 ))
    fi
    MIN_NEEDED="${MOLMO_MIN_MINUTES:-200}"
    if [ "$LEFT_MIN" -lt 0 ]; then
        echo "molmo arm: walltime check skipped (SLURM_JOB_END_TIME unset)"
    elif [ "$LEFT_MIN" -lt "$MIN_NEEDED" ]; then
        echo "ERROR: $RUN is a MolmoPoint arm and only ${LEFT_MIN} min of walltime remain." >&2
        echo "  Resubmit with --time=06:00:00 (or set MOLMO_MIN_MINUTES to override)." >&2
        exit 1
    else
        echo "molmo arm: ${LEFT_MIN} min of walltime available"
    fi
fi

export TEXT_CONTEXT_FILE="$TEXT_CTX"
bash experiments/13_robocasa365/eval_robocasa365.sh "$ENV_NAME" "$CKPT_DIR" "$CKPT_STEP" "rot6d" "${OPTS:-}"
