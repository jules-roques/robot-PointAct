#!/bin/bash
# Jean Zay counterpart to eval_grid.slurm: evaluate one ablation run at 10/20/30/40/50K as an
# array, one task per checkpoint.
#
#   sbatch --export=ALL,RUN=od-eef-n4096-s0 experiments/13_robocasa365/eval_grid_jeanzay.slurm
#
# A100 (rgx@a100, gpu_p5, qos_gpu_a100-t3): PTv3 uses FlashAttention so V100 is out, and
# RoboCasa365 does not render correctly on H100. Note there is no qos_gpu_a100-t4 -- a100 only
# has dev and t3, which is the likeliest cause of the "Invalid account/partition" once recorded
# in docs/clusters/jean-zay.md.
#
# Array tasks whose checkpoint does not exist yet exit 0, so this can be submitted with
# --dependency=afterany on the training job and resubmitted later for the stragglers.
#
# Walltime is 2h against a measured worst case of 1h15. It was 12h, which reserved 10x the
# job's actual need and made it invisible to backfill.
#
# Measured per 100-trial checkpoint: OpenDrawer ~1h00-1h12, PickPlaceCounterToStove ~1h03,
# TurnOnMicrowave ~42min. Note the policy calls the server once per replan_steps (8), NOT once
# per env step -- an earlier version of this comment claimed otherwise and used it to argue the
# stage-0 runs were far more expensive than the image-free ones. They are not: stage-0
# (live VLM) OpenDrawer ran 1h00-1h04 against ~1h12 image-free, because wall-clock here is
# dominated by episode LENGTH, and a more successful policy terminates earlier. Do not read a
# model's inference cost off these timings without controlling for success rate.
#
# 2h is also the qos_gpu_a100-dev ceiling, so this file can be sent to dev unmodified -- worth
# doing, since dev has priority 80 against t3's 50:
#
#   sbatch --qos=qos_gpu_a100-dev --export=ALL,RUN=tom-eef-n4096-s0 <this file>
#
# But dev caps submitted jobs at 10 per user and array elements count individually, so at most
# two 5-checkpoint arrays fit at once; a fuller sweep has to go to t3 or be submitted in waves.
#
#SBATCH --job-name=evalgrid
#SBATCH --account=rgx@a100
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --output=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%A_%a.out
#SBATCH --error=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%A_%a.out

set -euo pipefail

module load ffmpeg/6.1.1

REPO="${SLURM_SUBMIT_DIR:-$WORK/code/robot-PointAct}"
cd "$REPO"

RUN="${RUN:?set RUN=<run name, e.g. od-eef-n4096-s0> via --export}"
EXPRS_DIR="${EXPRS_DIR:-$SCRATCH/PointAct_exprs/robocasa365/ablation}"
CKPT_DIR="$EXPRS_DIR/$RUN"

# 100 trials at every checkpoint -- see eval_grid.slurm for why the 50/50/50/150 split was
# dropped. Overridable so a checkpoint outside the default grid can be evaluated without
# editing this file, e.g. EVAL_STEPS="15000" EVAL_TRIALS="50" --array=0.
read -r -a STEPS <<< "${EVAL_STEPS:-10000 20000 30000 40000 50000}"
read -r -a TRIALS <<< "${EVAL_TRIALS:-100 100 100 100 100}"
CKPT_STEP="${STEPS[$SLURM_ARRAY_TASK_ID]}"
# Overridable so a smoke test can ask for a handful of trials; the client only
# writes results at the end, so a long default would produce nothing before a
# short walltime killed it.
export NUM_TRIALS="${NUM_TRIALS:-${TRIALS[$SLURM_ARRAY_TASK_ID]}}"

# Capture rollout animations at the final checkpoint only. They cost buffering and render
# time, and the figure worth keeping is the one from the policy the headline number describes.
if [ "$CKPT_STEP" = "50000" ]; then
    export VIZ_ROLLOUTS="${VIZ_ROLLOUTS:-1}"
fi

if [ ! -d "$CKPT_DIR/checkpoint-$CKPT_STEP" ]; then
    echo "checkpoint-$CKPT_STEP not written yet under $CKPT_DIR; nothing to do."
    exit 0
fi

# Task and the text-context cache both come from the run's own archived data config, so an
# eval can never silently disagree with the training it measures.
# `root` in the data config is repo-root-relative (the robot_data symlink), so the absolute
# paths are just $REPO/<root>/<repo_id>/...
DATA_CONFIG="$CKPT_DIR/data_config.yaml"
read -r ENV_NAME TEXT_CTX <<EOF
$($WORK/code/robot-PointAct/.venv/bin/python -c "
import yaml
ds = yaml.safe_load(open('$DATA_CONFIG'))['lerobot_datasets'][0]
cache = ds.get('text_context_file')
print(ds['repo_id'],
      f\"$REPO/{ds['root']}/{ds['repo_id']}/{cache}\" if cache else '')")
EOF

echo "host=$(hostname) run=$RUN ckpt=$CKPT_STEP env=$ENV_NAME trials=$NUM_TRIALS"
nvidia-smi -L

# The molmo arms carry a third process (MolmoPoint, ~0.78 s per replan) and take roughly twice
# as long as the arm they are compared against -- about 2h20 on OpenDrawer, more on the
# two-query PPCS objpan arm. The default 2h is sized for the other arms, so refuse to start
# rather than time out at trial 80: the client's partial dump does NOT match the
# per_trial_seed*_n*.json glob the pooling reads, so a killed run contributes nothing at all.
if grep -qiE '^\s*molmo_sampling:\s*true' "$DATA_CONFIG"; then
    LEFT=$(squeue -h -j "${SLURM_JOB_ID}" -o %L 2>/dev/null | tr -d ' ')
    LEFT_MIN=$(python3 -c "
import sys
t = sys.argv[1] if len(sys.argv) > 1 else ''
if not t:
    print(-1); raise SystemExit
d, _, rest = t.partition('-')
if rest:
    days, t = int(d), rest
else:
    days, t = 0, t
p = [int(x) for x in t.split(':')]
while len(p) < 3:
    p.insert(0, 0)
print(days * 1440 + p[0] * 60 + p[1] + (1 if p[2] else 0))
" "$LEFT")
    MIN_NEEDED="${MOLMO_MIN_MINUTES:-200}"
    if [ "$LEFT_MIN" -ge 0 ] && [ "$LEFT_MIN" -lt "$MIN_NEEDED" ]; then
        echo "ERROR: $RUN is a MolmoPoint arm and only ${LEFT_MIN} min of walltime remain." >&2
        echo "  Resubmit with --time=05:00:00 (or set MOLMO_MIN_MINUTES to override)." >&2
        exit 1
    fi
    echo "molmo arm: ${LEFT_MIN} min of walltime available"
fi

export TEXT_CONTEXT_FILE="$TEXT_CTX"
bash experiments/13_robocasa365/eval_robocasa365.sh "$ENV_NAME" "$CKPT_DIR" "$CKPT_STEP" "rot6d" "${OPTS:-}"
