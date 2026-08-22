#!/bin/bash
# Where does the Stage 6 no-sampler arm's ~1.45 s/step actually go?
#
#   sbatch experiments/13_robocasa365/profile_stage6.slurm
#
# Two measurements on one H100 node, because the question only has an answer when both are
# on the table:
#
#   1. profile_dataloader.py -- per-stage CPU cost inside __getitem__ and the throughput a
#      real 8-worker pool sustains, at 4096 vs the unsampled budget.
#   2. pilot_throughput.py   -- marginal GPU s/step at the same two budgets, startup and
#      warm-up subtracted out.
#
# If (1) clears the per-process rate training needs and (2) shows the step time tracking the
# point count, the cost is the network and the dataloader is exonerated.
#
# dev QoS: this is ~20 minutes of work and dev turns around far faster than t3.
#
#SBATCH --job-name=s6-profile
#SBATCH --account=rgx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

set -euo pipefail

module purge
module load arch/h100

REPO="${SLURM_SUBMIT_DIR:-$WORK/code/robot-PointAct}"
POINTACT_ENV="${POINTACT_ENV:-$WORK/code/robot-PointAct}"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export UV_OFFLINE=1
export WANDB_MODE=disabled
export POINTACT_VLM_PATH="$DSDIR/HuggingFace_Models/Qwen/Qwen2.5-VL-3B-Instruct"
[ -e robot_data ] || ln -s "$SCRATCH/datasets/robot_data" robot_data

RUN=experiments/13_robocasa365/runs/s6-od-nosampler-s0.yaml
OUT="$SCRATCH/logs/robocasa365/s6_profile_${SLURM_JOB_ID}"
mkdir -p "$OUT"

echo "###############  1. dataloader  ###############"
uv run --project "$POINTACT_ENV" --no-sync \
    python experiments/13_robocasa365/profile_dataloader.py "$RUN" \
        --npoints 4096 8192 32768 --samples 128 --batches 40 --warmup 5

echo
echo "###############  2. gpu step time  ###############"
# per-device 16, not the base yaml's 32: 32 OOMs at the unsampled budget, and the two cells
# are only comparable at equal batch.
uv run --project "$POINTACT_ENV" --no-sync \
    python experiments/13_robocasa365/pilot_throughput.py \
        --contexts text_cache --npoints 4096 32768 \
        --per-device-batch 16 --effective-batch 128 \
        --short-steps 20 --long-steps 80 \
        --out "$OUT/pilot.json"
