#!/bin/bash
# Full PointAct training on RoboCasa365 OpenDrawer: 4 H100, effective batch 128.
# Auto-resumes from the run's output_dir (train_or_resume detects checkpoints), so
# resubmitting this same job continues past the walltime limit.
#
# CLEPS has no module system for cuda/ffmpeg (unlike Jean Zay) — torch's cu126 wheels bundle
# their own CUDA runtime, and ffmpeg's shared libs come from a dedicated conda env
# (`conda create -n ffmpeg-libs -c conda-forge ffmpeg=6.1`) via LD_LIBRARY_PATH below.
#
# Submit either:
#   sbatch --export=ALL,RUN_CONFIG=experiments/13_robocasa365/runs/od-eef-n4096-s0.yaml \
#          experiments/13_robocasa365/train.slurm          # ablation grid (one yaml per run)
#   sbatch experiments/13_robocasa365/train.slurm concerto # legacy per-variant payload script
#
#SBATCH --job-name=train-od
#SBATCH --account=willow
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/jroques/logs/robocasa365/%x_%j.out
#SBATCH --error=/scratch/jroques/logs/robocasa365/%x_%j.out

set -euo pipefail

export LD_LIBRARY_PATH="/home/jroques/miniforge3/envs/ffmpeg-libs/lib:${LD_LIBRARY_PATH:-}"

export NUM_NODES=1
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export UV_OFFLINE=1

# W&B offline: compute nodes have no guaranteed outbound access to the wandb API even though
# they can reach HF. Runs are written under WANDB_DIR; upload them afterwards from the login
# node with:  wandb sync $SCRATCH/wandb/offline-run-*
export WANDB_MODE=offline
export WANDB_DIR="$SCRATCH/wandb"
mkdir -p "$WANDB_DIR"

REPO="$HOME/code/robot-PointAct"
cd "$REPO"

echo "host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

# accelerate (inside the train script) spawns the 4 GPU processes itself — no srun.
if [ -n "${RUN_CONFIG:-}" ]; then
    # Ablation grid: everything about the run lives in the yaml.
    uv run --no-sync bash experiments/13_robocasa365/train.sh "$RUN_CONFIG"
else
    # Legacy path, kept so the pre-ablation baseline/eefdensity/roi runs remain reproducible.
    VARIANT="${1:-concerto}"
    echo "variant=$VARIANT"
    uv run --no-sync bash "experiments/13_robocasa365/train_pointact_${VARIANT}.sh"
fi
