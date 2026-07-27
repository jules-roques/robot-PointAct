#!/bin/bash
# Full PointAct training on RoboCasa365 OpenDrawer: 4 H100, effective batch 128.
# Auto-resumes from the run's output_dir (train_or_resume detects checkpoints), so
# resubmitting this same job continues past the walltime limit.
#
# CLEPS has no module system for cuda/ffmpeg (unlike Jean Zay) — torch's cu126 wheels bundle
# their own CUDA runtime, and ffmpeg's shared libs come from a dedicated conda env
# (`conda create -n ffmpeg-libs -c conda-forge ffmpeg=6.1`) via LD_LIBRARY_PATH below.
#
# Submit:
#   sbatch experiments/13_robocasa365/train.slurm [concerto|concerto_eefdensity]
#
#SBATCH --job-name=rc365-train-od
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

REPO=/home/jroques/code/robot-PointAct
cd "$REPO"

VARIANT="${1:-concerto}"
echo "host=$(hostname) variant=$VARIANT gpus=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

# accelerate (inside the train script) spawns the 4 GPU processes itself — no srun.
uv run --no-sync bash "experiments/13_robocasa365/train_pointact_${VARIANT}.sh"
