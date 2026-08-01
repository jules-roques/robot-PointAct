#!/bin/bash
# ROI-guided-sampling PointAct training on RoboCASA OpenDrawer (4 H100, effective batch 128).
# Runs the WORKTREE code (roi_sampling + dataloader change) against the main checkout's
# training .venv. Auto-resumes from output_dir like the baseline train.slurm.
#
#   module load arch/h100
#   sbatch --constraint=h100 experiments/13_robocasa365/roi/train_roi.slurm
#
#SBATCH --job-name=train-od-roi
#SBATCH --account=rgx@h100
#SBATCH --partition=gpu_p6
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365_replay/%x_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365_replay/%x_%j.out

set -euo pipefail
module load arch/h100
module load ffmpeg/6.1.1

export NUM_NODES=1
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export UV_OFFLINE=1

export WANDB_MODE=offline
export WANDB_DIR="$SCRATCH/wandb"
mkdir -p "$WANDB_DIR"

MAIN_REPO=/lustre/fswork/projects/rech/rgx/unw24jl/code/robot-PointAct
WORKTREE="$MAIN_REPO/.claude/worktrees/roi-guided-sampling"
cd "$WORKTREE"

echo "host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-unset} cwd=$(pwd)"
nvidia-smi -L

# Code from the worktree (PYTHONPATH is prepended inside the train script via $(pwd)),
# training environment (.venv) from the main checkout via `uv run --project`.
uv run --project "$MAIN_REPO" --no-sync bash "experiments/13_robocasa365/train_pointact_concerto_roi.sh"
