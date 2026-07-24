#!/bin/bash
# Full PointAct training on RoboCasa365 OpenDrawer: 2 H100, effective batch 128.
# Auto-resumes from the run's output_dir (train_or_resume detects checkpoints), so
# resubmitting this same job continues past the QoS wall.
#
# Submit (arch/h100 must be loaded in the submitting shell):
#   module load arch/h100
#   sbatch --constraint=h100 experiments/13_robocasa365/train.slurm [concerto|utonia]
#
#SBATCH --job-name=rc365-train-od
#SBATCH --account=rgx@h100
#SBATCH --partition=gpu_p6
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --time=100:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365_replay/%x_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365_replay/%x_%j.out

set -euo pipefail
# Keep the arch/h100 MODULEPATH (do NOT purge) so ffmpeg's libav* resolves for torchcodec.
module load arch/h100
module load ffmpeg/6.1.1

export NUM_NODES=1
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export UV_OFFLINE=1

# W&B offline: compute nodes have no internet. Runs are written under WANDB_DIR; upload them
# afterwards from a login node with:  wandb sync $SCRATCH/wandb/offline-run-*
export WANDB_MODE=offline
export WANDB_DIR="$SCRATCH/wandb"
mkdir -p "$WANDB_DIR"

REPO=/lustre/fswork/projects/rech/rgx/unw24jl/code/robot-PointAct
cd "$REPO"

VARIANT="${1:-concerto}"
echo "host=$(hostname) variant=$VARIANT gpus=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

# accelerate (inside the train script) spawns the 2 GPU processes itself — no srun.
uv run --no-sync bash "experiments/13_robocasa365/train_pointact_${VARIANT}.sh"
