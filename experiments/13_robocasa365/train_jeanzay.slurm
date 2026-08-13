#!/bin/bash
# Jean Zay counterpart to train.slurm (CLEPS). Same payload script
# (train_pointact_concerto.sh), but Jean Zay and CLEPS are separate SLURM controllers, so
# #SBATCH account/partition/module directives cannot be shared between them — this file
# carries Jean Zay's. The account/qos below belong to one IDRIS project; override them on
# the command line (`sbatch --account=...`) rather than editing this file.
#
# Submit from the repository root (arch/h100 must be loaded in the submitting shell):
#   module load arch/h100
#   mkdir -p logs/robocasa365
#   sbatch --constraint=h100 experiments/13_robocasa365/train_jeanzay.slurm [concerto]
#
#SBATCH --job-name=rc365-train-od
#SBATCH --account=rgx@h100
#SBATCH --partition=gpu_p6
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --time=100:00:00
#SBATCH --output=logs/robocasa365/%x_%j.out
#SBATCH --error=logs/robocasa365/%x_%j.out

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

# SLURM runs the batch script from a spool copy, so the checkout is located via the submit
# directory instead (submit from the repository root), with an explicit override available.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

VARIANT="${1:-concerto}"
echo "host=$(hostname) variant=$VARIANT gpus=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

# accelerate (inside the train script) spawns the 4 GPU processes itself — no srun.
uv run --no-sync bash "experiments/13_robocasa365/train_pointact_${VARIANT}.sh"
