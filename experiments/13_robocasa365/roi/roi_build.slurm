#!/bin/bash
# Stage B: build the per-point ROI-flag LMDB with YOLO-World (.venv-roi, A100).
# A100 (not V100) because the ultralytics-pulled torch is a CUDA-13 build that no longer
# supports Volta (sm_70).
#   sbatch experiments/13_robocasa365/roi/roi_build.slurm            # all episodes
#   sbatch --export=ALL,EPISODES="0 1 2" --qos=qos_gpu_a100-dev --time=00:30:00 \
#          experiments/13_robocasa365/roi/roi_build.slurm            # dev subset
#
#SBATCH --job-name=rc365-roi-build
#SBATCH --account=rgx@a100
#SBATCH --partition=gpu_p5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --time=06:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365_replay/%x_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365_replay/%x_%j.out

set -euo pipefail
module purge
module load ffmpeg/6.1.1

REPO=/lustre/fswork/projects/rech/rgx/unw24jl/code/robot-PointAct/.claude/worktrees/roi-guided-sampling
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

VENV="$REPO/.venv-roi/bin/python"
DATA="$SCRATCH/datasets/robot_data/robocasa365/lerobot_point_lmdb/OpenDrawer"
CALIB="$DATA/roi_meta/camera_calib.npz"
WEIGHTS="$SCRATCH/models/YOLO-World/yoloworld_s_drawerhandle.pt"

nvidia-smi -L || true

srun "$VENV" data_prep/roi_sampling/build_roi_cache.py \
  --dataset-dir "$DATA" \
  --calib "$CALIB" \
  --weights "$WEIGHTS" \
  --out-dirname points_3views_roi \
  ${EPISODES:+--episodes $EPISODES}
