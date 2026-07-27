#!/bin/bash
# Stage B: build the per-point ROI-flag LMDB with YOLO-World (.venv-roi, V100).
#   sbatch experiments/13_robocasa365/roi/roi_build.slurm            # all episodes
#   sbatch --export=ALL,EPISODES="0 1 2" --qos=qos_gpu-dev --time=00:30:00 \
#          experiments/13_robocasa365/roi/roi_build.slurm            # dev subset
#
#SBATCH --job-name=rc365-roi-build
#SBATCH --account=rgx@v100
#SBATCH --partition=gpu_p13
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu-t3
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
