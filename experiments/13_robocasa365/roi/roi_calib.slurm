#!/bin/bash
# Stage A: dump RoboCASA left/right camera calibration (robocasa365 env, V100/EGL).
#   sbatch experiments/13_robocasa365/roi/roi_calib.slurm
#
#SBATCH --job-name=rc365-roi-calib
#SBATCH --account=rgx@v100
#SBATCH --partition=gpu_p13
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu-dev
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365_replay/%x_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365_replay/%x_%j.out

set -euo pipefail
module purge
module load ffmpeg/6.1.1
export MUJOCO_GL=egl
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_OFFLINE=1

REPO=/lustre/fswork/projects/rech/rgx/unw24jl/code/robot-PointAct/.claude/worktrees/roi-guided-sampling
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"   # shadow main-checkout pointact with the worktree

BASE="$SCRATCH/datasets/robot_data/robocasa365/v1.0/target/atomic/OpenDrawer/20250816"
OUT="$SCRATCH/datasets/robot_data/robocasa365/lerobot_point_lmdb/OpenDrawer/roi_meta/camera_calib.npz"

mkdir -p "$(dirname "$OUT")"
nvidia-smi -L || true

srun uv run --project envs/robocasa365 --no-sync \
  python data_prep/roi_sampling/dump_camera_calib.py \
  --source-dir "$BASE/lerobot" \
  --out "$OUT" \
  --num-episodes 8
