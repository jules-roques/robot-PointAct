#!/bin/bash
# Stage 1/3 of the RoboCasa365 -> LeRobot point-dataset pipeline: replay the source OpenDrawer
# demonstrations in sim and cache per-episode observations + fused point clouds.
# See data_prep/robocasa365_to_lerobot/README.md. Avoid H100 (RoboCasa365 sim misbehaves there);
# V100 is plenty since this stage does no model inference, just MuJoCo/EGL rendering.
# --resume skips already-cached episodes, so resubmitting this same job continues past the
# walltime limit.
#
#   sbatch experiments/13_robocasa365/data_prep_replay.slurm
#
#SBATCH --job-name=rc365-replay-od
#SBATCH --account=willow
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/jroques/logs/robocasa365/%x_%j.out
#SBATCH --error=/scratch/jroques/logs/robocasa365/%x_%j.out

set -euo pipefail

REPO=/home/jroques/code/robot-PointAct
cd "$REPO"

export MUJOCO_GL=egl
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export UV_OFFLINE=1
export LD_LIBRARY_PATH="/home/jroques/miniforge3/envs/ffmpeg-libs/lib:${LD_LIBRARY_PATH:-}"

TASKNAME=OpenDrawer
INPUT_DIR="${INPUT_DIR:-/scratch/jroques/robocasa_datasets/v1.0/target/atomic/${TASKNAME}/20250816}"

echo "host=$(hostname) input_dir=${INPUT_DIR}"
nvidia-smi -L

uv run --project envs/robocasa365 --no-sync \
    data_prep/robocasa365_to_lerobot/replay.py \
    --input-dir "${INPUT_DIR}/lerobot" \
    --cache-dir "${INPUT_DIR}/replay_cache" \
    --resume
