#!/bin/bash
# Stage 1/3 of the RoboCasa365 -> LeRobot point-dataset pipeline: replay the source OpenDrawer
# demonstrations in sim and cache per-episode observations + fused point clouds.
# See data_prep/robocasa365_to_lerobot/README.md. Avoid H100 (RoboCasa365 sim misbehaves there);
# V100 is plenty since this stage does no model inference, just MuJoCo/EGL rendering.
# --resume skips already-cached episodes, so resubmitting this same job continues past the
# walltime limit.
#
# Submit from the repository root (`mkdir -p logs/robocasa365` first). The account/partition
# directives are CLEPS-specific — override on the command line rather than editing this file.
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
#SBATCH --output=logs/robocasa365/%x_%j.out
#SBATCH --error=logs/robocasa365/%x_%j.out

set -euo pipefail

# SLURM runs the batch script from a spool copy, so the checkout is located via the submit
# directory instead (submit from the repository root), with an explicit override available.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

export MUJOCO_GL=egl
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export UV_OFFLINE=1
# libav* for torchcodec; see envs/robocasa365/README.md. Not needed where ffmpeg is a module.
FFMPEG_LIB_DIR="${FFMPEG_LIB_DIR:-$HOME/miniforge3/envs/ffmpeg-libs/lib}"
export LD_LIBRARY_PATH="$FFMPEG_LIB_DIR:${LD_LIBRARY_PATH:-}"

TASKNAME="${TASKNAME:-OpenDrawer}"
# Must match DATASET_BASE_PATH in robocasa/macros_private.py (see envs/robocasa365/README.md).
# The trailing date component is per task and comes from robocasa's dataset registry.
ROBOCASA_DATASET_ROOT="${ROBOCASA_DATASET_ROOT:-$SCRATCH/datasets/robot_data/robocasa365}"
INPUT_DIR="${INPUT_DIR:-$ROBOCASA_DATASET_ROOT/v1.0/target/atomic/${TASKNAME}/20250816}"

echo "host=$(hostname) input_dir=${INPUT_DIR}"
nvidia-smi -L

uv run --project envs/robocasa365 --no-sync \
    data_prep/robocasa365_to_lerobot/replay.py \
    --input-dir "${INPUT_DIR}/lerobot" \
    --cache-dir "${INPUT_DIR}/replay_cache" \
    --resume
