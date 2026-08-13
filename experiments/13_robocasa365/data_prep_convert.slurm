#!/bin/bash
# Stages 2-3 of the RoboCasa365 -> LeRobot point-dataset pipeline: pack the replay cache from
# data_prep_replay.slurm into the final LeRobot dataset + points_3views LMDB, then compute the
# state/action normalization stats training needs. CPU-only (no model inference), runs in the
# pointact (root) env.
#
# Submit from the repository root (`mkdir -p logs/robocasa365` first). The account/partition
# directives are CLEPS-specific — override on the command line rather than editing this file.
#
#   sbatch experiments/13_robocasa365/data_prep_convert.slurm
#
#SBATCH --job-name=rc365-convert-od
#SBATCH --account=willow
#SBATCH --partition=cpu_homogen
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/robocasa365/%x_%j.out
#SBATCH --error=logs/robocasa365/%x_%j.out

set -euo pipefail

# SLURM runs the batch script from a spool copy, so the checkout is located via the submit
# directory instead (submit from the repository root), with an explicit override available.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

export SVT_LOG=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=TRUE
export HDF5_USE_FILE_LOCKING=FALSE
# libav* for torchcodec; see envs/robocasa365/README.md. Not needed where ffmpeg is a module.
FFMPEG_LIB_DIR="${FFMPEG_LIB_DIR:-$HOME/miniforge3/envs/ffmpeg-libs/lib}"
export LD_LIBRARY_PATH="$FFMPEG_LIB_DIR:${LD_LIBRARY_PATH:-}"

TASKNAME="${TASKNAME:-OpenDrawer}"
# Must match DATASET_BASE_PATH in robocasa/macros_private.py (see envs/robocasa365/README.md).
ROBOCASA_DATASET_ROOT="${ROBOCASA_DATASET_ROOT:-$SCRATCH/datasets/robot_data/robocasa365}"
INPUT_DIR="${INPUT_DIR:-$ROBOCASA_DATASET_ROOT/v1.0/target/atomic/${TASKNAME}/20250816}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/datasets/robot_data/robocasa365/lerobot_point_lmdb/${TASKNAME}}"

echo "host=$(hostname) input_dir=${INPUT_DIR} output_dir=${OUTPUT_DIR}"

uv run --no-sync \
    data_prep/robocasa365_to_lerobot/convert.py \
    --cache-dir "${INPUT_DIR}/replay_cache" \
    --output-dir "${OUTPUT_DIR}" \
    --repo-id "${TASKNAME}" \
    --image-writer-threads 12

# Stage 3/3: state/action normalization stats (see experiments/13_robocasa365/README.md
# "State / action statistics" — --point_cloud_dir and --replace_zero_std are mandatory here).
uv run --no-sync \
    data_prep/prepare_robot_state_action_stats.py \
    --dataset_dirs "${OUTPUT_DIR}" \
    --output_file "${OUTPUT_DIR}/robot_state_action_stats/rot6d.json" \
    --point_cloud_dir points_3views \
    --state_xyz_slice 0 3 --action_xyz_slice 0 3 \
    --state_rotation_slice 3 7 --action_rotation_slice 3 7 \
    --rotation_type quat --target_rotation_type rot6d \
    --replace_zero_std
