#!/bin/bash
# Convert the RoboCasa365 replay cache into a LeRobot dataset + point-cloud LMDB.
# Stage 2 of data_prep/robocasa365_to_lerobot. Runs in the pointact (root) env, CPU only.
#
# Submit:
#   sbatch data_prep/robocasa365_to_lerobot/convert.slurm
#
# This is the Jean Zay variant (module system, IDRIS account/qos); the CLEPS equivalent is
# experiments/13_robocasa365/data_prep_convert.slurm, which also runs the stats step. Submit
# from the repository root (`mkdir -p logs/robocasa365` first).
#
#SBATCH --job-name=rc365-convert
#SBATCH --account=rgx@cpu
#SBATCH --partition=cpu_p1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_cpu-t3
#SBATCH --time=10:00:00
#SBATCH --output=logs/robocasa365/%x_%j.out
#SBATCH --error=logs/robocasa365/%x_%j.out

set -euo pipefail

module purge
module load ffmpeg/6.1.1   # LeRobot video encoding

# Recommended by data_prep/robocasa365_to_lerobot/README.md
export SVT_LOG=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=TRUE
export HDF5_USE_FILE_LOCKING=FALSE

# Compute nodes have no network: keep uv/HF fully offline.
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_OFFLINE=1

# SLURM runs the batch script from a spool copy, so the checkout is located via the submit
# directory instead (submit from the repository root), with an explicit override available.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
# Must match DATASET_BASE_PATH in robocasa/macros_private.py (see envs/robocasa365/README.md).
ROBOCASA_DATASET_ROOT="${ROBOCASA_DATASET_ROOT:-$SCRATCH/datasets/robot_data/robocasa365}"
BASE="${BASE:-$ROBOCASA_DATASET_ROOT/v1.0/target/atomic/OpenDrawer/20250816}"
OUT="${OUT:-$ROBOCASA_DATASET_ROOT/lerobot_point_lmdb/OpenDrawer}"

cd "$REPO"
echo "host=$(hostname) cache=$BASE/replay_cache out=$OUT episodes=${EPISODES:+subset}"

# EPISODES (optional, space-separated) restricts conversion to a subset — e.g. only the
# successful replays. Unset converts every cached episode.
srun uv run --no-sync \
  data_prep/robocasa365_to_lerobot/convert.py \
  --cache-dir "$BASE/replay_cache" \
  --output-dir "$OUT" \
  --repo-id OpenDrawer \
  --overwrite \
  ${EPISODES:+--episodes $EPISODES}
