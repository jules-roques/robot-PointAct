#!/bin/bash
# Convert the RoboCasa365 replay cache into a LeRobot dataset + point-cloud LMDB.
# Stage 2 of data_prep/robocasa365_to_lerobot. Runs in the pointact (root) env, CPU only.
#
# Submit:
#   sbatch data_prep/robocasa365_to_lerobot/convert.slurm
#
#SBATCH --job-name=rc365-convert
#SBATCH --account=rgx@cpu
#SBATCH --partition=cpu_p1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_cpu-t3
#SBATCH --time=10:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365_replay/%x_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365_replay/%x_%j.out

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

REPO=/lustre/fswork/projects/rech/rgx/unw24jl/code/robot-PointAct
BASE="$SCRATCH/datasets/robot_data/robocasa365/v1.0/target/atomic/OpenDrawer/20250816"
OUT="$SCRATCH/datasets/robot_data/robocasa365/lerobot_point_lmdb/OpenDrawer"

cd "$REPO"
echo "host=$(hostname) cache=$BASE/replay_cache out=$OUT"

srun uv run --no-sync \
  data_prep/robocasa365_to_lerobot/convert.py \
  --cache-dir "$BASE/replay_cache" \
  --output-dir "$OUT" \
  --repo-id OpenDrawer \
  --overwrite
