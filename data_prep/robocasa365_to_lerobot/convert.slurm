#!/bin/bash
# Convert the RoboCasa365 replay cache into a LeRobot dataset + point-cloud LMDB.
# Stage 2 of data_prep/robocasa365_to_lerobot. Runs in the pointact (root) env, CPU only.
#
# Submit:
#   sbatch data_prep/robocasa365_to_lerobot/convert.slurm
# Submit (another task — must match the TASK/DATE the replay stage used):
#   sbatch --export=ALL,TASK=CloseBlenderLid,DATE=20250816 \
#          data_prep/robocasa365_to_lerobot/convert.slurm
#
# This REWRITES $OUT, including the point LMDB. After the 2026-09-21 purge the old LMDBs
# survived while their parquet did not; they are not reusable as a shortcut, because the
# dataset's episode numbering is assigned here and must match the LMDB keys
# (see episode_index_map.py). Convert the whole task, not a subset, unless you know why.
#
#SBATCH --job-name=convert
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

# REPO is the code that runs, POINTACT_ENV the checkout owning the root .venv. See the same
# split in replay.slurm: a worktree has no .venv of its own, and the branch's convert.py is
# not the main checkout's.
REPO="${REPO:-/lustre/fswork/projects/rech/rgx/unw24jl/code/robot-PointAct}"
POINTACT_ENV="${POINTACT_ENV:-/lustre/fswork/projects/rech/rgx/unw24jl/code/robot-PointAct}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# TASK/DATE match replay.slurm's; see the note there. Pass the SAME pair to both stages or
# this converts one task's cache into another task's output directory.
TASK="${TASK:-OpenDrawer}"
DATE="${DATE:-20250816}"
BASE="$SCRATCH/datasets/robot_data/robocasa365/v1.0/target/atomic/$TASK/$DATE"
# VOXEL_SIZE (m) selects the render grid. Anything but the 1 cm default lands in its own
# `_g<mm>` cache and dataset root, derived here rather than left to the caller: convert.slurm
# runs --overwrite, so a finer grid pointed at the default paths would rebuild over the 1 cm
# data every existing run trains on.
VOXEL_SIZE="${VOXEL_SIZE:-0.01}"
GRID_SUFFIX=$(python3 -c "v=float('$VOXEL_SIZE'); print('' if abs(v-0.01)<1e-12 else '_g%gmm' % (v*1000))")
CACHE_DIR="$BASE/replay_cache$GRID_SUFFIX"
OUT="$SCRATCH/datasets/robot_data/robocasa365/lerobot_point_lmdb$GRID_SUFFIX/$TASK"

if [ ! -d "$CACHE_DIR" ]; then
    echo "ERROR: no replay cache at $CACHE_DIR -- run replay.slurm (same VOXEL_SIZE) first" >&2
    exit 1
fi
# The cache records the grid it was rendered at; refuse to file it under another one.
CACHE_GRID=$(python3 -c "import json; print(json.load(open('$CACHE_DIR/cache_meta.json'))['voxel_size'])")
if ! python3 -c "import sys; sys.exit(abs($CACHE_GRID - $VOXEL_SIZE) > 1e-12)"; then
    echo "ERROR: $CACHE_DIR was rendered at voxel_size=$CACHE_GRID, not VOXEL_SIZE=$VOXEL_SIZE" >&2
    exit 1
fi

cd "$REPO"
echo "host=$(hostname) cache=$CACHE_DIR out=$OUT episodes=${EPISODES:+subset}"

# EPISODES (optional, space-separated) restricts conversion to a subset — e.g. only the
# successful replays. Unset converts every cached episode.
srun uv run --project "$POINTACT_ENV" --no-sync \
  data_prep/robocasa365_to_lerobot/convert.py \
  --cache-dir "$CACHE_DIR" \
  --output-dir "$OUT" \
  --repo-id "$TASK" \
  --overwrite \
  ${EPISODES:+--episodes $EPISODES}
