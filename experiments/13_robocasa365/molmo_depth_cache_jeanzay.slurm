#!/bin/bash
# Merge the depth-anchor shards and write them out as an LMDB anchor cache (CPU, root env).
#
#   sbatch --export=ALL,TASK=TurnOnMicrowave experiments/13_robocasa365/molmo_depth_cache_jeanzay.slurm
#
# Run this after molmo_depth_anchors_jeanzay.slurm's array has completed. It is CPU-only --
# no simulator, no model -- but it is NOT a login-node job: writing the cache holds every
# record in memory before the LMDB transaction commits and the login node kills it (exit 137)
# partway through, leaving a half-written cache directory behind. Ask for a compute node.
#
#SBATCH --job-name=molmo-depth-cache
#SBATCH --account=rgx@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-dev
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%j.out

set -euo pipefail

MAIN_REPO=/lustre/fswork/projects/rech/rgx/unw24jl/code/robot-PointAct
REPO="${SLURM_SUBMIT_DIR:-$MAIN_REPO}"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"   # shadow the installed pointact with the worktree

TASK="${TASK:?set TASK=<OpenDrawer|TurnOnMicrowave> via --export}"
DATA_ROOT="${DATA_ROOT:-$SCRATCH/datasets/robot_data/robocasa365/lerobot_point_lmdb}"
DATASET="$DATA_ROOT/$TASK"
OUT_DIR="$DATASET/roi_meta"

OUT_NAME="${OUT_NAME:-depth_anchors}"
MERGED="${MERGED:-$OUT_DIR/${OUT_NAME}.npz}"
CACHE_DIRNAME="${CACHE_DIRNAME:-points_3views_molmo_depth}"
# Which estimator in the merged npz becomes the cache: xyz (the --point-window median),
# xyz_w1 (the same median over a fixed 3x3), or xyz_near. One merged file holds all of them,
# so switching arms costs a cache write and not another replay.
CACHE_FIELD="${CACHE_FIELD:-xyz}"

PY="$MAIN_REPO/.venv/bin/python"

echo "host=$(hostname) task=$TASK merged=$MERGED cache=$CACHE_DIRNAME"

# Merge is idempotent; skip it if a merged file is already in place and newer than the shards.
# SKIP_MERGE=1 when the merged file is the input and its shards are long gone (the stage-5
# best-view selection reads the per-view merge that stage 3 produced).
if [ -z "${SKIP_MERGE:-}" ]; then
    "$PY" -m data_prep.roi_sampling.dump_molmo_depth_anchors \
        --dataset-dir "$DATASET" \
        --merge "$OUT_DIR/${OUT_NAME}_shard*.npz" \
        --out "$MERGED"
fi

# SELECT_GT=1 collapses each query's per-view centres to the one nearest the ground truth --
# the stage-5 privileged upper bound on the pointer. Pure numpy over rows that already exist,
# so it costs minutes and no GPU; see dump_molmo_depth_anchors.select_closest_gt.
if [ -n "${SELECT_GT:-}" ]; then
    GT_NPZ="${GT_NPZ:-$OUT_DIR/target_positions.npz}"
    [ -f "$GT_NPZ" ] || { echo "ERROR: no $GT_NPZ; run target_positions_jeanzay.slurm for $TASK first" >&2; exit 1; }
    SELECTED="${SELECTED:-$OUT_DIR/${OUT_NAME}_bestgt.npz}"
    "$PY" -m data_prep.roi_sampling.dump_molmo_depth_anchors \
        --dataset-dir "$DATASET" \
        --task "$TASK" \
        --select-closest-gt "$MERGED" \
        --gt-npz "$GT_NPZ" \
        --out "$SELECTED"
    MERGED="$SELECTED"
fi

# A partial cache from a killed run is not resumable -- LMDB will happily open it and serve
# the records that made it in. Clear it so a failure here is a missing cache, not a short one.
rm -rf "$DATASET/$CACHE_DIRNAME"

"$PY" -m data_prep.roi_sampling.dump_molmo_depth_anchors \
    --dataset-dir "$DATASET" \
    --write-cache "$MERGED" \
    --out-dirname "$CACHE_DIRNAME" \
    --cache-field "$CACHE_FIELD"
