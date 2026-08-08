#!/bin/bash
# Re-lift an existing MolmoPoint cache at new lift/fusion parameters (CPU, root env).
#
#   sbatch --export=ALL,TASK=TurnOnMicrowave,OUT_DIRNAME=points_3views_molmo_lonewrist,\
#EXTRA_ARGS="--allow-lone-wrist" experiments/13_robocasa365/molmo_relift_jeanzay.slurm
#
# No GPU: the 8B forwards were paid once at build time and the pixels are in the cache.
# Run this on a compute node, not the login node -- the login node kills it partway
# through the load phase, which looks like a crash in the tool and is not.
#
# NOTE: relift_molmo_cache defaults to --point-window 1 while the caches were BUILT at 2.
# Pass the window explicitly or the comparison silently changes two things at once.
#
#SBATCH --job-name=molmo-relift
#SBATCH --account=rgx@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-dev
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%j.out

set -euo pipefail

MAIN_REPO=/lustre/fswork/projects/rech/rgx/unw24jl/code/robot-PointAct
REPO="${SLURM_SUBMIT_DIR:-$MAIN_REPO}"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

TASK="${TASK:?set TASK=<task> via --export}"
DATA_ROOT="${DATA_ROOT:-$SCRATCH/datasets/robot_data/robocasa365/lerobot_point_lmdb}"
DATASET="$DATA_ROOT/$TASK"

ARGS=(--dataset-dir "$DATASET"
      --point-window "${POINT_WINDOW:-2}"
      --min-in-window "${MIN_IN_WINDOW:-1}")
[ -n "${OUT_DIRNAME:-}" ] && ARGS+=(--out-dirname "$OUT_DIRNAME")
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=(${EXTRA_ARGS})

echo "host=$(hostname) task=$TASK dataset=$DATASET args=${ARGS[*]}"

"$MAIN_REPO/.venv/bin/python" -m data_prep.roi_sampling.relift_molmo_cache "${ARGS[@]}"
