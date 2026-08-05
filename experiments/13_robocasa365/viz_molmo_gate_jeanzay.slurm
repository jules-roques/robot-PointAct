#!/bin/bash
# Stage 3.0 gate: render what MolmoPoint actually pointed at, from the finished anchor cache.
#
#   sbatch experiments/13_robocasa365/viz_molmo_gate_jeanzay.slurm                    # all 3 tasks
#   sbatch --export=ALL,TASKS=TurnOnMicrowave,EPISODE=7 <this file>                   # one task/episode
#
# CPU only: this reads the cache and decodes two episode videos per task, so it needs plotly
# and av rather than a GPU. Runs in the root env.
#
# W&B is offline here -- Jean Zay compute nodes have no internet (docs/clusters/jean-zay.md).
# The run only appears on wandb.ai after a LOGIN-node sync:
#   bash experiments/13_robocasa365/wandb_sync_jeanzay.sh
# The mp4s and the interactive HTML are written to $OUT_DIR regardless, so the gate can be
# reviewed without W&B at all.
#
# This is a gate, not a report: nothing downstream trains until a human has looked at it.
# What to look for is documented in the script's own docstring.
#
#SBATCH --job-name=viz-molmo-gate
#SBATCH --account=rgx@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-dev
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%j.out

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$WORK/code/robot-PointAct}"
cd "$REPO"

# The root env has `pointact` installed, so without this the installed copy shadows the
# worktree's source and none of the wrist-camera record layout is visible (docs/envs.md).
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-$REPO/robot_data/robocasa365/lerobot_point_lmdb}"
OUT_DIR="${OUT_DIR:-$SCRATCH/viz/molmo_gate}"
CACHE_DIRNAME="${CACHE_DIRNAME:-points_3views_molmo}"

export WANDB_MODE=offline
export WANDB_DIR="${WANDB_DIR:-$SCRATCH/wandb}"
mkdir -p "$WANDB_DIR" "$OUT_DIR"

ARGS=(--dataset-root "$DATA_ROOT" --out-dir "$OUT_DIR" --cache-dirname "$CACHE_DIRNAME" --wandb)
[ -n "${TASKS:-}" ] && ARGS+=(--tasks ${TASKS})
[ -n "${EPISODE:-}" ] && ARGS+=(--episode "$EPISODE")
[ -n "${STRIDE:-}" ] && ARGS+=(--stride "$STRIDE")

echo "host=$(hostname) out=$OUT_DIR cache=$CACHE_DIRNAME tasks=${TASKS:-all}"

# The main checkout's .venv, not the worktree's -- a worktree has no env of its own.
"$WORK/code/robot-PointAct/.venv/bin/python" \
    -m data_prep.roi_sampling.viz_molmo_gate "${ARGS[@]}"

echo
echo "Artifacts: $OUT_DIR"
echo "Then sync from a login node: bash experiments/13_robocasa365/wandb_sync_jeanzay.sh"
