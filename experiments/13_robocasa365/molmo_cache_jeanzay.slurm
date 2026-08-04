#!/bin/bash
# Build the MolmoPoint anchor cache for one RoboCasa365 task.
#
#   sbatch --export=ALL,TASK=OpenDrawer experiments/13_robocasa365/molmo_cache_jeanzay.slurm
#
# Smoke test first (one episode, and probe the checkpoint's point-tuple field order):
#   sbatch --export=ALL,TASK=OpenDrawer,EPISODES="0",EXTRA="--verify-point-order" \
#          --qos=qos_gpu_h100-dev --time=00:40:00 <this file>
#
# Split a full task across an array (episodes are independent):
#   sbatch --export=ALL,TASK=PickPlaceCounterToStove,PER_TASK=64 --array=0-7 <this file>
#
# H100 is fine here and A100 is not needed: this is pure torch inference with no MuJoCo, so
# the "RoboCasa365 misbehaves on H100" constraint does not apply (docs/clusters/jean-zay.md).
#
#SBATCH --job-name=molmocache
#SBATCH --account=rgx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=06:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%A_%a.out
#SBATCH --error=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%A_%a.out

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$WORK/code/robot-PointAct}"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

TASK="${TASK:?set TASK=<OpenDrawer|PickPlaceCounterToStove|TurnOnMicrowave> via --export}"
DATA="${DATA:-$REPO/robot_data/robocasa365/lerobot_point_lmdb/$TASK}"
MODEL="${MODEL:-$SCRATCH/models/MolmoPoint-8B}"

# The venv lives on $SCRATCH, not in the repo: $WORK is at ~96% of its 500k inode quota and
# a torch install is ~40k files. See docs/envs.md.
VENV="${MOLMO_VENV:-$SCRATCH/venvs/molmo}"

export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ ! -d "$MODEL" ] || ! ls "$MODEL"/*.safetensors >/dev/null 2>&1; then
    echo "ERROR: no weights at $MODEL -- run download_molmo.slurm first" >&2
    exit 1
fi

# Idempotent, and cheap next to model loading. Guards against a re-downloaded checkpoint
# silently reverting to the unpatched remote code (see the script for what it fixes).
"$VENV/bin/python" -m data_prep.roi_sampling.patch_molmo_remote_code --model-dir "$MODEL"

ARGS=(--dataset-dir "$DATA" --model-dir "$MODEL")
if [ -n "${EPISODES:-}" ]; then
    ARGS+=(--episodes ${EPISODES})
elif [ -n "${PER_TASK:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    LO=$(( SLURM_ARRAY_TASK_ID * PER_TASK ))
    ARGS+=(--episode-range "$LO" $(( LO + PER_TASK )))
fi

echo "host=$(hostname) task=$TASK data=$DATA"
nvidia-smi -L

srun "$VENV/bin/python" -m data_prep.roi_sampling.build_molmo_cache \
    "${ARGS[@]}" ${EXTRA:-}
