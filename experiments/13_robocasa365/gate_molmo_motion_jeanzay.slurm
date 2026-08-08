#!/bin/bash
# Stage 4.0 gate: score MolmoMotion's gripper forecast against the demos' own future.
#
#   sbatch experiments/13_robocasa365/gate_molmo_motion_jeanzay.slurm                  # all 3 tasks
#   sbatch --export=ALL,TASKS=OpenDrawer,EPISODES=2 <this file>                        # smoke test
#
# GPU, unlike the Stage 3 gate. That one read a finished anchor cache and so ran on CPU;
# here there is no cache yet, so the 4B model runs in the gate itself.
#
# Budget: ~0.5 s per forward, one forward per (replan frame, history stride, view). At the
# defaults -- 8 episodes x 3 tasks, stride 8, 2 history strides, left view only -- that is
# roughly 1,500 forwards, well inside the hour. Raise --episodes before trusting a number
# for anything beyond gating.
#
# H100 is fine and A100 is not required: pure torch inference, no MuJoCo, so the "RoboCasa365
# misbehaves on H100" constraint does not apply (docs/clusters/jean-zay.md).
#
# W&B is offline here -- compute nodes have no internet. The run appears on wandb.ai only
# after a LOGIN-node sync:
#   bash experiments/13_robocasa365/wandb_sync_jeanzay.sh
# The overlay mp4s and gate_summary.json land in $OUT_DIR regardless, so the gate can be
# reviewed without W&B at all.
#
# This is a gate, not a report: nothing trains until a human has looked at it. What to look
# for is documented in the script's own docstring -- above all, whether the forecast beats
# the static "the gripper does not move" baseline, which is the column that decides whether
# the model is worth its forwards.
#
#SBATCH --job-name=gate-molmo-motion
#SBATCH --account=rgx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%j.out

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$WORK/code/robot-PointAct}"
cd "$REPO"
# Without this the installed `pointact` shadows the worktree's source and none of the new
# geometry helpers are visible (docs/envs.md).
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-$SCRATCH/datasets/robot_data/robocasa365/lerobot_point_lmdb}"
MODEL="${MODEL:-$SCRATCH/models/MolmoMotion-4B-H3-F30}"
OUT_DIR="${OUT_DIR:-$SCRATCH/viz/molmo_motion_gate}"

# The venv lives on $SCRATCH, not in the repo: $WORK is near its 500k inode quota and a
# torch install is tens of thousands of files. See docs/envs.md.
VENV="${MOLMO_MOTION_VENV:-$SCRATCH/venvs/molmo_motion}"

export HF_HOME="$SCRATCH/.cache/huggingface"
# Belt and braces: the model is loaded from a local dir, and without this a missing file
# turns into a silent hang on a node with no route to the hub rather than an error.
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DIR="${WANDB_DIR:-$SCRATCH/wandb}"
mkdir -p "$WANDB_DIR" "$OUT_DIR"

if [ ! -d "$MODEL" ]; then
    echo "ERROR: no weights at $MODEL -- run download_molmo_motion.slurm first" >&2
    exit 1
fi

ARGS=(--dataset-root "$DATA_ROOT" --model-dir "$MODEL" --out-dir "$OUT_DIR" --wandb)
[ -n "${TASKS:-}" ] && ARGS+=(--tasks ${TASKS})
[ -n "${EPISODES:-}" ] && ARGS+=(--episodes "$EPISODES")
[ -n "${VIEWS:-}" ] && ARGS+=(--views ${VIEWS})
[ -n "${HISTORY_STRIDES:-}" ] && ARGS+=(--history-strides ${HISTORY_STRIDES})
[ -n "${EXTRA:-}" ] && ARGS+=(${EXTRA})

echo "host=$(hostname) model=$MODEL out=$OUT_DIR tasks=${TASKS:-all}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$VENV/bin/python" -m data_prep.roi_sampling.gate_molmo_motion "${ARGS[@]}"

echo
echo "Artifacts: $OUT_DIR"
echo "Then sync from a login node: bash experiments/13_robocasa365/wandb_sync_jeanzay.sh"
