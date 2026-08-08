#!/bin/bash
# Ground-truth target positions for the MolmoPoint accuracy study (robocasa365 env, V100).
#
#   sbatch --export=ALL,TASK=TurnOnMicrowave experiments/13_robocasa365/target_positions_jeanzay.slurm
#   sbatch --export=ALL,TASK=TurnOnMicrowave,SMOKE=1 <this file>     # 2 episodes, 40 frames
#
# Replays every episode with its recorded actions and records where the manipulated part
# really is, in the robot-base frame -- the reference `eval_molmo_accuracy.py` scores the
# pointer against. Nothing is rendered or written per frame beyond three float triples, so
# this is far cheaper than the replay that built the point clouds.
#
# V100 like the calibration job: MuJoCo needs a GL context, not a big GPU. The array splits
# the episode range; each task writes its own shard and `--merge` stitches them.
#
#SBATCH --job-name=gt-targets
#SBATCH --account=rgx@v100
#SBATCH --partition=gpu_p13
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=05:00:00
#SBATCH --array=0-3
#SBATCH --output=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365_replay/%x_%A_%a.out
#SBATCH --error=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365_replay/%x_%A_%a.out

set -euo pipefail
module purge
module load ffmpeg/6.1.1
export MUJOCO_GL=egl
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_OFFLINE=1

MAIN_REPO=/lustre/fswork/projects/rech/rgx/unw24jl/code/robot-PointAct
REPO="${SLURM_SUBMIT_DIR:-$MAIN_REPO}"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"   # shadow main-checkout pointact with the worktree

TASK="${TASK:?set TASK=<OpenDrawer|TurnOnMicrowave> via --export}"

SRC_ROOT="$SCRATCH/datasets/robot_data/robocasa365/v1.0/target/atomic/$TASK"
BASE=$(ls -d "$SRC_ROOT"/*/lerobot 2>/dev/null | head -1)
[ -z "$BASE" ] && { echo "no source episodes under $SRC_ROOT" >&2; exit 1; }

OUT_DIR="$SCRATCH/datasets/robot_data/robocasa365/lerobot_point_lmdb/$TASK/roi_meta"
mkdir -p "$OUT_DIR"

# Episode count from the source dirs, split evenly across the array.
N_EPS=$(ls -d "$BASE"/extras/episode_* | wc -l)
N_TASKS="${SLURM_ARRAY_TASK_COUNT:-1}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
PER=$(( (N_EPS + N_TASKS - 1) / N_TASKS ))
START=$(( IDX * PER ))
STOP=$(( START + PER ))

OUT_NAME="${OUT_NAME:-target_positions}"
ARGS=(--task "$TASK" --source-dir "$BASE"
      --out "$OUT_DIR/${OUT_NAME}_shard${IDX}.npz"
      --episode-range "$START" "$STOP")
[ -n "${SMOKE:-}" ] && ARGS=(--task "$TASK" --source-dir "$BASE"
                             --out "$OUT_DIR/${OUT_NAME}_smoke.npz"
                             --max-episodes 2 --max-frames 40)
# RESET_ONLY=1 for a target a full-replay probe has shown is static (microwave buttons);
# EXTRA_ARGS carries anything else, e.g. a subset for the cross-check dump.
[ -n "${RESET_ONLY:-}" ] && ARGS+=(--reset-only)
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=(${EXTRA_ARGS})

echo "task=$TASK src=$BASE eps=$N_EPS shard=$IDX range=[$START,$STOP)"
nvidia-smi -L || true

srun uv run --project "$MAIN_REPO/envs/robocasa365" --no-sync \
  python -m data_prep.roi_sampling.dump_target_positions "${ARGS[@]}"
