#!/bin/bash
# Re-lift the MolmoPoint anchors from per-pixel depth (robocasa365 env, V100 array).
#
#   sbatch --export=ALL,TASK=TurnOnMicrowave experiments/13_robocasa365/molmo_depth_anchors_jeanzay.slurm
#   sbatch --export=ALL,TASK=TurnOnMicrowave,SMOKE=1 <this file>    # 2 episodes, 40 frames
#
# The cache's original lift medians the *voxelised* cloud inside a pixel window, which fails
# whenever the window subtends less than one voxel -- i.e. for anything nearer than ~33 cm in
# the wrist camera. This replays each episode and reads the organised per-pixel cloud instead,
# so an anchor exists wherever depth does. Only the lift changes; fusion is untouched.
#
# Unlike the ground-truth dump this one MUST render: it needs the depth buffer for all three
# cameras at the resolution the pixels were predicted on (256), so it is ~5x slower per
# episode. Eight-way array to keep it inside a dev-QoS slot.
#
#SBATCH --job-name=molmo-depth
#SBATCH --account=rgx@v100
#SBATCH --partition=gpu_p13
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --array=0-7
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

DATASET="$SCRATCH/datasets/robot_data/robocasa365/lerobot_point_lmdb/$TASK"
OUT_DIR="$DATASET/roi_meta"
mkdir -p "$OUT_DIR"

IDX="${SLURM_ARRAY_TASK_ID:-0}"
N_TASKS="${SLURM_ARRAY_TASK_COUNT:-1}"

OUT_NAME="${OUT_NAME:-depth_anchors}"
ARGS=(--task "$TASK" --source-dir "$BASE" --dataset-dir "$DATASET"
      --out "$OUT_DIR/${OUT_NAME}_shard${IDX}.npz"
      --shard "$IDX" --num-shards "$N_TASKS")
[ -n "${SMOKE:-}" ] && ARGS=(--task "$TASK" --source-dir "$BASE" --dataset-dir "$DATASET"
                             --out "$OUT_DIR/${OUT_NAME}_smoke.npz"
                             --max-episodes 2 --max-frames 40)
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=(${EXTRA_ARGS})

echo "task=$TASK src=$BASE shard=$IDX/$N_TASKS"
nvidia-smi -L || true

srun uv run --project "$MAIN_REPO/envs/robocasa365" --no-sync \
  python -m data_prep.roi_sampling.dump_molmo_depth_anchors "${ARGS[@]}"
