#!/bin/bash
# Stage A: dump RoboCASA left/right camera calibration (robocasa365 env, V100/EGL).
#
#   sbatch --export=ALL,TASK=PickPlaceCounterToStove experiments/13_robocasa365/calib_jeanzay.slurm
#
# The MolmoPoint cache lifts a 2D point to 3D by reprojecting the stored base-frame cloud
# into a camera, which needs the intrinsics and base->cam transform recorded here. The
# agentview cameras are robot-mounted, so one calibration serves a whole task; the script
# checks that across episodes and warns if it does not hold. Treat that warning as fatal --
# the whole lift assumes it.
#
# V100 is fine and deliberate: this only resets the simulator to read camera matrices, so
# it needs neither FlashAttention nor a big GPU.
#
#SBATCH --job-name=roi-calib
#SBATCH --account=rgx@v100
#SBATCH --partition=gpu_p13
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu-dev
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365_replay/%x_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365_replay/%x_%j.out

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

TASK="${TASK:?set TASK=<OpenDrawer|PickPlaceCounterToStove|TurnOnMicrowave> via --export}"

# Each task was replayed on its own date, so the source directory is discovered rather than
# hard-coded -- hard-coding it is how this script stayed OpenDrawer-only.
SRC_ROOT="$SCRATCH/datasets/robot_data/robocasa365/v1.0/target/atomic/$TASK"
BASE=$(ls -d "$SRC_ROOT"/*/lerobot 2>/dev/null | head -1)
[ -z "$BASE" ] && { echo "no source episodes under $SRC_ROOT" >&2; exit 1; }

OUT="$SCRATCH/datasets/robot_data/robocasa365/lerobot_point_lmdb/$TASK/roi_meta/camera_calib.npz"
mkdir -p "$(dirname "$OUT")"
echo "task=$TASK src=$BASE out=$OUT"
nvidia-smi -L || true

# robocasa365 env (.venv) lives in the main checkout; worktree provides the code via PYTHONPATH.
srun uv run --project "$MAIN_REPO/envs/robocasa365" --no-sync \
  python data_prep/roi_sampling/dump_camera_calib.py \
  --source-dir "$BASE" \
  --out "$OUT" \
  --num-episodes 8
