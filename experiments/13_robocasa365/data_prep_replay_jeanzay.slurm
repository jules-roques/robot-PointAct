#!/bin/bash
# Stage 1/3 on Jean Zay: replay demonstrations in sim and cache fused point clouds.
#
#   sbatch --export=ALL,TASK=TurnOnMicrowave experiments/13_robocasa365/data_prep_replay_jeanzay.slurm
#
# V100, deliberately: this stage does MuJoCo/EGL rendering and no model inference, so it needs
# no Ampere features -- and RoboCasa365 sim misbehaves on H100 anyway. V100 is also the Jean Zay
# tier with the shortest queue.
#
# --resume skips already-cached episodes, so resubmitting this same job continues past the
# walltime limit rather than starting over.
#
#SBATCH --job-name=rc365-replay
#SBATCH --account=rgx@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

set -euo pipefail

module purge
module load ffmpeg/6.1.1

WORKTREE="$WORK/code/robot-PointAct/.claude/worktrees/pointcount-ablation"
cd "$WORKTREE"
export PYTHONPATH="$WORKTREE:${PYTHONPATH:-}"

export MUJOCO_GL=egl
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export UV_OFFLINE=1

source experiments/13_robocasa365/task_paths.sh
TASK="${TASK:?set TASK=<task name> via --export}"
INPUT_DIR="${INPUT_DIR:-$(task_input_dir "$TASK")}"

echo "host=$(hostname) task=$TASK input_dir=$INPUT_DIR"
nvidia-smi -L

if [ ! -d "$INPUT_DIR/lerobot" ]; then
    echo "no $INPUT_DIR/lerobot -- run data_prep_download_jeanzay.slurm first" >&2
    exit 1
fi

uv run --project envs/robocasa365 --no-sync \
    python data_prep/robocasa365_to_lerobot/replay.py \
        --input-dir "$INPUT_DIR/lerobot" \
        --cache-dir "$INPUT_DIR/replay_cache" \
        --resume
