#!/bin/bash
# Stage 1/3 on Jean Zay: replay demonstrations in sim and cache fused point clouds.
#
#   sbatch --export=ALL,TASK=TurnOnMicrowave experiments/13_robocasa365/data_prep_replay_jeanzay.slurm
#
# V100, deliberately: this stage does MuJoCo/EGL rendering and no model inference, so it needs
# no Ampere features -- and RoboCasa365 sim misbehaves on H100 anyway. V100 is also the Jean Zay
# tier with the shortest queue.
#
# Sharded 8 ways as an array. ~500 episodes in a single 20h job is uncomfortably tight, and
# the cache is one .npz + one .npy per episode (replay.py:episode_cache_path), so array tasks
# writing disjoint episode sets into the same cache dir never touch the same file.
#
# --resume skips already-cached episodes, so resubmitting continues past the walltime limit
# rather than starting over.
#
#SBATCH --job-name=rc365-replay
#SBATCH --account=rgx@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --array=0-7
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out

set -euo pipefail

module purge
module load ffmpeg/6.1.1

WORKTREE="$WORK/code/robot-PointAct/.claude/worktrees/pointcount-ablation"
cd "$WORKTREE"
export PYTHONPATH="$WORKTREE:${PYTHONPATH:-}"

# Point uv at the MAIN checkout's simulator env, never the worktree's: the worktree has
# no envs/robocasa365/.venv, and uv would silently build an empty one (thousands of
# inodes on a WORK filesystem already at 93%) that lacks robocasa entirely. PYTHONPATH
# above is what makes the worktree's own code shadow the installed copy. See docs/envs.md.
ROBOCASA_ENV="$WORK/code/robot-PointAct/envs/robocasa365"

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

# Stride the episodes across the array rather than splitting into contiguous blocks: episode
# length varies a lot within a task, and striding keeps each shard's total work comparable.
NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
SHARD=${SLURM_ARRAY_TASK_ID:-0}
EPISODES=$(python - <<EOF
import json
total = json.load(open("$INPUT_DIR/lerobot/meta/info.json"))["total_episodes"]
print(" ".join(str(i) for i in range($SHARD, total, $NUM_SHARDS)))
EOF
)
echo "shard $SHARD/$NUM_SHARDS: $(echo $EPISODES | wc -w) episodes"

uv run --project "$ROBOCASA_ENV" --no-sync \
    python data_prep/robocasa365_to_lerobot/replay.py \
        --input-dir "$INPUT_DIR/lerobot" \
        --cache-dir "$INPUT_DIR/replay_cache" \
        --episodes $EPISODES \
        --resume
