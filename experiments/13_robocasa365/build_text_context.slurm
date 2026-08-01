#!/bin/bash
# Build the cached text-only VLM context for one or more tasks.
#
#   sbatch --export=ALL,TASKS="OpenDrawer" experiments/13_robocasa365/build_text_context.slurm
#
# Runs the frozen Qwen text-only forward once per unique instruction, so this is seconds of
# GPU per task -- a V100 is plenty, and V100 is the Jean Zay tier with the least queue.
# (FlashAttention would rule out V100, but the cache script asks for sdpa.)
#
#SBATCH --job-name=rc365-textctx
#SBATCH --account=rgx@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

set -euo pipefail

module purge

WORKTREE="$WORK/code/robot-PointAct/.claude/worktrees/pointcount-ablation"
cd "$WORKTREE"
export PYTHONPATH="$WORKTREE:${PYTHONPATH:-}"
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

LMDB_ROOT="$SCRATCH/datasets/robot_data/robocasa365/lerobot_point_lmdb"
VLM="$DSDIR/HuggingFace_Models/Qwen/Qwen2.5-VL-3B-Instruct"
TASKS="${TASKS:-OpenDrawer}"

for task in $TASKS; do
    echo "=== $task ==="
    uv run --project "$WORK/code/robot-PointAct" --no-sync \
        python data_prep/cache_text_context.py \
            --dataset-dir "$LMDB_ROOT/$task" \
            --vlm-path "$VLM"
done
