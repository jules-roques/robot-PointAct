#!/bin/bash
# Build the cached per-frame VLM context (3 camera views + instruction) for one task.
#
#   sbatch --export=ALL,TASK=OpenDrawer experiments/13_robocasa365/build_frame_context_jeanzay.slurm
#   # smoke: the first 4 episodes only (a later full run resumes past them):
#   sbatch --qos=qos_gpu_h100-dev --time=00:30:00 --export=ALL,TASK=OpenDrawer,LIMIT_EPISODES=4 <this file>
#   # quick check of what is stored against a fresh batch-1 forward:
#   sbatch --qos=qos_gpu_h100-dev --time=00:20:00 --export=ALL,TASK=OpenDrawer,VERIFY=64 <this file>
#
# One frozen Qwen2.5-VL-3B forward per frame (~130K frames and ~150 GB fp16 for OpenDrawer).
# H100 because the cache must be built with flash_attention_2 -- the attention the training
# config names, and so the one eval re-attaches the VLM with -- and V100 cannot run it.
# RESUMABLE: an episode whose last frame is stored is skipped, so a walltime kill costs
# only the episode in flight; resubmit the identical command.
#
#SBATCH --job-name=framectx
#SBATCH --account=rgx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=06:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%j.out

set -euo pipefail

# Not `purge`: see eval_task_jeanzay.slurm. ffmpeg's libav* is what torchcodec decodes with.
module unload arch 2>/dev/null || true
module load arch/h100
module load ffmpeg/6.1.1

REPO="${SLURM_SUBMIT_DIR:-$WORK/code/robot-PointAct}"
POINTACT_ENV="${POINTACT_ENV:-$WORK/code/robot-PointAct}"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

TASK="${TASK:-OpenDrawer}"
DATASET="$SCRATCH/datasets/robot_data/robocasa365/lerobot_point_lmdb/$TASK"
VLM="${POINTACT_VLM_PATH:-$DSDIR/HuggingFace_Models/Qwen/Qwen2.5-VL-3B-Instruct}"

for path in "$DATASET/meta/episodes.jsonl" "$DATASET/videos" "$VLM"; do
    [ -e "$path" ] || { echo "ERROR: missing $path" >&2; exit 1; }
done
echo "host=$(hostname) task=$TASK vlm=$VLM verify=${VERIFY:-0}"

uv run --project "$POINTACT_ENV" --no-sync \
    python data_prep/cache_frame_context.py \
        --dataset-dir "$DATASET" \
        --vlm-path "$VLM" \
        --num-workers 12 \
        ${LIMIT_EPISODES:+--limit-episodes $LIMIT_EPISODES} \
        ${VERIFY:+--verify $VERIFY}
