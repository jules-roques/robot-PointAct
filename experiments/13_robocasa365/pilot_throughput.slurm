#!/bin/bash
# Throughput pilot on Jean Zay H100s: measure what the ablation budget assumed.
#
#   sbatch experiments/13_robocasa365/pilot_throughput.slurm
#
# Uses the 2h dev QoS -- twelve short runs (six configurations x two step counts) fit
# comfortably, and dev turns around far faster than t3/t4 for a job this small.
#
#SBATCH --job-name=pilot
#SBATCH --account=rgx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

set -euo pipefail

module purge
module load arch/h100
# The vlm arm decodes video, so torchcodec needs libav* from the ffmpeg module. The
# text_cache arm does not (it skips decoding entirely), which is why only the vlm cells
# failed without this.
module load ffmpeg/6.1.1

WORKTREE="$WORK/code/robot-PointAct/.claude/worktrees/pointcount-ablation"
cd "$WORKTREE"

# Run worktree code against the main checkout's venv: prepend the worktree to PYTHONPATH so it
# shadows the env-installed `pointact` (see docs/envs.md).
export PYTHONPATH="$WORKTREE:${PYTHONPATH:-}"
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export UV_OFFLINE=1
export WANDB_MODE=disabled

# The run yamls refer to the VLM as $POINTACT_VLM_PATH. Needed even for the text_cache arms:
# the *config* supplies text_config.hidden_size, which must match the cached embeddings.
export POINTACT_VLM_PATH="$DSDIR/HuggingFace_Models/Qwen/Qwen2.5-VL-3B-Instruct"

# Data configs use the repo-root-relative `robot_data` symlink (Libero convention).
[ -e robot_data ] || ln -s "$SCRATCH/datasets/robot_data" robot_data

echo "host=$(hostname)"
nvidia-smi -L

uv run --project "$WORK/code/robot-PointAct" --no-sync \
    python experiments/13_robocasa365/pilot_throughput.py \
        --gpus "${SLURM_GPUS_ON_NODE:-4}" \
        --short-steps 40 --long-steps 160 \
        --out "$SCRATCH/pilot_throughput.json"
