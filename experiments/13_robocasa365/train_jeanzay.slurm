#!/bin/bash
# Jean Zay counterpart to train.slurm (CLEPS). Same payload (train.sh + a run yaml), but Jean
# Zay and CLEPS are separate SLURM controllers, so #SBATCH account/partition/module directives
# cannot be shared between them — this file carries Jean Zay's.
#
# Submit (arch/h100 must be loaded in the submitting shell):
#   module load arch/h100
#   sbatch --constraint=h100 \
#          --export=ALL,RUN_CONFIG=experiments/13_robocasa365/runs/od-eef-n4096-s0.yaml \
#          experiments/13_robocasa365/train_jeanzay.slurm
#
#SBATCH --job-name=train-od
#SBATCH --account=rgx@h100
#SBATCH --partition=gpu_p6
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
# t3/20h fits every cached-context run in this grid: 50K steps at 8192 points is ~8h on 4
# H100s, and asking for 100h on t4 only buys a worse place in the queue. The live-VLM arms
# are the exception at ~21.4h -- they exceed the t3 cap and must override at submit time:
#   sbatch --qos=qos_gpu_h100-t4 --time=40:00:00 ...
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365_replay/%x_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365_replay/%x_%j.out

set -euo pipefail
# Keep the arch/h100 MODULEPATH (do NOT purge) so ffmpeg's libav* resolves for torchcodec.
module load arch/h100
module load ffmpeg/6.1.1

export NUM_NODES=1
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export UV_OFFLINE=1

# W&B offline: compute nodes have no internet. Runs are written under WANDB_DIR; upload them
# afterwards from a LOGIN node with:
#     bash experiments/13_robocasa365/wandb_sync_jeanzay.sh
# (runs live at $WANDB_DIR/wandb/offline-run-*, one level deeper than WANDB_DIR)
export WANDB_MODE=offline
export WANDB_DIR="$SCRATCH/wandb"
mkdir -p "$WANDB_DIR"

# The repo lives at a different place on each cluster ($HOME/code on CLEPS, $WORK/code on
# Jean Zay), and sbatch is always run from the repo root, so trust the submit dir first and
# fall back to the known locations. Hardcoding $HOME/code here silently broke Jean Zay jobs.
REPO=""
for candidate in "${SLURM_SUBMIT_DIR:-}" "$HOME/code/robot-PointAct" "${WORK:-}/code/robot-PointAct"; do
    if [ -n "$candidate" ] && [ -d "$candidate/experiments/13_robocasa365" ]; then
        REPO="$candidate"
        break
    fi
done
if [ -z "$REPO" ]; then
    echo "Could not locate the robot-PointAct checkout" >&2
    exit 1
fi
cd "$REPO"

echo "host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

# --project must name a checkout that HAS a .venv. When REPO is a worktree it does not, and a
# bare `uv run` would silently build an empty one there (see docs/envs.md). PYTHONPATH, set by
# train.sh from $(pwd), is what makes the worktree's own code shadow the installed package.
POINTACT_ENV="${POINTACT_ENV:-$WORK/code/robot-PointAct}"

# accelerate (inside the train script) spawns the 4 GPU processes itself — no srun.
RUN_CONFIG="${RUN_CONFIG:?set RUN_CONFIG=<experiments/13_robocasa365/runs/*.yaml> via --export}"
# Everything about the run lives in that yaml -- no training arguments here.
echo "run_config=$RUN_CONFIG"
uv run --project "$POINTACT_ENV" --no-sync \
    bash experiments/13_robocasa365/train.sh "$RUN_CONFIG"
