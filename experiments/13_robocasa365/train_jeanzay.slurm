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
# Smoke a config on the dev QoS first when anything about the data path changed -- 2 h cap,
# fast turnaround, and a bad run costs minutes instead of a t3 slot:
#   sbatch --constraint=h100 --qos=qos_gpu_h100-dev --time=00:40:00 \
#          --export=ALL,RUN_CONFIG=<run>.yaml,SMOKE_STEPS=30 <this file>
#
# Neutral default: pass --job-name=<arm> at submission, or four arms share one name in
# squeue and there is no way to tell which is which.
#SBATCH --job-name=train
#SBATCH --account=rgx@h100
#SBATCH --partition=gpu_p6
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
# t3 (20 h) rather than t4 (100 h): a 50K-step run at 4 GPUs finishes in ~7 h, so t4 asks the
# scheduler to reserve five times what the job needs and queues behind everything. A run that
# genuinely needs longer (the stage-0 VLM arms are ~22 h) should override BOTH on the command
# line -- `sbatch --qos=qos_gpu_h100-t4 --time=100:00:00 ...` -- rather than raising it here.
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

# SMOKE_STEPS: train a handful of steps to prove the config, the data path and the sampling
# arm all work, then stop. It writes to its own output_dir and its own run name -- a smoke
# sharing the real run's directory would leave checkpoints the real run then resumes from,
# turning a 30-step sanity check into the first 30 steps of the experiment.
if [ -n "${SMOKE_STEPS:-}" ]; then
    SMOKE_NAME="smoke-$(basename "${RUN_CONFIG%.yaml}")"
    export EXTRA_TRAIN_ARGS="--max_steps ${SMOKE_STEPS} --save_steps ${SMOKE_STEPS} \
        --logging_steps 1 --output_dir $SCRATCH/PointAct_exprs/robocasa365/smoke/$SMOKE_NAME \
        --run_name $SMOKE_NAME"
    export WANDB_MODE=disabled
    echo "SMOKE: $SMOKE_STEPS steps -> $SCRATCH/PointAct_exprs/robocasa365/smoke/$SMOKE_NAME"
fi
uv run --project "$POINTACT_ENV" --no-sync \
    bash experiments/13_robocasa365/train.sh "$RUN_CONFIG"
