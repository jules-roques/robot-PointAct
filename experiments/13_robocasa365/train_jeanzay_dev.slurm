#!/bin/bash
# Jean Zay H100 training under the *dev* QoS, which caps jobs at 2 h but turns around far
# faster than qos_gpu_h100-t3/t4 when the cluster is busy (NeurIPS rebuttal week).
#
# A 20-epoch run does not fit in 2 h, so each job:
#   1. exits immediately if the run already finished (a checkpoint-final-* exists),
#   2. queues its own successor with `afterany` BEFORE training starts, so the chain survives
#      being killed at the walltime, and
#   3. resumes from the newest checkpoint (scripts/train.py does this automatically via
#      has_resume_checkpoint on output_dir).
# CHAIN_REMAINING bounds the chain so it cannot resubmit forever.
#
# The effective batch is held at 128 whatever the GPU count, and the run is named by effective
# batch, so falling back 4 -> 2 -> 1 GPUs RESUMES the same run instead of starting a new one:
#     4 GPUs -> 32/GPU x1     2 GPUs -> 64/GPU x1     1 GPU -> 32 x4 accumulation
#
# Submit (arch/h100 must be loaded in the submitting shell):
#   module load arch/h100
#   sbatch --export=ALL,VARIANT=concerto_eefdensity,CHAIN_REMAINING=20 \
#          --constraint=h100 experiments/13_robocasa365/train_jeanzay_dev.slurm
# To fall back to fewer GPUs, override the request on the command line (--nodes=1 keeps them
# on one node); the script adapts the batch split automatically:
#   sbatch --gres=gpu:2 --cpus-per-task=48 ... same as above
#
#SBATCH --job-name=rc365-dev
#SBATCH --account=rgx@h100
#SBATCH --partition=gpu_p6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365_replay/%x_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365_replay/%x_%j.out

set -euo pipefail

VARIANT="${VARIANT:-concerto}"
CHAIN_REMAINING="${CHAIN_REMAINING:-0}"
EFFECTIVE_BATCH="${EFFECTIVE_BATCH:-128}"

# Follow what SLURM actually granted rather than what was requested, so the batch split is
# always consistent with the allocation. --nodes=1 above guarantees they are on one node.
GPUS="${GPUS:-${SLURM_GPUS_ON_NODE:-4}}"

# Per-device batch per GPU count, chosen to fit H100 memory; accumulation makes up the rest.
if [ -z "${PER_DEVICE_BATCH_SIZE:-}" ]; then
    case "${GPUS}" in
        4) PER_DEVICE_BATCH_SIZE=32 ;;
        2) PER_DEVICE_BATCH_SIZE=64 ;;
        *) PER_DEVICE_BATCH_SIZE=32 ;;
    esac
fi
GRAD_ACCUM=$(( EFFECTIVE_BATCH / (GPUS * PER_DEVICE_BATCH_SIZE) ))
[ "${GRAD_ACCUM}" -lt 1 ] && GRAD_ACCUM=1

if [ $(( GPUS * PER_DEVICE_BATCH_SIZE * GRAD_ACCUM )) -ne "${EFFECTIVE_BATCH}" ]; then
    echo "Refusing to run: ${GPUS} x ${PER_DEVICE_BATCH_SIZE} x ${GRAD_ACCUM} != ${EFFECTIVE_BATCH}." \
         "The ablation arms are only comparable at equal effective batch." >&2
    exit 1
fi

export GPUS PER_DEVICE_BATCH_SIZE GRAD_ACCUM
export NUM_NODES=1
# Checkpoint often enough that every 2 h slice leaves recoverable progress, and keep a few
# behind the newest in case a job is killed midway through writing one.
export SAVE_STEPS="${SAVE_STEPS:-200}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"

# Keep the arch/h100 MODULEPATH (do NOT purge) so ffmpeg's libav* resolves for torchcodec.
module load arch/h100
module load ffmpeg/6.1.1

export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export UV_OFFLINE=1

# W&B offline: compute nodes have no internet. Runs are written under WANDB_DIR; upload them
# afterwards from a login node with:  wandb sync $SCRATCH/wandb/offline-run-*
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

SCRIPT="experiments/13_robocasa365/train_pointact_${VARIANT}.sh"
if [ ! -f "$SCRIPT" ]; then
    echo "No such training variant: $SCRIPT" >&2
    exit 1
fi

# Ask the payload script where it will write, without starting it, so the completion guard and
# the log agree with the real output_dir instead of duplicating its naming rules here.
OUTPUT_DIR=$(POINTACT_PRINT_OUTPUT_DIR=1 bash "$SCRIPT" | tail -1)

echo "host=$(hostname) variant=${VARIANT} chain_remaining=${CHAIN_REMAINING}"
echo "gpus=${GPUS} per_device_bs=${PER_DEVICE_BATCH_SIZE} grad_accum=${GRAD_ACCUM}" \
     "-> effective batch ${EFFECTIVE_BATCH}"
echo "output_dir=${OUTPUT_DIR}"
nvidia-smi -L

# --- 1. stop the chain once the run has finished ---
if [ -n "${OUTPUT_DIR}" ] && compgen -G "${OUTPUT_DIR}/checkpoint-final-*" > /dev/null; then
    echo "Run already complete: $(ls -d ${OUTPUT_DIR}/checkpoint-final-* | tail -1)"
    echo "Not training and not resubmitting."
    exit 0
fi

if [ -n "${OUTPUT_DIR}" ] && compgen -G "${OUTPUT_DIR}/checkpoint-*" > /dev/null; then
    echo "Resuming from: $(ls -d ${OUTPUT_DIR}/checkpoint-* | sort -V | tail -1)"
else
    echo "No checkpoint yet; starting from scratch."
fi

# --- 2. queue the successor BEFORE training, so a walltime kill still continues the chain ---
if [ "${CHAIN_REMAINING}" -gt 0 ]; then
    NEXT=$((CHAIN_REMAINING - 1))
    NEXT_ID=$(sbatch --parsable \
        --dependency=afterany:${SLURM_JOB_ID} \
        --constraint=h100 \
        --nodes=1 \
        --gres=gpu:${GPUS} \
        --cpus-per-task=${SLURM_CPUS_PER_TASK} \
        --export=ALL,VARIANT=${VARIANT},CHAIN_REMAINING=${NEXT},EFFECTIVE_BATCH=${EFFECTIVE_BATCH},SAVE_STEPS=${SAVE_STEPS},SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT},GPUS=,PER_DEVICE_BATCH_SIZE= \
        experiments/13_robocasa365/train_jeanzay_dev.slurm)
    echo "queued successor ${NEXT_ID} (chain_remaining=${NEXT}, afterany:${SLURM_JOB_ID})"
else
    echo "chain exhausted; this is the last job"
fi

# --- 3. train (resumes automatically if output_dir holds a checkpoint) ---
# Must go through `uv run` so the project venv is active: accelerate lives there, and plain
# `bash` fails with "accelerate: command not found".
START_TS=$SECONDS
set +e
uv run --no-sync bash "$SCRIPT"
RC=$?
set -e
ELAPSED=$(( SECONDS - START_TS ))

# A chain that resubmits on immediate failure just burns through slots (and, on a busy dev
# QoS, other people's turn) without ever training. If we died quickly, stop the chain.
if [ "${RC}" -ne 0 ] && [ "${ELAPSED}" -lt 600 ] && [ -n "${NEXT_ID:-}" ]; then
    echo "Training failed after ${ELAPSED}s (rc=${RC}); cancelling successor ${NEXT_ID}" \
         "rather than looping."
    scancel "${NEXT_ID}" || true
fi

echo "training exited rc=${RC} after ${ELAPSED}s"
exit "${RC}"
