#!/bin/bash
# Launch one run of the point-count / task ablation from its run yaml.
#
#   bash experiments/13_robocasa365/train.sh experiments/13_robocasa365/runs/od-eef-n4096-s0.yaml
#
# Everything about the run -- hyperparameters, data config, ablation coordinates, W&B
# grouping -- lives in that one file (see pointact/train/run_config.py). This script only
# supplies what genuinely depends on the machine: the VLM path and the accelerate topology.
#
# Replaces the per-arm train_pointact_concerto*.sh scripts, which duplicated ~40 flags each.
set -euo pipefail

ulimit -u 2048

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

RUN_CONFIG=${1:?usage: train.sh <run.yaml>}

# Jean Zay exposes the shared IDRIS model tree at $DSDIR; CLEPS has no $DSDIR, so fall back to
# a $SCRATCH copy. The run yamls refer to this as $POINTACT_VLM_PATH so no cluster path is
# baked into shared config. Still needed under context_source=text_cache: the *config* (not
# the weights) supplies text_config.hidden_size, which must match the cached embeddings.
POINTACT_VLM_PATH=${DSDIR:+$DSDIR/HuggingFace_Models/Qwen/Qwen2.5-VL-3B-Instruct}
export POINTACT_VLM_PATH=${POINTACT_VLM_PATH:-$SCRATCH/models/Qwen2.5-VL-3B-Instruct}

# Effective batch is pinned at 128 for every arm. Ablation arms are only comparable at equal
# effective batch: 2 GPUs x 32 is not equivalent to 4 x 32, it halves the effective batch and
# doubles the optimiser steps. Derive the split from what SLURM actually granted, so a
# fallback from 4 to 2 GPUs adjusts accumulation instead of silently changing the recipe.
EFFECTIVE_BATCH=128
GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}
GPUS=${GPUS:-1}
NUM_NODES=${NUM_NODES:-1}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-32}

TOTAL_PROCS=$((GPUS * NUM_NODES))
ACCUM=$((EFFECTIVE_BATCH / (TOTAL_PROCS * PER_DEVICE_BATCH_SIZE)))
if [ $((TOTAL_PROCS * PER_DEVICE_BATCH_SIZE * ACCUM)) -ne $EFFECTIVE_BATCH ]; then
    echo "refusing to run: ${TOTAL_PROCS} proc x ${PER_DEVICE_BATCH_SIZE} x accum ${ACCUM}" \
         "!= effective batch ${EFFECTIVE_BATCH}" >&2
    exit 1
fi
echo "effective batch ${EFFECTIVE_BATCH} = ${TOTAL_PROCS} proc x ${PER_DEVICE_BATCH_SIZE} x accum ${ACCUM}"

if [ "$TOTAL_PROCS" -eq 1 ]; then
    ACCELERATE_ARGS="--num_processes 1 --num_machines 1"
elif [ "$NUM_NODES" -eq 1 ]; then
    ACCELERATE_ARGS="--multi_gpu --num_processes ${GPUS} --num_machines 1 --machine_rank 0"
else
    ACCELERATE_ARGS="--multi_gpu --num_processes ${TOTAL_PROCS} --num_machines ${NUM_NODES} \
        --machine_rank ${SLURM_NODEID} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT}"
fi

export WANDB_ENTITY=${WANDB_ENTITY:-diffusion4robots}
export WANDB_PROJECT=${WANDB_PROJECT:-pointact-robocasa365}
# Deliberately no WANDB_NAME: transformers' WandbCallback overwrites it with --run-name
# anyway (integration_utils.py sets init_args["name"] = args.run_name), so exporting it only
# creates the illusion of control. The name comes from the run yaml's meta block.
# WANDB_RUN_ID / RESUME / RUN_GROUP / JOB_TYPE / TAGS are set by configure_wandb_identity().

# TF32 needs Ampere or newer; the yaml asks for it and this turns it off on older cards
# rather than letting torch warn per step.
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1)
if [ -n "$COMPUTE_CAP" ] && [ "${COMPUTE_CAP%%.*}" -lt 8 ]; then
    echo "compute capability ${COMPUTE_CAP} < 8.0: disabling tf32"
    export POINTACT_DISABLE_TF32=1
fi

accelerate launch $ACCELERATE_ARGS scripts/train.py "$RUN_CONFIG" \
    --gradient-accumulation-steps ${ACCUM} \
    --per-device-train-batch-size ${PER_DEVICE_BATCH_SIZE}
