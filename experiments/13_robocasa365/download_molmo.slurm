#!/bin/bash
# Fetch MolmoPoint-8B to $SCRATCH so compute nodes (no internet) can load it offline.
#
#   sbatch experiments/13_robocasa365/download_molmo.slurm
#
# Why a job and not a login-node one-liner: the repo is ~36 GB (F32 weights) and a
# backgrounded download on a login node gets reaped -- observed here, the process
# vanished after ~2.3 GB with the shards frozen mid-transfer and no error written to the
# log. The prepost partition has internet, a 20 h limit, and is not subject to that.
#
# The download resumes: hf_hub keeps .incomplete shards under the target's .cache, so
# re-running after a failure continues rather than restarting.
#
#SBATCH --job-name=dl-molmo
#SBATCH --account=rgx@cpu
#SBATCH --partition=prepost
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%j.out

set -euo pipefail

REPO_ID="${REPO_ID:-allenai/MolmoPoint-8B}"
DEST="${DEST:-$SCRATCH/models/$(basename "$REPO_ID")}"

export HF_HOME="$SCRATCH/.cache/huggingface"
# Default is 10s; Lustre + a busy hub makes that fire spuriously mid-shard.
export HF_HUB_DOWNLOAD_TIMEOUT=60

MAIN_REPO=/lustre/fswork/projects/rech/rgx/unw24jl/code/robot-PointAct
HF="$MAIN_REPO/.venv/bin/hf"

echo "host=$(hostname) repo=$REPO_ID dest=$DEST"
mkdir -p "$DEST"

# --max-workers 1: the parallel workers contend on a single .gitignore lock under the
# target directory, and on Lustre that produced repeated "Still waiting to acquire lock"
# stalls. One worker is slower in principle and faster in practice here.
"$HF" download "$REPO_ID" --local-dir "$DEST" --max-workers 1

echo "--- downloaded ---"
du -sh "$DEST"
ls -la "$DEST"/*.safetensors 2>/dev/null | tail -5

# Fail the job if the weights are absent, rather than leaving a green job and a cache
# build that dies on a compute node 20 minutes into its allocation.
if ! ls "$DEST"/*.safetensors >/dev/null 2>&1; then
    echo "ERROR: no .safetensors in $DEST after download" >&2
    exit 1
fi

# The shipped remote code combines two boolean tensors with Python `and`, which crashes on
# any batch larger than one. Patch here so a fresh download is immediately usable.
PYTHONPATH="$MAIN_REPO" "$MAIN_REPO/.venv/bin/python" \
    -m data_prep.roi_sampling.patch_molmo_remote_code --model-dir "$DEST"

echo "OK"
