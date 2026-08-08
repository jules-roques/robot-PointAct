#!/bin/bash
# Fetch MolmoMotion-4B-H3-F30 to $SCRATCH so compute nodes (no internet) can load it offline.
#
#   sbatch experiments/13_robocasa365/download_molmo_motion.slurm
#
# H3, not H1: the H1 checkpoint sees a single history frame and therefore cannot infer the
# gripper's velocity from the images, which is most of what makes the forecast useful.
#
# Same shape as download_molmo.slurm, minus the remote-code patch: MolmoMotion ships a real
# pip package rather than trust_remote_code modelling files, so there is nothing to patch.
#
# Why a job and not a login-node one-liner: a backgrounded download on a login node gets
# reaped mid-transfer with no error written to the log (observed with MolmoPoint). The
# prepost partition has internet and a 20 h limit.
#
#SBATCH --job-name=dl-molmo-motion
#SBATCH --account=rgx@cpu
#SBATCH --partition=prepost
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%j.out

set -euo pipefail

REPO_ID="${REPO_ID:-allenai/MolmoMotion-4B-H3-F30}"
DEST="${DEST:-$SCRATCH/models/$(basename "$REPO_ID")}"

# On $SCRATCH deliberately: $WORK is near its 500k inode quota and a hub cache is tens of
# thousands of files (docs/clusters/jean-zay.md).
export HF_HOME="$SCRATCH/.cache/huggingface"
# Default is 10 s; Lustre plus a busy hub makes that fire spuriously mid-shard.
export HF_HUB_DOWNLOAD_TIMEOUT=60

MAIN_REPO="${MAIN_REPO:-$WORK/code/robot-PointAct}"
HF="$MAIN_REPO/.venv/bin/hf"

echo "host=$(hostname) repo=$REPO_ID dest=$DEST"
mkdir -p "$DEST"

# --max-workers 1: parallel workers contend on a single .gitignore lock under the target
# directory, which stalls repeatedly on Lustre (see download_molmo.slurm).
"$HF" download "$REPO_ID" --local-dir "$DEST" --max-workers 1

echo "--- downloaded ---"
du -sh "$DEST"
ls -la "$DEST"/*.safetensors 2>/dev/null | tail -5

# Fail the job if the weights are absent, rather than leaving a green job and a gate that
# dies on a compute node 20 minutes into its allocation.
if ! ls "$DEST"/*.safetensors >/dev/null 2>&1; then
    echo "ERROR: no .safetensors in $DEST after download" >&2
    exit 1
fi

echo "OK"
