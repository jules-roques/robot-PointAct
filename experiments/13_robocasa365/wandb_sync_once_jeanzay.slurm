#!/bin/bash
# Push one finished run's W&B data from Jean Zay. Chained after a training job:
#
#   sbatch --dependency=afterany:<train jobid> \
#          experiments/13_robocasa365/wandb_sync_once_jeanzay.slurm
#
# afterany, deliberately: a run killed at its walltime, or one that crashed at step 40K, still
# has curves worth keeping. afterok would throw away exactly the runs whose logs matter most.
#
# This is the guarantee that nothing is lost. wandb_sync_watch_jeanzay.slurm gives live-ish
# curves during training but is best-effort -- it can queue behind other work or hit its own
# walltime.
#
#SBATCH --job-name=wandb-sync
#SBATCH --account=rgx@cpu
#SBATCH --partition=prepost
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%j.out

set -uo pipefail

REPO="${SLURM_SUBMIT_DIR:-$WORK/code/robot-PointAct}"
cd "$REPO"

echo "host=$(hostname)"
# No --active-within here: this is the final pass, so everything still unsynced or grown since
# the last watcher pass should go up, whether or not it is still being written.
bash experiments/13_robocasa365/wandb_sync_jeanzay.sh
