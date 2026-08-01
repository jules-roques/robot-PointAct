#!/bin/bash
# Keep W&B curves roughly live during a Jean Zay campaign.
#
#   sbatch experiments/13_robocasa365/wandb_sync_watch_jeanzay.slurm
#   sbatch --export=ALL,INTERVAL=900 experiments/13_robocasa365/wandb_sync_watch_jeanzay.slurm
#
# Compute nodes have no internet, so training writes W&B data offline and it only reaches
# wandb.ai when pushed from somewhere that does. `prepost` has outbound access and a 20h
# walltime, which is what makes this possible at all -- a login node is the wrong place for a
# multi-hour loop.
#
# One job covers every run in the campaign. Runs keep a stable WANDB_RUN_ID (pinned by
# configure_wandb_identity), so re-pushing a growing offline run updates the same cloud run
# rather than creating a duplicate on each pass.
#
# This is best-effort monitoring, not the guarantee: it can hit its own 20h limit or sit in the
# queue. The guarantee is the per-run sync chained with --dependency=afterany by
# submit_stage_a.sh, which runs once the training job ends however it ended.
#
#SBATCH --job-name=wandb-sync-watch
#SBATCH --account=rgx@cpu
#SBATCH --partition=prepost
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%j.out

set -uo pipefail

REPO="${SLURM_SUBMIT_DIR:-$WORK/code/robot-PointAct}"
cd "$REPO"

INTERVAL="${INTERVAL:-1800}"          # seconds between passes
# Only re-push runs written to recently. Generous relative to INTERVAL so a slow pass or a
# stalled writer does not make an active run look finished.
ACTIVE_WITHIN="${ACTIVE_WITHIN:-$(( INTERVAL / 30 + 60 ))}"   # minutes

echo "host=$(hostname) interval=${INTERVAL}s active_within=${ACTIVE_WITHIN}min"

# Stop shortly before the walltime rather than being killed mid-upload, which would leave a
# half-pushed run and a deleted .synced marker.
DEADLINE=$(( SECONDS + 20 * 3600 - 600 ))
pass=0
while [ $SECONDS -lt $DEADLINE ]; do
    pass=$(( pass + 1 ))
    echo "=== pass ${pass} ($(date '+%H:%M:%S')) ==="
    bash experiments/13_robocasa365/wandb_sync_jeanzay.sh --active-within "$ACTIVE_WITHIN" \
        2>&1 | tail -4
    sleep "$INTERVAL"
done
echo "reached the walltime guard; resubmit to keep watching"
