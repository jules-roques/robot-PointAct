#!/bin/bash
# Push offline W&B runs from Jean Zay to wandb.ai.
#
# Jean Zay compute nodes have no internet, so training runs with WANDB_MODE=offline and the
# curves only appear once they are synced from a LOGIN node. Run this there, as often as you
# want progress to show up:
#
#   bash experiments/13_robocasa365/wandb_sync_jeanzay.sh          # in-progress runs too
#   bash experiments/13_robocasa365/wandb_sync_jeanzay.sh --finished-only
#
# Note the directory layout: wandb creates its runs under $WANDB_DIR/wandb/offline-run-*, i.e.
# one level deeper than WANDB_DIR itself, so the obvious `wandb sync $SCRATCH/wandb/offline-run-*`
# silently matches nothing.

set -uo pipefail

RUN_ROOT="${WANDB_DIR:-$SCRATCH/wandb}/wandb"
FINISHED_ONLY=0
[ "${1:-}" = "--finished-only" ] && FINISHED_ONLY=1

if [ ! -d "$RUN_ROOT" ]; then
    echo "No W&B run directory at $RUN_ROOT" >&2
    exit 1
fi

REPO=""
for candidate in "${SLURM_SUBMIT_DIR:-}" "$PWD" "$HOME/code/robot-PointAct" "${WORK:-}/code/robot-PointAct"; do
    if [ -n "$candidate" ] && [ -d "$candidate/experiments/13_robocasa365" ]; then
        REPO="$candidate"
        break
    fi
done
[ -z "$REPO" ] && { echo "Could not locate the robot-PointAct checkout" >&2; exit 1; }
cd "$REPO"

export UV_OFFLINE=1
shopt -s nullglob
synced=0
skipped=0

for run in "$RUN_ROOT"/offline-run-*; do
    [ -d "$run" ] || continue
    marker=$(echo "$run"/run-*.wandb.synced)

    if [ -e "$marker" ]; then
        if [ "$FINISHED_ONLY" = "1" ]; then
            skipped=$((skipped + 1))
            continue
        fi
        # A run that is still training keeps growing after its first sync. wandb skips anything
        # carrying a .synced marker, so drop it to re-push; the run id is unchanged, so this
        # updates the same W&B run rather than creating a duplicate.
        rm -f "$marker"
    fi

    echo "--- syncing $(basename "$run")"
    if uv run --no-sync wandb sync "$run" 2>&1 | tail -3; then
        synced=$((synced + 1))
    else
        echo "    FAILED: $(basename "$run")" >&2
    fi
done

echo
echo "synced=${synced} skipped=${skipped} from ${RUN_ROOT}"
