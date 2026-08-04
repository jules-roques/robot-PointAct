#!/bin/bash
# Push offline W&B runs from Jean Zay to wandb.ai.
#
# Jean Zay compute nodes have no internet, so training runs with WANDB_MODE=offline and the
# curves only appear once they are synced from a LOGIN node. Run this there, as often as you
# want progress to show up:
#
#   bash experiments/13_robocasa365/wandb_sync_jeanzay.sh          # new + still-training runs
#   bash experiments/13_robocasa365/wandb_sync_jeanzay.sh --all    # force-repush everything
#   bash experiments/13_robocasa365/wandb_sync_jeanzay.sh --finished-only
#
# By default this pushes only what can have changed: runs never synced, and runs whose .wandb
# was written in the last $ACTIVE_WITHIN minutes (i.e. still training). A finished run that is
# already synced has nothing new to send, so re-pushing it is pure cost -- and that cost grows
# with the campaign, because every pass would re-upload every run ever recorded, including the
# ~150 MB rollout figures. Use --all only when a previous pass is known to have been corrupted
# or interrupted mid-run.
#
# Note the directory layout: wandb creates its runs under $WANDB_DIR/wandb/offline-run-*, i.e.
# one level deeper than WANDB_DIR itself, so the obvious `wandb sync $SCRATCH/wandb/offline-run-*`
# silently matches nothing.

set -uo pipefail

RUN_ROOT="${WANDB_DIR:-$SCRATCH/wandb}/wandb"
FINISHED_ONLY=0
# Re-push a synced run only if its .wandb changed in the last N minutes, i.e. it is still being
# written. 30 comfortably exceeds the checkpoint interval of the training jobs, so an in-flight
# run is never mistaken for a finished one; --all disables the check.
ACTIVE_WITHIN=30
# Retry dirs carrying a .sync-skip marker (see the loop below for what earns one).
RETRY_FAILED=0
# Per-run wall clock. Generous next to the ~2 min a healthy 30 MB run takes, tight enough that
# a wedged dir cannot swallow the pass.
SYNC_TIMEOUT="${SYNC_TIMEOUT:-10m}"
while [ $# -gt 0 ]; do
    case "$1" in
        --finished-only) FINISHED_ONLY=1 ;;
        --active-within) ACTIVE_WITHIN="$2"; shift ;;
        --all) ACTIVE_WITHIN="" ;;
        --retry-failed) RETRY_FAILED=1 ;;
        --timeout) SYNC_TIMEOUT="$2"; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

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

# Use the main checkout's venv explicitly. `uv run` resolves the project from the CWD, and
# when that is a worktree (which has no .venv) it builds an empty one and `wandb` is not found.
# See docs/envs.md.
WANDB_BIN="${POINTACT_ENV:-${WORK:-$HOME}/code/robot-PointAct}/.venv/bin/wandb"
if [ ! -x "$WANDB_BIN" ]; then
    echo "no wandb binary at $WANDB_BIN -- set POINTACT_ENV to a checkout that has a .venv" >&2
    exit 1
fi
shopt -s nullglob
synced=0
skipped=0
failed=0

for run in "$RUN_ROOT"/offline-run-*; do
    [ -d "$run" ] || continue
    marker=$(echo "$run"/run-*.wandb.synced)

    # A run that can never sync would otherwise be retried on every pass forever, because
    # "no .synced marker" is indistinguishable from "never attempted". See the note written
    # into .sync-skip for what that cost and how it corrupted live runs' configs.
    if [ -e "$run/.sync-skip" ] && [ "$RETRY_FAILED" = "0" ]; then
        skipped=$((skipped + 1))
        continue
    fi

    if [ -e "$marker" ]; then
        if [ "$FINISHED_ONLY" = "1" ]; then
            skipped=$((skipped + 1))
            continue
        fi
        # Already synced and no longer being written: nothing new to push.
        if [ -n "$ACTIVE_WITHIN" ] && \
           [ -z "$(find "$run" -name 'run-*.wandb' -mmin "-${ACTIVE_WITHIN}" -print -quit)" ]; then
            skipped=$((skipped + 1))
            continue
        fi
        # A run that is still training keeps growing after its first sync. wandb skips anything
        # carrying a .synced marker, so drop it to re-push; the run id is unchanged, so this
        # updates the same W&B run rather than creating a duplicate.
        rm -f "$marker"
    fi

    echo "--- syncing $(basename "$run")"
    # Bound each run. A dir that wedges at "setting up run" hangs indefinitely, and without a
    # timeout one such dir stalls the whole pass -- which is how a routine sync came to run for
    # over an hour without reaching the runs that actually had new data.
    if timeout "$SYNC_TIMEOUT" "$WANDB_BIN" sync "$run" 2>&1 | tail -3; then
        synced=$((synced + 1))
    else
        failed=$((failed + 1))
        echo "    FAILED (or timed out after ${SYNC_TIMEOUT}): $(basename "$run")" >&2
        # Do not let it burn another $SYNC_TIMEOUT on the next pass. Re-arm with --retry-failed
        # once the cause is understood (server-side run deleted, corrupt .wandb, proxy down).
        printf '%s\n' \
            "wandb sync failed or timed out at $(date -Is)." \
            "Skipped by wandb_sync_jeanzay.sh until you delete this file or pass --retry-failed." \
            > "$run/.sync-skip"
    fi
done

echo
if [ -n "$ACTIVE_WITHIN" ]; then
    mode="new runs + those written in the last ${ACTIVE_WITHIN} min"
else
    mode="ALL runs (--all)"
fi
echo "synced=${synced} skipped=${skipped} failed=${failed} from ${RUN_ROOT}  [${mode}]"
[ "$failed" -gt 0 ] && echo "failed dirs are now marked .sync-skip; re-arm with --retry-failed" >&2
exit 0
