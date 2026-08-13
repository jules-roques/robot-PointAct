#!/bin/bash
# Archive a run's TRAINING artifacts to a durable location before SCRATCH purges them (~30 days
# of no access). Training is the expensive, hard-to-reproduce part; eval is cheap and can be
# re-run, so results/ is deliberately NOT archived. Tars the FINAL checkpoint (weights + config
# + norm stats) into:
#   $STORE/PointAct/robocasa365/<run>.tar   (Jean Zay, permanent+backed-up, low inode quota)
#   $HOME/archives/PointAct/robocasa365/<run>.tar   (CLEPS, no $STORE; $HOME is durable instead)
# Either way, one tar per run (not loose files) to conserve inode/space quota. Training curves
# are kept separately via `wandb sync` (see README).
#
# Usage: bash experiments/13_robocasa365/archive_run.sh [run_dir]
#   run_dir defaults to the concerto OpenDrawer run.
set -euo pipefail

DEFAULT_RUN="$SCRATCH/PointAct_exprs/robocasa365/pointact/VLAEncDec3DWithActionRegressionModel-concerto-data-robocasa365-opendrawer-point-rot6d-image.leftright_ck16_lr5e-5_gpu4_bs32_epoch20-freeze.vlm"
RUN_DIR="${1:-$DEFAULT_RUN}"
[ -d "$RUN_DIR" ] || { echo "run dir not found: $RUN_DIR" >&2; exit 1; }

RUN_NAME=$(basename "$RUN_DIR")
DEST_DIR="${STORE:+$STORE/PointAct/robocasa365}"
DEST_DIR="${DEST_DIR:-$HOME/archives/PointAct/robocasa365}"
mkdir -p "$DEST_DIR"
DEST="$DEST_DIR/$RUN_NAME.tar"

cd "$RUN_DIR"
FINAL_CKPT=$(ls -d checkpoint-final-* 2>/dev/null | head -1 || true)
[ -n "$FINAL_CKPT" ] || { echo "no checkpoint-final-* found in $RUN_DIR" >&2; exit 1; }

# Only the final checkpoint (weights + config + norm stats). Eval results/ are cheap to
# regenerate and deliberately excluded.
echo "archiving : $FINAL_CKPT"
echo "  from    : $RUN_DIR"
echo "  to      : $DEST"
# Plain (uncompressed) tar: safetensors are already ~incompressible, and it keeps the archive
# fast to write and cheap to partially extract.
tar -cf "$DEST" "$FINAL_CKPT"
echo "done      : $(du -sh "$DEST" | cut -f1)  $DEST"
echo "restore   : tar -xf '$DEST' -C <target_dir>"
