#!/bin/bash
# Stage 0 (login node, INTERNET required): build the isolated ROI preprocessing venv
# and pre-bake the YOLO-World prompt. Kept separate from the training .venv so the
# training environment (and the fair-comparison baseline) is untouched.
#
#   bash data_prep/roi_sampling/setup_roi_env.sh
#
set -euo pipefail

REPO=/lustre/fswork/projects/rech/rgx/unw24jl/code/robot-PointAct/.claude/worktrees/roi-guided-sampling
cd "$REPO"

VENV="$REPO/.venv-roi"
WEIGHTS_DIR="$SCRATCH/models/YOLO-World"
BAKED="$WEIGHTS_DIR/yoloworld_s_drawer.pt"

echo "== creating $VENV =="
uv venv "$VENV" --python 3.10

echo "== installing deps (torch pulled by ultralytics) =="
uv pip install --python "$VENV" \
    ultralytics plotly "av>=14" lmdb msgpack msgpack-numpy "numpy<2" tqdm

echo "== baking YOLO-World prompt -> $BAKED =="
mkdir -p "$WEIGHTS_DIR"
cd "$WEIGHTS_DIR"   # let ultralytics drop its auto-downloads here
PYTHONPATH="$REPO" "$VENV/bin/python" "$REPO/data_prep/roi_sampling/bake_yoloworld.py" \
    --prompt "drawer" --out "$BAKED"

echo "== done. venv=$VENV baked=$BAKED =="
