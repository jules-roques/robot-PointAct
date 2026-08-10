#!/bin/bash
# Quantify MolmoPoint's detection accuracy against simulator ground truth (CPU, root env).
#
#   sbatch --export=ALL,TASK=OpenDrawer      experiments/13_robocasa365/molmo_accuracy_jeanzay.slurm
#   sbatch --export=ALL,TASK=TurnOnMicrowave experiments/13_robocasa365/molmo_accuracy_jeanzay.slurm
#   sbatch --export=ALL,TASK=PickPlaceCounterToStove,QUERY=1 <this file>   # 0=object, 1=pan
#
# CPU only: this reads the anchor cache, the point/label LMDBs and a calibration, and does
# geometry. No model runs -- the 8B forwards were paid once when the cache was built.
#
# Ground truth differs by task and the script picks neither for you:
#   OpenDrawer       --gt labels  (MuJoCo handle centroid; the GT-oracle arm's own anchor)
#   TurnOnMicrowave  --gt geom    (start_button geom, from target_positions_jeanzay.slurm)
#   PickPlace...     --gt geom    (obj / pan object centroids, same job)
# Run that job first for the two --gt geom tasks; neither has a label LMDB and neither will.
#
#SBATCH --job-name=molmo-acc
#SBATCH --account=rgx@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-dev
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%j.out

set -euo pipefail

MAIN_REPO=/lustre/fswork/projects/rech/rgx/unw24jl/code/robot-PointAct
REPO="${SLURM_SUBMIT_DIR:-$MAIN_REPO}"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"   # shadow the installed pointact with the worktree

TASK="${TASK:?set TASK=<OpenDrawer|TurnOnMicrowave> via --export}"
DATA_ROOT="${DATA_ROOT:-$SCRATCH/datasets/robot_data/robocasa365/lerobot_point_lmdb}"
DATASET="$DATA_ROOT/$TASK"
# PickPlaceCounterToStove points twice per frame, so its default output carries the query
# index. Without that the second run would overwrite the first and the two files would
# describe different anchors under the same name.
QUERY="${QUERY:-0}"
OUT_DEFAULT="$DATASET/roi_meta/molmo_accuracy.json"
[ "$TASK" = PickPlaceCounterToStove ] && \
    OUT_DEFAULT="$DATASET/roi_meta/molmo_accuracy_q${QUERY}.json"
OUT="${OUT:-$OUT_DEFAULT}"
# Derive the row dump from OUT rather than hardcoding it: with a fixed name, scoring a
# variant cache (a re-lift, a different fusion) silently overwrites the baseline's rows
# while leaving its JSON in place, so the two stop describing the same run.
ROWS="${ROWS:-${OUT%.json}_rows.npz}"

ARGS=(--dataset-dir "$DATASET" --out "$OUT" --dump-rows "$ROWS")
case "$TASK" in
    OpenDrawer)
        ARGS+=(--gt labels)
        ;;
    TurnOnMicrowave)
        ARGS+=(--gt geom --target start_button --distractors stop_button microwave)
        ;;
    PickPlaceCounterToStove)
        # One cache, two pointing queries: 0 is the food item the instruction names, 1 is the
        # pan it goes into. Each is scored against its own geom set with the other as the
        # distractor -- "pointed at the pan when asked for the apple" is the failure that
        # matters for the two-anchor arm, and a distance alone cannot show it. Query 0 is what
        # the one-anchor arm was trained on.
        case "$QUERY" in
            0) ARGS+=(--gt geom --target obj --distractors pan --query-id 0) ;;
            1) ARGS+=(--gt geom --target pan --distractors obj --query-id 1) ;;
            *) echo "QUERY must be 0 (object) or 1 (pan), got '$QUERY'" >&2; exit 1 ;;
        esac
        ;;
    *)
        echo "no ground-truth source configured for $TASK" >&2; exit 1 ;;
esac
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=(${EXTRA_ARGS})

echo "host=$(hostname) task=$TASK dataset=$DATASET out=$OUT"

"$MAIN_REPO/.venv/bin/python" -m data_prep.roi_sampling.eval_molmo_accuracy "${ARGS[@]}"
