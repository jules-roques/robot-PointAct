#!/bin/bash
# Quantify MolmoPoint's detection accuracy against simulator ground truth (CPU, root env).
#
#   sbatch --export=ALL,TASK=OpenDrawer      experiments/13_robocasa365/molmo_accuracy_jeanzay.slurm
#   sbatch --export=ALL,TASK=TurnOnMicrowave experiments/13_robocasa365/molmo_accuracy_jeanzay.slurm
#
# CPU only: this reads the anchor cache, the point/label LMDBs and a calibration, and does
# geometry. No model runs -- the 8B forwards were paid once when the cache was built.
#
# Ground truth differs by task and the script picks neither for you:
#   OpenDrawer       --gt labels  (MuJoCo handle centroid; the GT-oracle arm's own anchor)
#   TurnOnMicrowave  --gt geom    (start_button geom, from target_positions_jeanzay.slurm)
# Run that job first for TurnOnMicrowave; there is no label LMDB for it and there will not be.
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
OUT="${OUT:-$DATASET/roi_meta/molmo_accuracy.json}"

ARGS=(--dataset-dir "$DATASET" --out "$OUT")
case "$TASK" in
    OpenDrawer)
        ARGS+=(--gt labels)
        ;;
    TurnOnMicrowave)
        ARGS+=(--gt geom --target start_button --distractors stop_button microwave)
        ;;
    *)
        echo "no ground-truth source configured for $TASK" >&2; exit 1 ;;
esac
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=(${EXTRA_ARGS})

echo "host=$(hostname) task=$TASK dataset=$DATASET out=$OUT"

"$MAIN_REPO/.venv/bin/python" -m data_prep.roi_sampling.eval_molmo_accuracy "${ARGS[@]}"
