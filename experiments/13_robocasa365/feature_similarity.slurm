#!/bin/bash
# Dump full-resolution point-feature cosine similarity from a trained checkpoint (Stage 6).
#
#   sbatch --export=ALL,CKPT=$SCRATCH/PointAct_exprs/robocasa365/ablation/od-none-s0/checkpoint-final-50000 \
#          experiments/13_robocasa365/feature_similarity.slurm
#
# Needs the per-point label LMDB (points_3views_labels) that the oracle sampling arm uses --
# this arm never opens it during training, so it is a preflight check here rather than an
# assumption.
#
# Pure torch inference on a handful of frames -- no simulator, so H100 is fine (see
# docs/clusters/jean-zay.md) and dev QoS turns around fastest. Minutes of GPU.
#
#SBATCH --job-name=s6-featsim
#SBATCH --account=rgx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=00:40:00
#SBATCH --output=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%j.out

set -euo pipefail

# See docs/clusters/jean-zay.md: `module` and $DSDIR come from the LOGIN profile, so a job
# submitted over `ssh host 'sbatch ...'` has neither.
if ! type module >/dev/null 2>&1; then
    . /lustre/fshomisc/sup/spack_soft/environment-modules/current/init/bash
    export MODULEPATH=/lustre/fshomisc/sup/hpe/pub/module-rh/modulefiles:/lustre/fshomisc/sup/hpe/pub/modules-idris-env4/modulefiles/linux-rhel9-skylake_avx512
fi
module load arch/h100
module load ffmpeg/6.1.1
module list 2>&1
: "${DSDIR:=/lustre/fsmisc/dataset}"

REPO="${SLURM_SUBMIT_DIR:-$WORK/code/robot-PointAct}"
POINTACT_ENV="${POINTACT_ENV:-$WORK/code/robot-PointAct}"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export UV_OFFLINE=1
export WANDB_MODE=disabled
export POINTACT_VLM_PATH="$DSDIR/HuggingFace_Models/Qwen/Qwen2.5-VL-3B-Instruct"
[ -e robot_data ] || ln -s "$SCRATCH/datasets/robot_data" robot_data

CKPT="${CKPT:?set CKPT=<checkpoint dir> via --export}"
FRAMES="${FRAMES:-4}"
OUT="${OUT:-$SCRATCH/jz_tmp/s6_featsim_$(basename "$CKPT").npz}"
mkdir -p "$(dirname "$OUT")"

LABELS="${LABELS:-points_3views_labels}"
for path in "$CKPT/config.json" "$CKPT/../data_config.yaml" "$POINTACT_ENV/.venv" \
            "robot_data/robocasa365/lerobot_point_lmdb/OpenDrawer/$LABELS"; do
    [ -e "$path" ] || { echo "missing: $path" >&2; exit 1; }
done

uv run --project "$POINTACT_ENV" --no-sync \
    python experiments/13_robocasa365/feature_similarity.py "$CKPT" \
        --frames "$FRAMES" --label-dirname "$LABELS" --out "$OUT"
echo "OUT=$OUT"
