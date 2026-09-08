#!/bin/bash
# What does a point policy cost as a function of the point budget? (Stage 7, cost half)
#
#   sbatch experiments/13_robocasa365/profile_stage7.slurm
#
# Two measurements, because "point policies are computationally expensive" is a claim about
# both the magnitude and where it comes from:
#
#   1. profile_point_cost.py -- one process, one GPU, explicit torch.cuda.synchronize() around
#      each region. Gives data loading / host-to-device / forward / backward / optimiser as a
#      mean AND a standard deviation over 30 timed iterations. This is the decomposition and
#      the spread.
#   2. pilot_throughput.py   -- the marginal s/step a real 4-GPU accelerate run sustains at
#      each budget, obtained by differencing two step counts so startup and CUDA autotuning
#      cancel. This is the number that predicts wall-clock.
#
# They answer different questions and (1) will read LOWER than (2): it has no gradient
# accumulation, no DDP all-reduce and no checkpoint writes. Quote both.
#
# The batch is pinned at 16 for the WHOLE sweep of (1). Holding it fixed is what makes the
# columns comparable; 16 rather than the base yaml's 32 because 32 does not fit at the top of
# the budget axis, and a sweep that changes batch mid-way measures two things at once.
#
# dev QoS: this is well under an hour and turns around in seconds, where t3 sits behind the
# training queue this same stage is filling.
#
#SBATCH --job-name=s7-profile
#SBATCH --account=rgx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --time=01:45:00
#SBATCH --output=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%j.out

set -euo pipefail

# `module` is a shell FUNCTION exported by the login profile; a job submitted over
# `ssh host 'sbatch ...'` never inherits it. Bootstrap from the paths z_modules.sh itself uses
# (the /etc/profile.d sibling is a no-op stub that defines `module` against an empty
# MODULEPATH, so every load then fails while `module` appears to work).
if ! type module >/dev/null 2>&1; then
    . /lustre/fshomisc/sup/spack_soft/environment-modules/current/init/bash
    export MODULEPATH=/lustre/fshomisc/sup/hpe/pub/module-rh/modulefiles:/lustre/fshomisc/sup/hpe/pub/modules-idris-env4/modulefiles/linux-rhel9-skylake_avx512
fi
module load arch/h100
module load ffmpeg/6.1.1

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

# Check every path up front rather than one queue wait at a time.
missing=""
for path in "$POINTACT_VLM_PATH" "$POINTACT_ENV/.venv" "$REPO/robot_data/robocasa365" \
            "$SCRATCH/models/Pointcept-Utonia/utonia.pth"; do
    [ -e "$path" ] || missing="$missing\n  $path"
done
if [ -n "$missing" ]; then
    echo -e "missing paths:$missing" >&2
    exit 1
fi
echo "paths ok; DSDIR=$DSDIR REPO=$REPO"

RUN=experiments/13_robocasa365/runs/s7-od-uniform-n4096-s0.yaml
BUDGETS="${BUDGETS:-1024 2048 4096 8192 16384}"
OUT="$SCRATCH/logs/robocasa365/s7_profile_${SLURM_JOB_ID}"
mkdir -p "$OUT"

echo "###############  1. step cost, decomposed (1 GPU)  ###############"
# One GPU on purpose: the decomposition is a per-process quantity, and DDP would fold an
# all-reduce into the backward column.
CUDA_VISIBLE_DEVICES=0 uv run --project "$POINTACT_ENV" --no-sync \
    python -u experiments/13_robocasa365/profile_point_cost.py "$RUN" \
        --npoints $BUDGETS --reps 30 --warmup 10 --batch-size 16 \
        --out "$OUT/point_cost.json"

echo
echo "###############  2. end-to-end marginal s/step (4 GPU)  ###############"
uv run --project "$POINTACT_ENV" --no-sync \
    python -u experiments/13_robocasa365/pilot_throughput.py \
        --contexts text_cache --npoints $BUDGETS \
        --per-device-batch 16 --effective-batch 128 \
        --short-steps 40 --long-steps 200 \
        --out "$OUT/pilot.json"

echo
echo "results in $OUT"
