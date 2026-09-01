#!/bin/bash
# What does ptv3_apply_point_ca actually cost? (Stage 7 go/no-go)
#
#   sbatch experiments/13_robocasa365/profile_point_ca.slurm
#
# Stage 6 and every arm before it ran with the point branch of each CABlock disabled, so the
# cross-attention weights those checkpoints carry (q/kv/proj, ~15 M) are trained by the 17
# action tokens alone. Turning the flag on routes all ~19 K points through those SAME
# projections and adds a LayerNorm pair + a 4x MLP per CA block -- ~25 M new parameters, and
# per point per block roughly 10*C^2 MACs on top of the 12*C^2 the self-attention block
# already spends.
#
# The analytic estimate lands near +20% of encoder FLOPs, but this network is not FLOP-bound:
# the xCPE submanifold convolution carries ~57% of the MACs at a fraction of a GEMM's
# efficiency, and the measured point-count scaling is strongly sublinear (4.6x points cost
# 1.46x step time). So the analytic number is a prior, not an answer. Measure it.
#
# Two settings, matched at per_device_train_batch_size=16 (what Stage 6 used) and effective
# batch 128, marginal rate across 20 and 80 steps so startup and warm-up cancel. If the
# ca=true cell OOMs at 16, that is itself the answer -- a second pass at 8 then gives the
# ratio at a batch both settings fit in.
#
# dev QoS: ~35 minutes of work, and dev turns around far faster than t3.
#
#SBATCH --job-name=s7-ca-cost
#SBATCH --account=rgx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

set -euo pipefail

# `module` is a shell FUNCTION, exported by the login profile and carried into the job by
# --export=ALL -- but only if the SUBMITTING shell had it, which `ssh host 'sbatch ...'` does
# not. Bootstrap it when it is missing, from the shared lustre install rather than from
# /etc/profile.d: z_modules.sh (the file that does this on the login nodes) is not present on
# the compute nodes, and its sibling modules.sh is a deliberate no-op stub that defines
# `module` against an empty MODULEPATH -- so every load then fails with "Unable to locate a
# modulefile" while `module` itself appears to work. Both were tried; these are the paths
# z_modules.sh itself uses.
if ! type module >/dev/null 2>&1; then
    . /lustre/fshomisc/sup/spack_soft/environment-modules/current/init/bash
    export MODULEPATH=/lustre/fshomisc/sup/hpe/pub/module-rh/modulefiles:/lustre/fshomisc/sup/hpe/pub/modules-idris-env4/modulefiles/linux-rhel9-skylake_avx512
fi
module load arch/h100
module load ffmpeg/6.1.1
# Fail here rather than 20 minutes in: a missing module shows up as an import error deep in
# torch, which reads as an env problem rather than a module one.
module list 2>&1

# Same inheritance story as `module` above: $DSDIR is set by the IDRIS *login* profile, so a
# job submitted with `ssh host 'sbatch ...'` never sees it and `set -u` kills the script on
# first use. $WORK and $SCRATCH survive (they come from ~/.bashrc, which ssh does source).
# Defaulting a cluster path is fine HERE -- this is a slurm script; see CLAUDE.md.
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

# Check every path up front. Three submissions of this job died one line at a time on an
# environment the login node has and the compute node does not; each cost a queue wait to
# discover. Fail on all of them at once instead.
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

OUT="$SCRATCH/logs/robocasa365/s7_ca_cost_${SLURM_JOB_ID}"
mkdir -p "$OUT"

echo "###############  matched at per-device batch 16  ###############"
uv run --project "$POINTACT_ENV" --no-sync \
    python experiments/13_robocasa365/pilot_throughput.py \
        --contexts text_cache --npoints 32768 \
        --point-ca false true \
        --per-device-batch 16 --effective-batch 128 \
        --short-steps 20 --long-steps 80 \
        --out "$OUT/ca16.json" || true

# Did the ca=true cell survive? Only fall back to a smaller batch if it did not -- the
# matched-at-16 ratio is the one that answers the question, and re-running both settings at 8
# costs another ~20 minutes.
if grep -q '"error"' "$OUT/ca16.json" 2>/dev/null || [ ! -s "$OUT/ca16.json" ]; then
    echo
    echo "###############  a cell failed at 16; retrying both at 8  ###############"
    uv run --project "$POINTACT_ENV" --no-sync \
        python experiments/13_robocasa365/pilot_throughput.py \
            --contexts text_cache --npoints 32768 \
            --point-ca false true \
            --per-device-batch 8 --effective-batch 128 \
            --short-steps 20 --long-steps 80 \
            --out "$OUT/ca8.json" || true
fi

echo
echo "results in $OUT"
