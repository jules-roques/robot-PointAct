#!/bin/bash
# Jean Zay counterpart to eval.slurm (CLEPS) — separate SLURM controllers, so #SBATCH
# account/partition/module directives can't be shared; the payload (eval_robocasa365.sh) is.
# Runs on A100: the model uses FlashAttention in both the Qwen VLM and the PTv3 point backbone,
# which needs Ampere+ (fails on V100). H100 works too; A100 is the default for queue reasons.
#
# Smoke (3 trials + videos, short dev QoS):
#   sbatch --qos=qos_gpu_a100-dev --time=01:00:00 \
#          --export=ALL,CKPT_STEP=1000,NUM_TRIALS=3,OPTS="--args.save_video --args.verbose" \
#          experiments/13_robocasa365/eval_jeanzay.slurm
# Full (50 trials -> success rate):
#   sbatch --export=ALL,CKPT_STEP=final-48750 experiments/13_robocasa365/eval_jeanzay.slurm
#
#SBATCH --job-name=rc365-eval-od
#SBATCH --account=rgx@a100
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --time=10:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365_replay/%x_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365_replay/%x_%j.out

set -euo pipefail
# Load the A100 software stack (do NOT purge) so ffmpeg's libav* resolves for torchcodec and
# EGL is available for MuJoCo rendering.
module load arch/a100
module load ffmpeg/6.1.1

REPO="$HOME/code/robot-PointAct"
cd "$REPO"

RUN_DIR="${RUN_DIR:-$SCRATCH/PointAct_exprs/robocasa365/pointact/VLAEncDec3DWithActionRegressionModel-concerto-data-robocasa365-opendrawer-point-rot6d-image.leftright_ck16_lr5e-5_gpu4_bs32_epoch20-freeze.vlm}"
CKPT_DIR="${CKPT_DIR:-$RUN_DIR}"
CKPT_STEP="${CKPT_STEP:?set CKPT_STEP=<step or final-N> via --export}"
ENV_NAME="${ENV_NAME:-OpenDrawer}"
ROT="${ROT:-rot6d}"
OPTS="${OPTS:-}"

echo "host=$(hostname)  ckpt=${CKPT_DIR}/checkpoint-${CKPT_STEP}  trials=${NUM_TRIALS:-50}"
nvidia-smi -L

export NUM_TRIALS="${NUM_TRIALS:-50}"
bash experiments/13_robocasa365/eval_robocasa365.sh "$ENV_NAME" "$CKPT_DIR" "$CKPT_STEP" "$ROT" "$OPTS"
