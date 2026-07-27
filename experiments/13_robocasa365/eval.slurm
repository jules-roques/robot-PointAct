#!/bin/bash
# Evaluate a trained PointAct checkpoint on a RoboCasa365 task (server + client on one GPU).
# Runs on A100: the model uses FlashAttention in both the Qwen VLM and the PTv3 point backbone,
# which needs Ampere+ (fails on V100), while MuJoCo/EGL rendering runs fine on A100. (H100 is
# avoided because RoboCasa365 does not run correctly there.)
#
# Smoke (3 trials + videos, short dev QoS):
#   sbatch --qos=qos_gpu_a100-dev --time=01:00:00 \
#          --export=ALL,NUM_TRIALS=3,OPTS="--args.save_video --args.verbose" \
#          experiments/13_robocasa365/eval.slurm
# Full (50 trials -> success rate):
#   sbatch experiments/13_robocasa365/eval.slurm
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

REPO=/lustre/fswork/projects/rech/rgx/unw24jl/code/robot-PointAct
cd "$REPO"

# Which checkpoint to evaluate. Override any of these via --export.
RUN_DIR="$SCRATCH/PointAct_exprs/robocasa365/pointact/VLAEncDec3DWithActionRegressionModel-concerto-data-robocasa365-opendrawer-point-rot6d-image.leftview_ck16_lr5e-5_gpu4_bs32_epoch50-freeze.vlm"
CKPT_DIR="${CKPT_DIR:-$RUN_DIR}"
CKPT_STEP="${CKPT_STEP:-final-48750}"
ENV_NAME="${ENV_NAME:-OpenDrawer}"
ROT="${ROT:-rot6d}"
OPTS="${OPTS:-}"

echo "host=$(hostname)  ckpt=${CKPT_DIR}/checkpoint-${CKPT_STEP}  trials=${NUM_TRIALS:-50}"
nvidia-smi -L

# eval_robocasa365.sh sets MUJOCO_GL/HF offline flags, loads ffmpeg, and manages the
# server (root env) + client (robocasa365 env) lifecycle. NUM_TRIALS is read from the env.
export NUM_TRIALS="${NUM_TRIALS:-50}"
bash experiments/13_robocasa365/eval_robocasa365.sh "$ENV_NAME" "$CKPT_DIR" "$CKPT_STEP" "$ROT" "$OPTS"
