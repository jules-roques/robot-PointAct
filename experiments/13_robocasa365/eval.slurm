#!/bin/bash
# Evaluate a trained PointAct checkpoint on a RoboCasa365 task (server + client on one GPU).
# Runs on A100: the model uses FlashAttention in both the Qwen VLM and the PTv3 point backbone,
# which needs Ampere+ (fails on V100). H100 works too; A100 is the default for queue reasons.
#
# Smoke (3 trials + videos, short walltime):
#   sbatch --time=01:00:00 \
#          --export=ALL,CKPT_STEP=1000,NUM_TRIALS=3,OPTS="--args.save_video --args.verbose" \
#          experiments/13_robocasa365/eval.slurm
# Full (50 trials -> success rate):
#   sbatch --export=ALL,CKPT_STEP=final-48750 experiments/13_robocasa365/eval.slurm
#
#SBATCH --job-name=rc365-eval-od
#SBATCH --account=willow
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/jroques/logs/robocasa365/%x_%j.out
#SBATCH --error=/scratch/jroques/logs/robocasa365/%x_%j.out

set -euo pipefail

REPO="$HOME/code/robot-PointAct"
cd "$REPO"

# Which checkpoint to evaluate. Override any of these via --export (CKPT_DIR points at a
# specific run's output_dir; the two RUN_DIR fallbacks below are for the baseline / eef-density
# runs launched from this checkout, named after their data-config basenames).
RUN_DIR="${RUN_DIR:-$SCRATCH/PointAct_exprs/robocasa365/pointact/VLAEncDec3DWithActionRegressionModel-concerto-data-robocasa365-opendrawer-point-rot6d-image.leftright_ck16_lr5e-5_gpu4_bs32_epoch20-freeze.vlm}"
CKPT_DIR="${CKPT_DIR:-$RUN_DIR}"
CKPT_STEP="${CKPT_STEP:?set CKPT_STEP=<step or final-N> via --export}"
ENV_NAME="${ENV_NAME:-OpenDrawer}"
ROT="${ROT:-rot6d}"
OPTS="${OPTS:-}"

echo "host=$(hostname)  ckpt=${CKPT_DIR}/checkpoint-${CKPT_STEP}  trials=${NUM_TRIALS:-50}"
nvidia-smi -L

# eval_robocasa365.sh sets MUJOCO_GL/HF offline flags, ffmpeg LD_LIBRARY_PATH, and manages the
# server (root env) + client (robocasa365 env) lifecycle. NUM_TRIALS is read from the env.
export NUM_TRIALS="${NUM_TRIALS:-50}"
bash experiments/13_robocasa365/eval_robocasa365.sh "$ENV_NAME" "$CKPT_DIR" "$CKPT_STEP" "$ROT" "$OPTS"
