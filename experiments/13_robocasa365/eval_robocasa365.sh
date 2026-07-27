#!/bin/bash
# Evaluate a PointAct checkpoint on a RoboCasa365 task via server (pointact env) + client
# (robocasa365 env). Adapted from experiments/2_libero/eval_libero.sh to uv environments.
#
# Both processes need the SAME GPU (server = model inference, client = MuJoCo/EGL rendering).
# Needs Ampere+ (A100): both the Qwen VLM and the PTv3 backbone use FlashAttention, which fails
# on V100/pre-Ampere; H100 is avoided separately because RoboCasa365 sim misbehaves there. See
# experiments/13_robocasa365/eval.slurm for the CLEPS sbatch invocation.
#
# Usage: eval_robocasa365.sh <env_name> <ckpt_dir> <ckpt_step> <pred_rot_type> [client_opts]

env_name=$1        # e.g. OpenDrawer
ckpt_dir=$2
ckpt_step=$3
pred_rot_type=$4   # rot6d, euler
options=$5         # e.g. " --args.save_video --args.verbose"

host=localhost
port=$((10000 + RANDOM % 10000))
seed=7
num_denoise_steps=10

# $HOME differs per cluster but "code/robot-PointAct" underneath it doesn't, so this resolves
# correctly on both Jean Zay and CLEPS without editing this file per-cluster.
REPO="$HOME/code/robot-PointAct"
cd "$REPO"
export PYTHONPATH=$(pwd):$PYTHONPATH

# Client-side rendering + offline (treat compute nodes as if they had no network, for
# reproducibility with Jean Zay even though CLEPS compute nodes do have internet access).
export MUJOCO_GL=egl
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_OFFLINE=1
# Jean Zay provides ffmpeg via `module load ffmpeg/6.1.1` (done by the submitting slurm script).
# CLEPS has no such module — torchcodec/av's libav* come from a dedicated conda env instead:
# conda create -n ffmpeg-libs -c conda-forge ffmpeg=6.1. Detect via $DSDIR (Jean Zay-only).
if [ -z "${DSDIR:-}" ]; then
  export LD_LIBRARY_PATH="$HOME/miniforge3/envs/ffmpeg-libs/lib:${LD_LIBRARY_PATH:-}"
fi

set -euo pipefail

cleanup() {
  echo "[cleanup] stopping server..."
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 2
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

is_used() { lsof -i TCP:$1 >/dev/null 2>&1; }
while is_used "$port"; do port=$((10000 + RANDOM % 10000)); done
echo "port=$port"

# --- Policy server (pointact / root env) ---
uv run --no-sync scripts/run_server.py \
    --args.seed ${seed} \
    --args.pretrained_path ${ckpt_dir}/checkpoint-${ckpt_step} \
    --args.num_denoise_steps ${num_denoise_steps} \
    --args.host ${host} --args.port ${port} &
SERVER_PID=$!
echo "Server started, PID=$SERVER_PID"
sleep 20

# --- Sim client (robocasa365 env) ---
uv run --project envs/robocasa365 --no-sync \
    experiments/13_robocasa365/run_robocasa365_client.py \
    --args.seed ${seed} \
    --args.host ${host} --args.port ${port} \
    --args.env_name ${env_name} \
    --args.num_trials ${NUM_TRIALS:-50} \
    --args.pred_rot_type ${pred_rot_type} \
    --args.replan_steps 8 \
    --args.use_depth \
    --args.save_dir ${ckpt_dir}/results/checkpoint-${ckpt_step} \
    ${options}

echo ${ckpt_dir}/checkpoint-${ckpt_step}
echo "Client finished"

# Tolerate the server's kill signal so `set -e` doesn't mark the job FAILED after a clean run.
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
echo "Server exited"
