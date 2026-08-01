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
# Overridable so a follow-up eval can draw FRESH scenes. Each trial pulls the next scene from
# this seeded stream, so re-running with the same seed replays the same episodes -- results
# from two runs at the same seed must not be pooled as if they were independent trials.
seed=${SEED:-7}
num_denoise_steps=10

# The checkout is at $HOME/code on CLEPS but $WORK/code on Jean Zay, so probe rather than
# assume (the previous $HOME-only form simply failed on Jean Zay).
REPO=""
for candidate in "${SLURM_SUBMIT_DIR:-}" "$PWD" "$HOME/code/robot-PointAct" "${WORK:-}/code/robot-PointAct"; do
    if [ -n "$candidate" ] && [ -d "$candidate/experiments/13_robocasa365" ]; then
        REPO="$candidate"
        break
    fi
done
[ -z "$REPO" ] && { echo "Could not locate the robot-PointAct checkout" >&2; exit 1; }
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

# --- How was this checkpoint trained to sample points? -------------------------------------
# A policy trained on clouds concentrated near the end-effector or near the task's handle must
# be evaluated the same way; sampling uniformly instead is a train/test mismatch that looks
# like a drop in success rate. Derive it from the run's own training_args.json -> data config
# rather than relying on the caller to pass a matching flag by hand.
POINT_SAMPLING="${POINT_SAMPLING:-}"
ORACLE_ANCHOR=""
if [ -z "$POINT_SAMPLING" ]; then
    DATA_CFG=$(python3 -c "
import json,sys
try:
    print(json.load(open('${ckpt_dir}/training_args.json')).get('data_path',''))
except Exception:
    print('')
" 2>/dev/null)
    if [ -n "$DATA_CFG" ] && [ -f "$DATA_CFG" ]; then
        if grep -qE '^\s*oracle_sampling:\s*True' "$DATA_CFG"; then
            POINT_SAMPLING=anchor
        elif grep -qE '^\s*eef_sampling:\s*True' "$DATA_CFG"; then
            POINT_SAMPLING=eef
        else
            POINT_SAMPLING=uniform
        fi
        echo "derived point_sampling=${POINT_SAMPLING} from ${DATA_CFG}"
    else
        POINT_SAMPLING=uniform
        echo "WARNING: no data config found for ${ckpt_dir}; defaulting to uniform sampling" >&2
    fi
fi
[ "$POINT_SAMPLING" = "anchor" ] && ORACLE_ANCHOR="--args.oracle_anchor"

# --- Policy server (pointact / root env) ---
uv run --no-sync scripts/run_server.py \
    --args.seed ${seed} \
    --args.pretrained_path ${ckpt_dir}/checkpoint-${ckpt_step} \
    --args.num_denoise_steps ${num_denoise_steps} \
    --args.point_sampling ${POINT_SAMPLING} \
    ${TEXT_CONTEXT_FILE:+--args.text_context_file ${TEXT_CONTEXT_FILE}} \
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
    ${ORACLE_ANCHOR} \
    ${options}

echo ${ckpt_dir}/checkpoint-${ckpt_step}
echo "Client finished"

# Tolerate the server's kill signal so `set -e` doesn't mark the job FAILED after a clean run.
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
echo "Server exited"
