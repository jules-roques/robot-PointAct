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

# SAVE_SUFFIX appends to the results directory only, not the checkpoint path. A rollout-only
# pass sets it to "-viz" so its handful of episodes land in checkpoint-<step>-viz and cannot be
# pooled into the real success rate (log_eval_to_wandb skips non-numeric suffixes but still
# picks the figures up).
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

# REPO may be a worktree, which has no .venv of its own. `uv run` resolves the project from the
# CWD, so it would build an empty environment there and the server starts without numpy and the
# client without imageio -- the failure mode that killed the first stage-A eval sweep. Resolve
# the environments from a checkout that actually has them; PYTHONPATH above is what makes the
# worktree's own code shadow the installed package (docs/envs.md).
# Probe by CAPABILITY, not by path: a stray `uv run` in a worktree leaves an EMPTY .venv
# behind, and testing only for .venv/bin/python happily selects it -- which is exactly how the
# first fix for this still produced "No module named numpy".
env_has() { [ -x "$1/bin/python" ] && "$1/bin/python" -c "import $2" >/dev/null 2>&1; }

POINTACT_ENV="${POINTACT_ENV:-}"
if [ -z "$POINTACT_ENV" ]; then
    for candidate in "${WORK:-}/code/robot-PointAct" "$HOME/code/robot-PointAct" "$REPO"; do
        if [ -n "$candidate" ] && env_has "$candidate/.venv" numpy; then
            POINTACT_ENV="$candidate"
            break
        fi
    done
fi
[ -z "$POINTACT_ENV" ] && { echo "No checkout whose .venv can import numpy" >&2; exit 1; }

ROBOCASA_ENV="${ROBOCASA_ENV:-}"
if [ -z "$ROBOCASA_ENV" ]; then
    for candidate in "$POINTACT_ENV" "${WORK:-}/code/robot-PointAct" "$HOME/code/robot-PointAct"; do
        if [ -n "$candidate" ] && env_has "$candidate/envs/robocasa365/.venv" imageio; then
            ROBOCASA_ENV="$candidate/envs/robocasa365"
            break
        fi
    done
fi
[ -z "$ROBOCASA_ENV" ] && { echo "No envs/robocasa365/.venv that can import imageio" >&2; exit 1; }
echo "pointact env=$POINTACT_ENV  robocasa env=$ROBOCASA_ENV"

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
    # Prefer the copy archived beside the checkpoint: every run writes data_config.yaml into
    # its own output_dir, so the mode is recoverable without depending on a repo file that may
    # have moved since. Fall back to the data_path recorded in training_args.json, which is how
    # runs launched before that existed are resolved.
    DATA_CFG="${ckpt_dir}/data_config.yaml"
    if [ ! -f "$DATA_CFG" ]; then
        DATA_CFG=$(python3 -c "
import json
try:
    print(json.load(open('${ckpt_dir}/training_args.json')).get('data_path',''))
except Exception:
    print('')
" 2>/dev/null)
    fi

    if [ -n "$DATA_CFG" ] && [ -f "$DATA_CFG" ]; then
        # yaml booleans are True/true depending on writer (hand-edited vs yaml.safe_dump).
        if grep -qiE '^\s*oracle_sampling:\s*true' "$DATA_CFG"; then
            POINT_SAMPLING=anchor
        elif grep -qiE '^\s*eef_sampling:\s*true' "$DATA_CFG"; then
            POINT_SAMPLING=eef
        else
            # "uniform" must be a positive finding, not the leftover case. Listing only the
            # arms we know how to reproduce at eval time made this an open-world assumption:
            # `molmo_sampling: true` matched neither branch above and fell through to uniform,
            # so all four stage-3 arms were trained on MolmoPoint-centred clouds and then
            # evaluated on uniform ones -- exactly the mismatch the block below calls fatal,
            # arrived at silently. Any *_sampling knob we do not have an eval-time producer
            # for is now an error, so the next arm added to the dataloader cannot repeat it.
            UNKNOWN=$(grep -oiE '^\s*[a-z0-9_]+_sampling:\s*true' "$DATA_CFG" \
                      | grep -oiE '[a-z0-9_]+_sampling' \
                      | grep -viE '^(oracle|eef)_sampling$' | sort -u | tr '\n' ' ')
            if [ -n "$UNKNOWN" ]; then
                echo "ERROR: ${DATA_CFG} enables a sampling arm this script cannot reproduce" >&2
                echo "  at eval time: ${UNKNOWN}" >&2
                echo "  Evaluating it as 'uniform' is a train/test mismatch that reads as a" >&2
                echo "  plausible but wrong success rate. Add an anchor producer for it, or" >&2
                echo "  pass POINT_SAMPLING=... explicitly if you really mean to mismatch." >&2
                exit 1
            fi
            POINT_SAMPLING=uniform
        fi
        echo "derived point_sampling=${POINT_SAMPLING} from ${DATA_CFG}"
    else
        # Deliberately fatal. Defaulting to uniform here silently evaluates an eef- or
        # oracle-trained policy on uniformly drawn clouds -- a train/test mismatch that shows
        # up as a plausible-looking but wrong success rate, i.e. the one failure mode this
        # whole ablation cannot afford. Pass POINT_SAMPLING=... explicitly if you mean it.
        echo "ERROR: cannot determine the sampling mode for ${ckpt_dir}." >&2
        echo "  Looked for ${ckpt_dir}/data_config.yaml and the data_path in training_args.json." >&2
        echo "  Set POINT_SAMPLING=uniform|eef|anchor explicitly to override." >&2
        exit 1
    fi
fi
[ "$POINT_SAMPLING" = "anchor" ] && ORACLE_ANCHOR="--args.oracle_anchor"

# --- Policy server (pointact / root env) ---
uv run --project "$POINTACT_ENV" --no-sync scripts/run_server.py \
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
# repo_id must be the one the run TRAINED on: the processor's per-repo tables
# (select_video_keys_for_vlm, select_state_keys, features, norm stats) are keyed by it, so a
# mismatch is KeyError on the server and every trial is recorded as a skipped scene. The
# client's own default is "OpenDrawer", which silently produced 0/100 on all four stage-2 runs.
# env_name is itself read from the run's data config repo_id, so passing it through is exact.
repo_id=${REPO_ID:-$env_name}

uv run --project "$ROBOCASA_ENV" --no-sync \
    experiments/13_robocasa365/run_robocasa365_client.py \
    --args.seed ${seed} \
    --args.host ${host} --args.port ${port} \
    --args.env_name ${env_name} \
    --args.repo_id ${repo_id} \
    --args.num_trials ${NUM_TRIALS:-50} \
    --args.pred_rot_type ${pred_rot_type} \
    --args.replan_steps 8 \
    --args.use_depth \
    --args.save_dir ${ckpt_dir}/results/checkpoint-${ckpt_step}${SAVE_SUFFIX:-} \
    ${VIZ_ROLLOUTS:+--args.viz_rollouts --args.point_sampling_for_viz ${POINT_SAMPLING}} \
    ${ORACLE_ANCHOR} \
    ${options}

echo "Client finished"

# Render the rollout dumps in the POINTACT env: the sim env the client ran in has neither
# plotly nor lmdb, which the figure code needs.
if [ -n "${VIZ_ROLLOUTS:-}" ]; then
    uv run --project "$POINTACT_ENV" --no-sync \
        python experiments/13_robocasa365/render_rollouts.py \
        "${ckpt_dir}/results/checkpoint-${ckpt_step}${SAVE_SUFFIX:-}" || \
        echo "rollout rendering failed (eval results are unaffected)" >&2
fi

echo ${ckpt_dir}/checkpoint-${ckpt_step}

# Tolerate the server's kill signal so `set -e` doesn't mark the job FAILED after a clean run.
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
echo "Server exited"
