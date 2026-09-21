#!/bin/bash
# Replay RoboCasa365 demonstrations in simulation to build the point-cloud cache.
# Stage 1 of data_prep/robocasa365_to_lerobot (run in the robocasa365 env, on a GPU node).
#
# Submit (full OpenDrawer run):
#   sbatch data_prep/robocasa365_to_lerobot/replay.slurm
# Submit (another task — TASK and DATE select the source dump):
#   sbatch --export=ALL,TASK=CloseBlenderLid,DATE=20250816 \
#          data_prep/robocasa365_to_lerobot/replay.slurm
# Submit (quick validation on a few episodes, short dev QoS):
#   sbatch --qos=qos_gpu-dev --time=00:30:00 --export=ALL,EPISODES="0 1 2" \
#          data_prep/robocasa365_to_lerobot/replay.slurm
# Submit (sharded — 8 parallel chunks; per-episode caches are disjoint so this is safe):
#   sbatch --array=0-7 --export=ALL,TASK=CloseBlenderLid \
#          data_prep/robocasa365_to_lerobot/replay.slurm
#
# Targets the V100 partition because this stage only renders (no model inference), so the
# cheapest available GPU is the right one.
#
#SBATCH --job-name=replay
#SBATCH --account=rgx@v100
#SBATCH --partition=gpu_p13
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365_replay/%x_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365_replay/%x_%j.out

set -euo pipefail

module purge
module load ffmpeg/6.1.1   # libav* for torchcodec/av pulled in via lerobot

# Headless GPU rendering for MuJoCo (robosuite would auto-pick this; set it explicitly).
export MUJOCO_GL=egl

# Compute nodes have no network: keep uv, HF and transformers fully offline.
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_OFFLINE=1

# REPO is the code that RUNS; SIM_ENV is the checkout that owns envs/robocasa365/.venv.
# They are separate because a git worktree has the first and not the second: its
# envs/robocasa365/ has no .venv, and a bare `uv run --project` there would silently build an
# empty one (docs/envs.md). So point --project at the main checkout and let PYTHONPATH put the
# worktree's code in front -- the same split train_jeanzay.slurm makes with POINTACT_ENV.
#
# This matters for more than tidiness: the branch's replay.py is not the main checkout's. It
# additionally caches per-point labels and voxel keys, which stage 5's oracle arms need, while
# computing xyz/rgb identically. Running main's older copy would rebuild the point clouds
# correctly and quietly omit everything else.
REPO="${REPO:-/lustre/fswork/projects/rech/rgx/unw24jl/code/robot-PointAct}"
SIM_ENV="${SIM_ENV:-/lustre/fswork/projects/rech/rgx/unw24jl/code/robot-PointAct}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# TASK/DATE are overridable because the grid is no longer one task: stage 7 runs three, and
# the 2026-09-21 $SCRATCH purge means all of them may have to be re-prepped at once. DATE is
# the source dump's directory and is NOT the same for every task -- check what
# download_demos.py actually laid down rather than assuming OpenDrawer's.
TASK="${TASK:-OpenDrawer}"
DATE="${DATE:-20250816}"
BASE="$SCRATCH/datasets/robot_data/robocasa365/v1.0/target/atomic/$TASK/$DATE"

if [ ! -d "$BASE/lerobot" ]; then
    echo "ERROR: no source demos at $BASE/lerobot" >&2
    echo "  Download them first (login node, needs a gated-dataset HF login):" >&2
    echo "    uv run --project envs/robocasa365 \\" >&2
    echo "      data_prep/robocasa365_to_lerobot/download_demos.py --tasks $TASK" >&2
    echo "  Existing dumps for this task:" >&2
    ls -1 "$SCRATCH/datasets/robot_data/robocasa365/v1.0/target/atomic/$TASK" 2>/dev/null >&2 \
        || echo "    (none)" >&2
    exit 1
fi

cd "$REPO"

# Episode selection precedence:
#   1. EPISODES env (explicit space-separated list) — overrides everything.
#   2. Job array — each task replays a contiguous chunk (SLURM_ARRAY_TASK_ID of _COUNT).
#      Safe in parallel: per-episode caches are disjoint, replay_summary.jsonl appends are
#      atomic, and cache_meta.json is validated (not rewritten) under --resume.
#   3. Neither — the whole dataset in one job.
if [ -z "${EPISODES:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  TOTAL=$(python3 -c "import json;print(json.load(open('$BASE/lerobot/meta/info.json'))['total_episodes'])")
  N=${SLURM_ARRAY_TASK_COUNT}
  T=${SLURM_ARRAY_TASK_ID}

  # SLURM_ARRAY_TASK_COUNT is how many tasks THIS submission has, not the index range. So
  # resubmitting a failed subset as `--array=4-7` gives N=4 with T=4..7, and every chunk
  # starts past the end of the dataset -- each task prints "chunk empty" and exits 0. Four
  # jobs COMPLETED in four seconds having replayed nothing, which is the most expensive kind
  # of success. Refuse it instead: to redo part of an array, resubmit the WHOLE range and let
  # --resume skip what is cached (it is a per-episode file test, so it costs seconds).
  if [ "$T" -ge "$N" ]; then
    echo "ERROR: array index $T >= array size $N. This is a partial-array resubmission" >&2
    echo "  (e.g. --array=4-7), and the chunk maths needs the original full range to mean" >&2
    echo "  anything. Resubmit as --array=0-$(( N > T ? N-1 : T ))  (the full range you" >&2
    echo "  first used); --resume makes the already-finished shards cheap." >&2
    exit 1
  fi

  CHUNK=$(( (TOTAL + N - 1) / N ))
  START=$(( T * CHUNK ))
  END=$(( START + CHUNK )); [ "$END" -gt "$TOTAL" ] && END=$TOTAL
  if [ "$START" -ge "$TOTAL" ]; then echo "chunk $T empty (START=$START >= TOTAL=$TOTAL)"; exit 0; fi
  EPISODES=$(seq "$START" $((END-1)) | tr '\n' ' ')
  echo "array task $T/$N -> episodes [$START,$END)"
fi

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} episodes=${EPISODES:-ALL}"
nvidia-smi -L || true

srun uv run --project "$SIM_ENV/envs/robocasa365" --no-sync \
  data_prep/robocasa365_to_lerobot/replay.py \
  --input-dir "$BASE/lerobot" \
  --cache-dir "$BASE/replay_cache" \
  --resume \
  ${EPISODES:+--episodes $EPISODES}
