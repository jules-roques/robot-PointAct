#!/bin/bash
# Replay RoboCasa365 demonstrations in simulation to build the point-cloud cache.
# Stage 1 of data_prep/robocasa365_to_lerobot (run in the robocasa365 env, on a GPU node).
#
# Submit (full OpenDrawer run):
#   sbatch data_prep/robocasa365_to_lerobot/replay.slurm
# Submit (quick validation on a few episodes, short dev QoS):
#   sbatch --qos=qos_gpu-dev --time=00:30:00 --export=ALL,EPISODES="0 1 2" \
#          data_prep/robocasa365_to_lerobot/replay.slurm
#
# RoboCasa365 does not run correctly on H100, so this targets the V100 partition.
#
#SBATCH --job-name=rc365-replay
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

REPO=/lustre/fswork/projects/rech/rgx/unw24jl/code/robot-PointAct
BASE="$SCRATCH/datasets/robot_data/robocasa365/v1.0/target/atomic/OpenDrawer/20250816"

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
  CHUNK=$(( (TOTAL + N - 1) / N ))
  START=$(( T * CHUNK ))
  END=$(( START + CHUNK )); [ "$END" -gt "$TOTAL" ] && END=$TOTAL
  if [ "$START" -ge "$TOTAL" ]; then echo "chunk $T empty (START=$START >= TOTAL=$TOTAL)"; exit 0; fi
  EPISODES=$(seq "$START" $((END-1)) | tr '\n' ' ')
  echo "array task $T/$N -> episodes [$START,$END)"
fi

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} episodes=${EPISODES:-ALL}"
nvidia-smi -L || true

srun uv run --project envs/robocasa365 --no-sync \
  data_prep/robocasa365_to_lerobot/replay.py \
  --input-dir "$BASE/lerobot" \
  --cache-dir "$BASE/replay_cache" \
  --resume \
  ${EPISODES:+--episodes $EPISODES}
