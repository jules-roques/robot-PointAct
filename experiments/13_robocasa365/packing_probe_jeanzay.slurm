#!/bin/bash
# How many concurrent eval pairs fit on ONE GPU?
#
# Eval is simulator-bound: the policy server is queried once per replan_steps (8) while MuJoCo
# physics and EGL rendering run every step, so a single eval pair leaves the GPU idle much of
# the time. This probe measures how much of that idle time is actually recoverable by running
# K pairs side by side, and where the packing factor tops out (GPU memory, or contention).
#
# Submit (from a login shell, so `module` and $DSDIR exist -- see docs/clusters/jean-zay.md):
#   A100:  sbatch experiments/13_robocasa365/packing_probe_jeanzay.slurm
#   H100:  module load arch/h100 && sbatch --account=rgx@h100 --partition=gpu_p6 \
#            --constraint=h100 --qos=qos_gpu_h100-dev \
#            experiments/13_robocasa365/packing_probe_jeanzay.slurm
#
# Results are written under $SCRATCH/tmp/packprobe, NOT under the run's own results/ dir:
# log_eval_to_wandb.py globs results/checkpoint-*, so a probe writing there would be pooled
# into the real success rates.
#
#SBATCH --job-name=packprobe
#SBATCH --account=rgx@a100
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
# 32 = 4 concurrent pairs x the 8 cpus a single eval job asks for. Both node types have room
# (A100 64 physical cores, H100 96).
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%j.out

set -euo pipefail

# arch module follows the partition, so one file serves both GPU tiers.
case "${SLURM_JOB_PARTITION:-}" in
  gpu_p6) module load arch/h100 ;;
  gpu_p5) module load arch/a100 ;;
esac
module load ffmpeg/6.1.1

REPO="${SLURM_SUBMIT_DIR:-$WORK/code/robot-PointAct}"
cd "$REPO"

RUN="${RUN:-od-eef-n4096-s0}"
CKPT_STEP="${CKPT_STEP:-50000}"
ENV_NAME="${ENV_NAME:-OpenDrawer}"
TRIALS="${TRIALS:-12}"
CONCURRENCY="${CONCURRENCY:-1 2 3 4}"
SRC="${EXPRS_DIR:-$SCRATCH/PointAct_exprs/robocasa365/ablation}/$RUN"

# Throwaway ckpt_dir: the eval script derives save_dir from it, and we must not write into the
# real run's results/. Symlink rather than copy -- checkpoints are large.
BENCH="$SCRATCH/tmp/packprobe/$SLURM_JOB_ID/$RUN"
mkdir -p "$BENCH"
ln -sfn "$SRC/checkpoint-$CKPT_STEP" "$BENCH/checkpoint-$CKPT_STEP"

echo "host=$(hostname) partition=${SLURM_JOB_PARTITION:-?} run=$RUN ckpt=$CKPT_STEP trials=$TRIALS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

SUMMARY="$BENCH/summary.tsv"
printf "K\twall_s\ttrials_done\ttrials_per_h\tspeedup_vs_K1\tpeak_mem_MiB\tmean_util_pct\n" > "$SUMMARY"
BASE=""

for K in $CONCURRENCY; do
  echo "=============== K=$K concurrent eval pairs ==============="
  SAMPLE="$BENCH/smi_k$K.csv"
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -l 5 > "$SAMPLE" 2>/dev/null &
  SMI_PID=$!

  t0=$(date +%s)
  pids=()
  for i in $(seq 1 "$K"); do
    # Same seed for every instance on purpose: each pair then runs the identical scene
    # sequence, so wall-clock differences across K are contention and nothing else. This is
    # only true because scenes are now a pure function of (seed, episode_idx) -- see
    # run_robocasa365_client.py, seed_per_episode.
    SEED=7 NUM_TRIALS="$TRIALS" SAVE_SUFFIX="-k${K}-i${i}" \
      bash experiments/13_robocasa365/eval_robocasa365.sh \
        "$ENV_NAME" "$BENCH" "$CKPT_STEP" rot6d "" \
        > "$BENCH/k${K}_i${i}.log" 2>&1 &
    pids+=($!)
  done

  ok=0
  for p in "${pids[@]}"; do
    if wait "$p"; then ok=$((ok+1)); else echo "  [warn] one pair exited non-zero"; fi
  done
  t1=$(date +%s)
  kill "$SMI_PID" 2>/dev/null || true

  wall=$((t1 - t0))
  done_trials=$((ok * TRIALS))
  tph=$(awk -v d="$done_trials" -v w="$wall" 'BEGIN{printf "%.1f", (w>0)? d*3600.0/w : 0}')
  [ -z "$BASE" ] && BASE="$tph"
  spd=$(awk -v a="$tph" -v b="$BASE" 'BEGIN{printf "%.2f", (b>0)? a/b : 0}')
  mem=$(awk -F, 'NF>=2{gsub(/ /,"",$1); if($1+0>m) m=$1+0} END{print m+0}' "$SAMPLE" 2>/dev/null)
  utl=$(awk -F, 'NF>=2{gsub(/ /,"",$2); s+=$2; n++} END{printf "%.0f", (n>0)? s/n : 0}' "$SAMPLE" 2>/dev/null)

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$K" "$wall" "$done_trials" "$tph" "$spd" "$mem" "$utl" >> "$SUMMARY"
  echo "  K=$K  wall=${wall}s  ok_pairs=$ok/$K  trials/h=$tph  speedup=${spd}x  peak_mem=${mem}MiB  util=${utl}%"
  echo
done

echo "================= SUMMARY ($(hostname), ${SLURM_JOB_PARTITION:-?}) ================="
column -t "$SUMMARY"
echo
echo "artifacts: $BENCH"
