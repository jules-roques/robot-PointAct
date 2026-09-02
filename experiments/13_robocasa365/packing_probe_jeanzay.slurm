#!/bin/bash
# How many concurrent eval pairs fit on ONE GPU?
#
# Eval is simulator-bound: the policy server is queried once per replan_steps (8) while MuJoCo
# physics and EGL rendering run every step, so a single eval pair leaves the GPU idle much of
# the time. This probe measures how much of that idle time is recoverable by running K pairs
# side by side, and where the packing factor tops out (GPU memory, or contention).
#
# Submit (from a login shell, so `module` and $DSDIR exist -- see docs/clusters/jean-zay.md):
#   A100:  sbatch experiments/13_robocasa365/packing_probe_jeanzay.slurm
#   H100:  sbatch --account=rgx@h100 --partition=gpu_p6 --constraint=h100 \
#            --qos=qos_gpu_h100-dev experiments/13_robocasa365/packing_probe_jeanzay.slurm
#
# The headline metric is STEADY-STATE seconds per trial, taken from timestamps on the client's
# own per-trial lines, not from end-to-end wall. Model load is a fixed ~1-2 min that a short
# run would otherwise smear across the throughput number, forcing a long run to average it
# out. Measuring the inter-trial interval directly makes the probe short AND more accurate:
# 8 trials give 7 intervals, which is plenty for a median.
#
# Results go under $SCRATCH/tmp/packprobe, NOT the run's own results/: log_eval_to_wandb.py
# globs results/checkpoint-*, so a probe writing there would be pooled into real success rates.
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

# Unload first: --export=ALL carries the submitting shell's arch module, and loading a second
# one fails with "cannot be loaded due to a conflict" rather than replacing it. Not `purge` --
# ffmpeg's libav* must stay resolvable for torchcodec.
module unload arch 2>/dev/null || true
case "${SLURM_JOB_PARTITION:-}" in
  gpu_p6) module load arch/h100 ;;
  gpu_p5) module load arch/a100 ;;
esac
module load ffmpeg/6.1.1

REPO="${SLURM_SUBMIT_DIR:-$WORK/code/robot-PointAct}"
cd "$REPO"

RUN="${RUN:-od-eef-n4096-s0}"
CKPT_STEP="${CKPT_STEP:-50000}"
TRIALS="${TRIALS:-8}"
CONCURRENCY="${CONCURRENCY:-1 2 3 4}"
# Bounds the class of failure that cost this probe two 2 h jobs: the server dies on a config
# error, the client then blocks forever on a port nobody is listening to, and `wait` hangs
# until walltime. A pair that exceeds this is killed and the phase fails loudly.
PAIR_TIMEOUT="${PAIR_TIMEOUT:-1800}"
SRC="${EXPRS_DIR:-$SCRATCH/PointAct_exprs/robocasa365/ablation}/$RUN"

# Mirror every entry of the real run except results/, which must be a REAL directory here so
# probe output cannot land in the run's own results. data_config.yaml and training_args.json
# have to be present: the eval script reads them to infer the point-sampling mode.
BENCH="$SCRATCH/tmp/packprobe/$SLURM_JOB_ID/$RUN"
mkdir -p "$BENCH/results"
for entry in "$SRC"/*; do
  name=$(basename "$entry")
  [ "$name" = "results" ] && continue
  ln -sfn "$entry" "$BENCH/$name"
done
test -e "$BENCH/checkpoint-$CKPT_STEP" || { echo "no checkpoint-$CKPT_STEP in $SRC"; exit 1; }

# Task and text-context cache come from the run's own archived data config, exactly as
# eval_grid_jeanzay.slurm does it. Without TEXT_CONTEXT_FILE a text_cache-trained checkpoint
# aborts at server start -- which is how this probe first burned two jobs.
DATA_CONFIG="$BENCH/data_config.yaml"
read -r ENV_NAME TEXT_CTX <<EOF
$($WORK/code/robot-PointAct/.venv/bin/python -c "
import yaml
ds = yaml.safe_load(open('$DATA_CONFIG'))['lerobot_datasets'][0]
cache = ds.get('text_context_file')
print(ds['repo_id'],
      f\"$REPO/{ds['root']}/{ds['repo_id']}/{cache}\" if cache else '')")
EOF
export TEXT_CONTEXT_FILE="$TEXT_CTX"

# data_config roots are repo-relative ("robot_data/robocasa365/...") and resolve through a
# `robot_data` symlink at the top of the checkout. That symlink is gitignored, so it exists in
# the main checkout and is ABSENT in every fresh worktree -- which fails here as a
# FileNotFoundError from torch.load, several minutes into the job. Check it up front instead.
if [ -n "${TEXT_CONTEXT_FILE:-}" ] && [ ! -f "$TEXT_CONTEXT_FILE" ]; then
  echo "ERROR: text context cache not found:" >&2
  echo "  $TEXT_CONTEXT_FILE" >&2
  echo "This checkout is missing the gitignored robot_data symlink. Create it with:" >&2
  echo "  ln -s \$SCRATCH/datasets/robot_data $REPO/robot_data" >&2
  exit 1
fi

echo "host=$(hostname) partition=${SLURM_JOB_PARTITION:-?} run=$RUN ckpt=$CKPT_STEP env=$ENV_NAME trials=$TRIALS"
echo "text_context_file=${TEXT_CONTEXT_FILE:-<none>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

SUMMARY="$BENCH/summary.tsv"
printf "K\tsec_per_trial\ttrials_per_h\tspeedup_vs_K1\twall_s\tpeak_mem_MiB\tmean_util_pct\n" > "$SUMMARY"
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
    # sequence, so differences across K are contention and nothing else. Only true because
    # scenes are a pure function of (seed, episode_idx) -- see seed_per_episode.
    # awk stamps each line with an epoch second; --args.verbose is what emits one line per
    # completed trial, and those stamps are the actual measurement.
    (
      SEED=7 NUM_TRIALS="$TRIALS" SAVE_SUFFIX="-k${K}-i${i}" \
        timeout "$PAIR_TIMEOUT" bash experiments/13_robocasa365/eval_robocasa365.sh \
          "$ENV_NAME" "$BENCH" "$CKPT_STEP" rot6d " --args.verbose" 2>&1 \
        | awk '{print systime()"\t"$0; fflush()}' > "$BENCH/k${K}_i${i}.log"
    ) &
    pids+=($!)
  done

  ok=0
  for p in "${pids[@]}"; do
    if wait "$p"; then ok=$((ok+1)); else echo "  [warn] one pair exited non-zero"; fi
  done
  t1=$(date +%s)
  kill "$SMI_PID" 2>/dev/null || true

  if [ "$ok" -eq 0 ]; then
    echo "  [fatal] every pair failed at K=$K; tail of the first pair log:" >&2
    tail -25 "$BENCH/k${K}_i1.log" >&2 || true
    exit 1
  fi

  # Steady-state: median gap between consecutive completed trials, pooled over the K
  # instances. Excludes model load by construction -- only gaps between completions are used.
  spt=$(cat "$BENCH"/k${K}_i*.log 2>/dev/null | awk -F'\t' '
      $2 ~ /^\[[0-9]+\] success=/ { if (p) print $1-p; p=$1 }' | sort -n | \
      awk '{a[NR]=$1} END{ if(NR==0){print 0; exit} print (NR%2)? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2 }')

  wall=$((t1 - t0))
  tph=$(awk -v k="$K" -v s="$spt" 'BEGIN{printf "%.1f", (s>0)? k*3600.0/s : 0}')
  [ -z "$BASE" ] && BASE="$tph"
  spd=$(awk -v a="$tph" -v b="$BASE" 'BEGIN{printf "%.2f", (b>0)? a/b : 0}')
  mem=$(awk -F, 'NF>=2{gsub(/ /,"",$1); if($1+0>m) m=$1+0} END{print m+0}' "$SAMPLE" 2>/dev/null)
  utl=$(awk -F, 'NF>=2{gsub(/ /,"",$2); s+=$2; n++} END{printf "%.0f", (n>0)? s/n : 0}' "$SAMPLE" 2>/dev/null)

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$K" "$spt" "$tph" "$spd" "$wall" "$mem" "$utl" >> "$SUMMARY"
  echo "  K=$K ok=$ok/$K  ${spt}s/trial/instance  ${tph} trials/h  ${spd}x  wall=${wall}s  mem=${mem}MiB  util=${utl}%"
  echo
done

echo "================= SUMMARY ($(hostname), ${SLURM_JOB_PARTITION:-?}) ================="
column -t "$SUMMARY"
echo
echo "artifacts: $BENCH"
