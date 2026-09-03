#!/bin/bash
# Evaluate ONE task's whole grid inside ONE node-job, by packing many eval pairs side by side.
#
# Replaces the array-per-(checkpoint|seed) pattern of eval_grid_jeanzay.slurm and
# eval_seeds_jeanzay.slurm. Those submit one 1-GPU job per (checkpoint, seed) and each queues
# separately: stage 5's 5 ckpts x 5 seeds was 25 jobs of ~1 h, i.e. 25 independent queue waits
# for ~1 h of work each. Eval is simulator-bound, not GPU-bound, so a whole node runs 16 pairs
# concurrently at near-linear throughput (measured, see packing_probe_jeanzay.slurm):
#
#   K=16 pairs/node -> 12.4x (A100) / 13.3x (H100) of one pair, ~1600 trials/h either way,
#   with GPU memory at 11-23 GB of 80 and utilisation at 14-36%. The ceiling is ABOVE 16.
#
# Node throughput is the same on both tiers because the bottleneck is CPU/MuJoCo/EGL, so the
# tier is a queue-depth choice and gpu_p6 (H100) wins on that -- hence the defaults below.
#
#   sbatch --export=ALL,RUNS="od-uniform-n8192-s0 od-eef-n8192-s0" \
#       experiments/13_robocasa365/eval_task_jeanzay.slurm
#
#   A100 instead:  sbatch --account=rgx@a100 --partition=gpu_p5 --constraint=a100 \
#       --gres=gpu:8 --cpus-per-task=64 --qos=qos_gpu_a100-t3 --export=ALL,RUNS=... <this file>
#
# ONE JOB PER TASK, not one job for everything. All RUNS must train on the same task and the
# job refuses to start otherwise. Three reasons, and they are the whole design:
#   - Runtime is predictable only within a task. Wall-clock is dominated by episode LENGTH
#     (OpenDrawer ~1h00-1h12 per 100 trials, TurnOnMicrowave ~42 min), so a per-task job can
#     be sized against a per-task rate; a mixed job cannot.
#   - Blast radius. One node-job either gets its walltime or does not. Per task, a loss costs
#     one task's grid rather than the campaign's.
#   - The manifest stays readable, and the failure report below names pairs, not array indices.
#
# NOTHING IS CLUTTERED BY PACKING, because none of the outputs are keyed by job:
#   - Ports self-negotiate (eval_robocasa365.sh retries until an unused one is found).
#   - Results are content-addressed at <ckpt_dir>/results/checkpoint-<step>/, one file per
#     seed (per_trial_seed<S>_n<N>.json) -- exactly the glob summarize_stage_a.py and
#     log_eval_to_wandb.py already pool over. Different runs/steps/seeds cannot overlap.
#   - Each pair's stdout goes to its own log under $SCRATCH/logs/robocasa365/evaltask-<jobid>/,
#     so 16 concurrent pairs are 16 files rather than one interleaved stream.
#
# WHAT PACKING GIVES UP versus an array, and what is done about it here:
#   - Slurm's per-element exit codes and requeue of just the failed IDs. Replaced by the
#     STATUS report at the end, which judges each pair by whether its final JSON EXISTS rather
#     than by the exit code -- a pair can exit 0 having written nothing.
#   - All-or-nothing walltime. Mitigated two ways: a pair whose estimated duration does not
#     fit in the remaining walltime is NOT started (a killed run's partial dump is deliberately
#     named outside the pooling glob, so it contributes nothing -- starting it would waste the
#     slot), and every pair that already has its final JSON is skipped. Consequence: this job
#     is IDEMPOTENT. Resubmit the identical command to finish a grid that ran out of time.
#
# MolmoPoint arms are not supported and are rejected up front: the arm was dropped (it did not
# beat the eef-centred Gaussian) and it carries a third process on a second port, which this
# harness neither sizes walltime nor packing for.
#
# SMOKE-TESTED 2026-09-03, job 1710372 on gpu_p6 (4 H100, 96 cores): 4 pairs (2 arms x 2 seeds)
# x 2 trials on s5-od-*-n8192-s0 at checkpoint-30000, 4 ok / 0 failed in 2 min 33 s wall.
# Verified afterwards that every result landed in results/checkpoint-30000-smoke/, that the
# canonical checkpoint-30000/ files were untouched (still their August mtimes), that the pooling
# skips the -smoke directory while still reporting the real 371/500 and 358/500, and that the
# written records carry seed_per_episode=true. Do this before every new grid -- it is 3 minutes
# on the dev QoS against the 5 failed submissions and two burned 2 h jobs it would have caught.
#
#SBATCH --job-name=evaltask
#SBATCH --account=rgx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
# A whole gpu_p6 node: 4 H100 and 96 physical cores. The packing probe's first sweep asked for
# 32 cores and hit its own CPU ceiling at K=4 while the GPU sat at 42% -- ask for the node.
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-t3
# Sized for a grid, not a single pair: 4 arms x 5 steps x 5 seeds x 100 trials = 10000 trials,
# ~6 h at 1646 trials/h. Shrink to 02:00:00 --qos=qos_gpu_h100-dev (priority 80 vs t3's 50,
# starts in ~3 min) for anything that fits in ~3200 trials, and rely on idempotent resubmit.
#SBATCH --time=06:00:00
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

RUNS="${RUNS:?set RUNS=\"run-a run-b ...\" (all on the SAME task) via --export}"
EXPRS_DIR="${EXPRS_DIR:-$SCRATCH/PointAct_exprs/robocasa365/stage5}"
read -r -a STEPS <<< "${EVAL_STEPS:-5000 10000 15000 20000 25000 30000}"
read -r -a SEEDS <<< "${EVAL_SEEDS:-7 11 13 17 19}"
NUM_TRIALS="${NUM_TRIALS:-100}"
CONCURRENCY="${CONCURRENCY:-16}"
# Measured steady state at K=16 is 35-36 s/trial; 45 leaves margin for a longer task and is
# only used to decide whether a pair still FITS in the remaining walltime. Too low and a pair
# gets killed at trial 80 having written nothing poolable; too high and slots go unused.
SEC_PER_TRIAL="${SEC_PER_TRIAL:-45}"
EST_PAIR_S=$(( NUM_TRIALS * SEC_PER_TRIAL ))
# Bounds the failure that cost the packing probe two full jobs: the server dies on a config
# error and the client then blocks forever on a port nobody listens to.
PAIR_TIMEOUT="${PAIR_TIMEOUT:-$(( EST_PAIR_S * 2 ))}"

# SAVE_SUFFIX redirects results to <ckpt_dir>/results/checkpoint-<step><SUFFIX>/ instead of the
# canonical directory. This is THE way to smoke-test the harness against real checkpoints: both
# pooling consumers take the trailing "-"-separated token of the directory name and skip the
# directory unless it is all digits (log_eval_to_wandb.py's int() raises; summarize_stage_a.py's
# isdigit() is false), so `-smoke` output cannot reach a real success rate.
#
# Which makes a numeric suffix actively dangerous: SAVE_SUFFIX=-2 yields "checkpoint-30000-2",
# whose trailing token IS a digit, so it gets pooled -- as step 2. Refuse that outright.
SAVE_SUFFIX="${SAVE_SUFFIX:-}"
if [ -n "$SAVE_SUFFIX" ] && [ "${SAVE_SUFFIX##*-}" -eq "${SAVE_SUFFIX##*-}" ] 2>/dev/null; then
  echo "ERROR: SAVE_SUFFIX='$SAVE_SUFFIX' ends in a number, so results/checkpoint-<step>$SAVE_SUFFIX" >&2
  echo "  would be pooled as if it were checkpoint <the trailing number>. Use a suffix whose" >&2
  echo "  last '-'-separated token is not a digit, e.g. -smoke." >&2
  exit 1
fi
export SAVE_SUFFIX

# --args.verbose by default: it makes the client print one line per completed trial, which is
# the only way to see progress inside a pair's log during a multi-hour packed job, and the
# timestamps on those lines are what the packing probe measured throughput from. Override OPTS
# to change it; it costs a line per trial.
OPTS="${OPTS:- --args.verbose}"

LOGDIR="$SCRATCH/logs/robocasa365/evaltask-${SLURM_JOB_ID:-manual}"
mkdir -p "$LOGDIR"
MANIFEST="$LOGDIR/manifest.tsv"
STATUS="$LOGDIR/status.tsv"
: > "$MANIFEST"
: > "$STATUS"

NGPU=$(nvidia-smi -L | wc -l)
# -i 0 rather than `| head -1`: under `set -o pipefail` head closes the pipe, nvidia-smi takes
# SIGPIPE and the pipeline returns 141, which `set -e` turns into an instant job death.
GPUNAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)
echo "host=$(hostname) partition=${SLURM_JOB_PARTITION:-?} gpus=$NGPU ($GPUNAME) cpus=${SLURM_CPUS_PER_TASK:-?}"
echo "runs=$RUNS"
echo "steps=${STEPS[*]}  seeds=${SEEDS[*]}  trials=$NUM_TRIALS  concurrency=$CONCURRENCY"
[ -n "$SAVE_SUFFIX" ] && echo "save_suffix=$SAVE_SUFFIX  (results land beside, and OUT of, the real ones)"
echo "logs=$LOGDIR"
echo

# ---------------------------------------------------------------- manifest -------------------
# One line per (run, step, seed). Built before any pair starts so a misconfigured grid fails in
# seconds rather than at pair 40, and so the task-consistency check can see every run.
ENV_NAME=""
n_have=0 n_skip_done=0 n_skip_nockpt=0

for run in $RUNS; do
  ckpt_dir="$EXPRS_DIR/$run"
  data_config="$ckpt_dir/data_config.yaml"
  if [ ! -f "$data_config" ]; then
    echo "ERROR: no data_config.yaml under $ckpt_dir -- wrong EXPRS_DIR, or the run never trained." >&2
    exit 1
  fi

  if grep -qiE '^\s*molmo_sampling:\s*true' "$data_config"; then
    echo "ERROR: $run is a MolmoPoint arm. That arm was dropped (no gain over eef-centred" >&2
    echo "  Gaussian) and it needs a third process this harness does not size for." >&2
    exit 1
  fi

  # Task and the text-context cache come from the run's own archived data config, so an eval
  # can never silently disagree with the training it measures. `root` is repo-root-relative
  # (through the robot_data symlink), so absolute is just $REPO/<root>/<repo_id>/...
  read -r run_env text_ctx <<EOF
$($WORK/code/robot-PointAct/.venv/bin/python -c "
import yaml
ds = yaml.safe_load(open('$data_config'))['lerobot_datasets'][0]
cache = ds.get('text_context_file')
print(ds['repo_id'],
      f\"$REPO/{ds['root']}/{ds['repo_id']}/{cache}\" if cache else '')")
EOF

  # ONE TASK PER JOB. Mixing them is what makes walltime unpredictable, so refuse rather than
  # produce a job whose runtime nobody can size.
  if [ -z "$ENV_NAME" ]; then
    ENV_NAME="$run_env"
  elif [ "$run_env" != "$ENV_NAME" ]; then
    echo "ERROR: this job is one task only, but $run trains on $run_env and an earlier run" >&2
    echo "  trains on $ENV_NAME. Submit one job per task." >&2
    exit 1
  fi

  # data_config roots resolve through a `robot_data` symlink at the top of the checkout. That
  # symlink is gitignored, so it exists in the main checkout and is ABSENT in every fresh
  # worktree -- where it surfaces as a bare FileNotFoundError from torch.load, minutes into the
  # job. Check it here instead. See docs/clusters/jean-zay.md.
  if [ -n "$text_ctx" ] && [ ! -f "$text_ctx" ]; then
    echo "ERROR: text context cache not found:" >&2
    echo "  $text_ctx" >&2
    echo "If this is a fresh worktree it is missing the gitignored robot_data symlink:" >&2
    echo "  ln -s \$SCRATCH/datasets/robot_data $REPO/robot_data" >&2
    exit 1
  fi

  for step in "${STEPS[@]}"; do
    if [ ! -d "$ckpt_dir/checkpoint-$step" ]; then
      echo "  skip $run step=$step: checkpoint not written yet"
      n_skip_nockpt=$(( n_skip_nockpt + ${#SEEDS[@]} ))
      continue
    fi
    for seed in "${SEEDS[@]}"; do
      # Idempotency, and the reason a timed-out job can simply be resubmitted. The check globs
      # _n*.json rather than _n$NUM_TRIALS.json because episodes the simulator could not
      # present are skipped, so the final count can legitimately be below num_trials.
      if compgen -G "$ckpt_dir/results/checkpoint-${step}${SAVE_SUFFIX}/per_trial_seed${seed}_n*.json" > /dev/null; then
        n_skip_done=$(( n_skip_done + 1 ))
        continue
      fi
      printf '%s\t%s\t%s\t%s\t%s\n' "$run" "$step" "$seed" "$ckpt_dir" "$text_ctx" >> "$MANIFEST"
      n_have=$(( n_have + 1 ))
    done
  done
done

echo
echo "task=$ENV_NAME  pairs to run=$n_have  already done=$n_skip_done  no checkpoint=$n_skip_nockpt"
if [ "$n_have" -eq 0 ]; then
  echo "Nothing to do -- every requested pair already has its final JSON."
  exit 0
fi
echo "estimated: $n_have pairs x ~$(( EST_PAIR_S / 60 )) min, $CONCURRENCY at a time" \
     "=> ~$(( (n_have + CONCURRENCY - 1) / CONCURRENCY * EST_PAIR_S / 60 )) min"
echo

K="$CONCURRENCY"
[ "$K" -gt "$n_have" ] && K="$n_have"

# Rollout animations from ONE pair only: they cost buffering and render time, and the figure
# worth keeping is the one from the policy the headline number describes -- the last checkpoint
# of the first run, at the first seed.
VIZ_TAG="${VIZ_TAG:-$(awk -F'\t' -v s="${STEPS[${#STEPS[@]}-1]}" 'NR==1{print $1"_"s"_seed"$3}' "$MANIFEST")}"

# Torch would otherwise start ~96 OMP threads per process, i.e. 16x oversubscription at K=16.
# The probe measured near-linear scaling without this (the work is GPU/sim-bound), so it is
# insurance rather than a fix; set OMP_PER_PAIR=0 to go back to torch's default.
OMP_PER_PAIR="${OMP_PER_PAIR:-$(( ${SLURM_CPUS_PER_TASK:-8} / K ))}"
[ "$OMP_PER_PAIR" -lt 1 ] && OMP_PER_PAIR=1

# ---------------------------------------------------------------- workers --------------------
# Static round-robin over the manifest rather than a locked work queue: pairs within one task
# are near-homogeneous, so imbalance is small, and this needs no coordination between workers.
# A worker owns one GPU; server and client MUST share it, since the client renders through EGL
# on the device the policy runs on.
worker() {
  local w="$1" lineno=0 rc tag log left
  while IFS=$'\t' read -r run step seed ckpt_dir text_ctx; do
    lineno=$(( lineno + 1 ))
    [ $(( (lineno - 1) % K )) -eq "$w" ] || continue

    tag="${run}_${step}_seed${seed}"
    log="$LOGDIR/$tag.log"

    # Refuse to start a pair that cannot finish. A killed client's partial dump is named
    # outside the pooling glob on purpose, so an over-running pair contributes NOTHING -- the
    # slot is better left empty and the pair picked up by a resubmit.
    left=999999
    if [ -n "${SLURM_JOB_END_TIME:-}" ]; then
      left=$(( SLURM_JOB_END_TIME - $(date +%s) ))
    fi
    if [ "$left" -lt "$EST_PAIR_S" ]; then
      printf 'SKIP-WALLTIME\t%s\t%ss-left\n' "$tag" "$left" >> "$STATUS"
      echo "[w$w] stopping: ${left}s left, a pair needs ~${EST_PAIR_S}s. Resubmit to finish."
      break
    fi

    echo "[w$w] gpu=$(( w % NGPU )) start $tag"
    rc=0
    (
      export CUDA_VISIBLE_DEVICES=$(( w % NGPU ))
      export TEXT_CONTEXT_FILE="$text_ctx"
      export SEED="$seed"
      export NUM_TRIALS="$NUM_TRIALS"
      [ "$OMP_PER_PAIR" -gt 0 ] && export OMP_NUM_THREADS="$OMP_PER_PAIR"
      [ "$tag" = "$VIZ_TAG" ] && export VIZ_ROLLOUTS=1
      timeout "$PAIR_TIMEOUT" bash experiments/13_robocasa365/eval_robocasa365.sh \
        "$ENV_NAME" "$ckpt_dir" "$step" "rot6d" "${OPTS:-}"
    ) > "$log" 2>&1 || rc=$?

    # Judged by the ARTIFACT, not the exit code: a pair can exit 0 having written nothing
    # poolable, and that is precisely the failure an array's exit codes would have hidden.
    if compgen -G "$ckpt_dir/results/checkpoint-${step}${SAVE_SUFFIX}/per_trial_seed${seed}_n*.json" > /dev/null; then
      printf 'OK\t%s\trc=%s\n' "$tag" "$rc" >> "$STATUS"
      echo "[w$w] done  $tag (rc=$rc)"
    else
      printf 'MISSING\t%s\trc=%s\t%s\n' "$tag" "$rc" "$log" >> "$STATUS"
      echo "[w$w] FAILED $tag (rc=$rc) -- no final JSON; see $log"
    fi
  done < "$MANIFEST"
}

echo "launching $K workers over $NGPU GPUs (OMP_NUM_THREADS=$OMP_PER_PAIR per pair)"
echo
pids=()
for w in $(seq 0 $(( K - 1 ))); do
  worker "$w" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || true; done

# ---------------------------------------------------------------- report ---------------------
n_ok=$(grep -c '^OK' "$STATUS" 2>/dev/null || true)
n_missing=$(grep -c '^MISSING' "$STATUS" 2>/dev/null || true)
# NOT the count of SKIP-WALLTIME lines: a worker breaks out of its shard on the first pair that
# does not fit, so it logs one line and silently leaves the rest of its shard unattempted.
# Counting the lines reported "2 not started" for 4 unstarted pairs. Derive it from the
# manifest instead, which is the only number that adds up.
n_notstarted=$(( n_have - n_ok - n_missing ))

echo
echo "===================== $ENV_NAME: $n_ok ok / $n_missing failed / $n_notstarted not started ====="
sort "$STATUS" | column -t
echo
echo "logs: $LOGDIR"

if [ "${n_missing:-0}" -gt 0 ] || [ "${n_notstarted:-0}" -gt 0 ]; then
  echo
  echo "Resubmit the IDENTICAL command to finish: pairs whose final JSON exists are skipped."
  if [ "${n_missing:-0}" -gt 0 ]; then
    echo "Failed pairs re-run too, so read one of the MISSING logs above first."
  fi
fi
# Nonzero on real failures so a chained job does not treat a half-grid as complete. Pairs left
# unstarted for walltime are expected in a chain and do not fail the job -- chain with
# --dependency=afterany if failures should still be retried automatically.
[ "${n_missing:-0}" -eq 0 ]
