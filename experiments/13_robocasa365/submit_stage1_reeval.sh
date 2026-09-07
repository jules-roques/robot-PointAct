#!/bin/bash
# Re-score existing checkpoints under one experiment root with a given seed set. No training.
#
# Written to settle Stage 1 (Concerto init) vs Stage 5 (Utonia init): the two grids differed in
# backbone init AND step count AND eval protocol, so the collapse of the oracle's margin
# (26pp -> 4.6pp) could not be attributed to any one of them. Re-scoring the surviving Stage 1
# checkpoints at 30K with Stage 5's exact seed set matches everything except the init.
#
# The seed set matters and is not arbitrary. The scene stream is seed-deterministic, so reusing
# Stage 5's 7 11 13 17 19 makes the two grids see identical kitchens -- as close to paired as
# this setup gets. Never re-run a seed an arm already has: it is not new information, and it
# collides on the pooling guard.
#
# Legacy result files are renamed, not deleted. `pool_eval_results.py` globs
# per_trial_seed*_n*.json and raises on a duplicate seed -- but only after the GPU time is
# spent, and pooling n=50 files with n=100 ones would silently mix protocols. The rename takes
# them out of the glob while keeping the data.
#
#   EXPRS=... experiments/13_robocasa365/submit_stage1_reeval.sh <run> [run ...]
#
# Jean Zay specifics (module load, qos) live here rather than in shared code, per CLAUDE.md.
set -euo pipefail

# A non-interactive `ssh host '...'` gets no module function -- /etc/profile.d is only sourced
# for login shells. z_modules.sh is the one that defines it; modules.sh is an IDRIS stub.
[ -r /etc/profile.d/z_modules.sh ] && source /etc/profile.d/z_modules.sh

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
EXPRS="${EXPRS:?set EXPRS to the experiment root holding the arms to re-score}"
CKPT_STEP="${CKPT_STEP:-30000}"
LEGACY_GLOB="${LEGACY_GLOB:-per_trial_seed*_n50.json}"

[ "$#" -gt 0 ] || { echo "usage: EXPRS=... $0 <run> [run ...]" >&2; exit 1; }
cd "$REPO"

for r in "$@"; do
    d="$EXPRS/$r/results/checkpoint-$CKPT_STEP"
    [ -d "$d" ] || { echo "no checkpoint results at $d" >&2; exit 1; }
    for f in "$d"/$LEGACY_GLOB; do
        [ -e "$f" ] || continue
        mv "$f" "$d/legacy_$(basename "$f" | sed 's/^per_trial_//')"
        echo "  set aside $(basename "$f")"
    done
done

echo
for RUN in "$@"; do
    ID=$(sbatch --parsable \
        --job-name="ev-reeval-${RUN%-n*}" \
        --export=ALL,RUN="$RUN",EXPRS_DIR="$EXPRS",CKPT_STEP="$CKPT_STEP",VIZ_ROLLOUTS=0 \
        experiments/13_robocasa365/eval_seeds_jeanzay.slurm)
    echo "$RUN  eval=$ID  (seed set from eval_seeds_jeanzay.slurm @ checkpoint-$CKPT_STEP)"
done
