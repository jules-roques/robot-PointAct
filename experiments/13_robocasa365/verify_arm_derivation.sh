#!/bin/bash
# Pre-flight: replicate eval_robocasa365.sh's arm-identity derivation against a run's archived
# data_config.yaml, offline, before any GPU time is spent.
#
# Two separate failures in this campaign were arms silently scored as a *different* arm -- the
# molmo arms evaluated as uniform being the expensive one, since every number it produced had
# to be thrown away. The eval derives which sampler to use by grepping the archived
# data_config.yaml; this runs that same derivation and prints what it concluded, so a
# mislabelled arm is caught in a second rather than after a node-day.
#
# Keep the derivation below in sync with eval_robocasa365.sh. If they drift, this check is
# worse than none -- it would confirm an identity the eval does not actually use.
#
#   experiments/13_robocasa365/verify_arm_derivation.sh <exprs-dir> [run ...]
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

EXPRS="${1:?usage: verify_arm_derivation.sh <exprs-dir> [run ...]}"
shift
RUNS=("$@")
if [ "${#RUNS[@]}" -eq 0 ]; then
    # Every arm under the root that has an archived config.
    mapfile -t RUNS < <(cd "$EXPRS" && for d in */; do
        [ -f "$d/data_config.yaml" ] && basename "$d"
    done)
fi

for r in "${RUNS[@]}"; do
    CFG="$EXPRS/$r/data_config.yaml"
    [ -f "$CFG" ] || { printf "%-30s NO data_config.yaml\n" "$r"; continue; }

    ORACLE_GT=$(grep -oiE '^\s*oracle_gt:\s*[a-z]+' "$CFG" | head -1 | awk '{print $2}' || true)
    ORACLE_GT="${ORACLE_GT:-labels}"
    MVS=$(grep -oiE '^\s*molmo_view_select:\s*[a-z_]+' "$CFG" | head -1 | awk '{print $2}' || true)
    MVS="${MVS:-per_view}"
    FALLBACK=uniform

    if grep -qiE '^\s*oracle_sampling:\s*true' "$CFG"; then
        PS=anchor
    elif grep -qiE '^\s*eef_sampling:\s*true' "$CFG"; then
        PS=eef
    elif grep -qiE '^\s*molmo_sampling:\s*true' "$CFG"; then
        PS=anchor
    else
        # An unrecognised *_sampling flag must not fall through to "uniform": that is exactly
        # how an arm gets scored as a different arm.
        UNKNOWN=$(grep -oiE '^\s*[a-z0-9_]+_sampling:\s*true' "$CFG" \
                  | grep -oiE '[a-z0-9_]+_sampling' \
                  | grep -viE '^(oracle|eef|molmo)_sampling$' | sort -u | tr '\n' ' ' || true)
        [ -n "$UNKNOWN" ] && { echo "$r: UNKNOWN ARM $UNKNOWN"; continue; }
        PS=uniform
    fi

    printf "%-30s point_sampling=%-8s fallback=%-8s oracle_gt=%-7s view_select=%s\n" \
        "$r" "$PS" "$FALLBACK" "$ORACLE_GT" "$MVS"
done

echo
echo "=== does the eval carry trained anchor labels through? ==="
cd "$REPO_ROOT"
grep -n 'anchor_label\|oracle_anchor_labels\|ANCHOR_LABEL' \
    experiments/13_robocasa365/eval_robocasa365.sh \
    experiments/13_robocasa365/run_robocasa365_client.py | head -12
