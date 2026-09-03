#!/usr/bin/env bash
# Archive training runs from $SCRATCH (30-day purge) to $STORE (no purge, tape-backed).
#
# Keeps only what is needed to *re-evaluate* a checkpoint: the weights and the
# config/tokenizer files that AutoModel/AutoProcessor.from_pretrained() reads.
# Dropped: optimizer.pt, scheduler.pt, rng_state_*.pth -- resume-only state that
# accounts for ~2/3 of every checkpoint's bytes and is useless for eval.
#
# One uncompressed tar per run (safetensors are incompressible, and $STORE caps
# inodes at 100k, so many small files are the thing to avoid).
#
#   scripts/archive_checkpoints.sh --dry-run
#   scripts/archive_checkpoints.sh
#   scripts/archive_checkpoints.sh --steps "50000" --runs od-eef-n4096-s0
#   scripts/archive_checkpoints.sh --with-optimizer --steps final
#
# On Jean Zay, run it on the `archive` partition rather than a login node:
#   sbatch -p archive -t 20:00:00 --wrap "scripts/archive_checkpoints.sh"

set -euo pipefail

SRC_ROOT="${SRC_ROOT:-$SCRATCH/PointAct_exprs/robocasa365}"
DEST_ROOT="${DEST_ROOT:-$STORE/PointAct/robocasa365}"

# Checkpoints worth keeping. "final" matches checkpoint-final-*; the eval grid is
# 10/20/30/40/50K (see robocasa365 experiment conventions).
STEPS="10000 20000 30000 40000 50000 final"
DRY_RUN=0
WITH_OPTIMIZER=0
RUN_FILTER=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)        DRY_RUN=1; shift ;;
        --with-optimizer) WITH_OPTIMIZER=1; shift ;;
        --steps)          STEPS="$2"; shift 2 ;;
        --runs)           shift; while [[ $# -gt 0 && "$1" != --* ]]; do RUN_FILTER+=("$1"); shift; done ;;
        --src)            SRC_ROOT="$2"; shift 2 ;;
        --dest)           DEST_ROOT="$2"; shift 2 ;;
        -h|--help)        sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

[[ -d "$SRC_ROOT" ]] || { echo "no such source root: $SRC_ROOT" >&2; exit 1; }

GIT_SHA="$(git -C "$(dirname "$0")/.." rev-parse HEAD 2>/dev/null || echo unknown)"

# Files inside a checkpoint dir that eval never reads.
EXCLUDES=(--exclude='optimizer.pt' --exclude='scheduler.pt' --exclude='rng_state_*.pth')
[[ $WITH_OPTIMIZER -eq 1 ]] && EXCLUDES=()

want_step() {  # $1 = checkpoint dir basename, e.g. checkpoint-50000 / checkpoint-final-50000
    local name="${1#checkpoint-}"
    for s in $STEPS; do
        [[ "$s" == "final" && "$name" == final-* ]] && return 0
        [[ "$name" == "$s" ]] && return 0
    done
    return 1
}

total_runs=0; total_bytes=0; skipped=0

# A run dir is any directory that directly contains at least one checkpoint-*.
while IFS= read -r run_dir; do
    run_name="$(basename "$run_dir")"
    rel_parent="$(dirname "${run_dir#$SRC_ROOT/}")"
    [[ "$rel_parent" == "." ]] && rel_parent=""

    if [[ ${#RUN_FILTER[@]} -gt 0 ]]; then
        match=0
        for f in "${RUN_FILTER[@]}"; do [[ "$run_name" == *"$f"* ]] && match=1; done
        [[ $match -eq 1 ]] || continue
    fi

    dest_dir="$DEST_ROOT${rel_parent:+/$rel_parent}"
    tar_path="$dest_dir/$run_name.tar"

    # Don't redo work, and respect the two runs already archived flat under DEST_ROOT.
    if [[ -f "$tar_path" || -f "$DEST_ROOT/$run_name.tar" ]]; then
        echo "SKIP  $run_name (already archived)"
        skipped=$((skipped + 1))
        continue
    fi

    members=(); n_ckpt=0
    for ck in "$run_dir"/checkpoint-*; do
        [[ -d "$ck" && -f "$ck/model.safetensors" ]] || continue
        if want_step "$(basename "$ck")"; then
            members+=("$run_name/$(basename "$ck")")
            n_ckpt=$((n_ckpt + 1))
        fi
    done
    # Provenance and the eval record: results/ holds the per-trial JSONs the paper
    # numbers come from, viz/ the sampling animations. Both are small; keep them.
    for extra in training_args.json results viz; do
        [[ -e "$run_dir/$extra" ]] && members+=("$run_name/$extra")
    done

    if [[ $n_ckpt -eq 0 ]]; then
        echo "SKIP  $run_name (no matching checkpoints)"
        skipped=$((skipped + 1))
        continue
    fi

    bytes=0
    for m in "${members[@]}"; do
        b=$(du -sb --exclude=optimizer.pt --exclude=scheduler.pt --exclude='rng_state_*.pth' \
              "$SRC_ROOT${rel_parent:+/$rel_parent}/$m" 2>/dev/null | cut -f1 || echo 0)
        bytes=$((bytes + b))
    done
    total_bytes=$((total_bytes + bytes))
    total_runs=$((total_runs + 1))

    printf 'PACK  %-40s %2d ckpt  %6.1f GiB -> %s\n' \
        "$run_name" "$n_ckpt" "$(echo "$bytes/1073741824" | bc -l)" "$tar_path"

    [[ $DRY_RUN -eq 1 ]] && continue

    mkdir -p "$dest_dir"
    tmp_tar="$tar_path.partial"
    tar -cf "$tmp_tar" -C "$(dirname "$run_dir")" "${EXCLUDES[@]}" "${members[@]}"

    # Verify the archive is readable before it becomes the only copy.
    n_entries="$(tar tf "$tmp_tar" | wc -l)"
    [[ "$n_entries" -gt 0 ]] || { echo "ERROR: empty archive for $run_name" >&2; rm -f "$tmp_tar"; exit 1; }
    mv "$tmp_tar" "$tar_path"

    {
        echo "run:        $run_name"
        echo "source:     $run_dir"
        echo "archived:   $(date -Iseconds)"
        echo "git_commit: $GIT_SHA"
        echo "steps:      $STEPS"
        echo "optimizer:  $([[ $WITH_OPTIMIZER -eq 1 ]] && echo included || echo dropped)"
        echo "entries:    $n_entries"
        echo "bytes:      $(stat -c%s "$tar_path")"
        echo "sha256:     $(sha256sum "$tar_path" | cut -d' ' -f1)"
    } > "$tar_path.meta"

    echo "  ok  $n_entries entries, $(du -h "$tar_path" | cut -f1)"
# A run dir directly contains checkpoint-*/model.safetensors. The model.safetensors
# test is what keeps eval-output dirs (<run>/results/checkpoint-*) from looking like runs.
done < <(find "$SRC_ROOT" -mindepth 1 -type d -name 'checkpoint-*' \
             -exec test -f '{}/model.safetensors' \; -printf '%h\n' | sort -u)

printf '\n%d run(s) to archive, %d skipped, %.1f GiB total\n' \
    "$total_runs" "$skipped" "$(echo "$total_bytes/1073741824" | bc -l)"
[[ $DRY_RUN -eq 1 ]] && echo "(dry run -- nothing written)"
exit 0
