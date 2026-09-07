#!/bin/bash
# Apply the required RoboCasa patches to the submodule working copy. Idempotent.
#
# The submodule is pinned, so these live as patches rather than as commits on a fork. They must
# be re-applied after every `git submodule update` and in every new worktree -- a worktree gets
# its own submodule working copy. See README.md in this directory for what breaks without them.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUB="$HERE/../robocasa"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

[ -d "$SUB/.git" ] || [ -f "$SUB/.git" ] || {
    echo "no robocasa checkout at $SUB -- run: git submodule update --init envs/robocasa365/robocasa" >&2
    exit 1
}

applied=0 skipped=0
for p in "$HERE"/[0-9][0-9][0-9][0-9]-*.patch; do
    [ -e "$p" ] || continue
    name="$(basename "$p")"
    if git -C "$SUB" apply --reverse --check "$p" 2>/dev/null; then
        echo "already applied: $name"
        skipped=$((skipped + 1))
    elif git -C "$SUB" apply --check "$p" 2>/dev/null; then
        git -C "$SUB" apply "$p"
        echo "applied:         $name"
        applied=$((applied + 1))
    else
        # Neither direction applies cleanly: the pinned revision moved, or the file was edited
        # by hand. Refuse rather than half-apply -- a partly-patched sampler still crashes, but
        # now for a reason this file no longer explains.
        echo "FAILED:          $name does not apply to $(git -C "$SUB" describe --always --dirty)" >&2
        echo "  the submodule revision may have moved; re-derive the patch before continuing" >&2
        exit 1
    fi
done
echo "patches: $applied applied, $skipped already present"

[ "$CHECK_ONLY" = 1 ] || exit 0

echo
echo "=== verifying ==="
cd "$HERE/../../.."
for t in test_empty_category_guard.py test_split_guard.py; do
    echo "--- $t"
    uv run --project envs/robocasa365 python "envs/robocasa365/patches/$t"
done
