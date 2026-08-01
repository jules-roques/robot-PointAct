#!/bin/bash
# Stage 0/3 on Jean Zay: download the target/human demonstrations for one or more tasks.
#
#   sbatch --export=ALL,TASKS="PickPlaceCounterToStove TurnOnMicrowave" \
#          experiments/13_robocasa365/data_prep_download_jeanzay.slurm
#
# Runs on `prepost`, not a login node and not a compute node: compute nodes here have no
# internet at all, and a multi-hour multi-GB pull is exactly what login nodes kill. prepost has
# outbound access and a 20h limit.
#
# ~500 human demos per task. Budget roughly 60-70 GB of raw data per task before replay, going
# by OpenDrawer's footprint on this cluster.
#
#SBATCH --job-name=download
#SBATCH --account=rgx@cpu
#SBATCH --partition=prepost
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

set -euo pipefail

module purge

WORKTREE="$WORK/code/robot-PointAct/.claude/worktrees/pointcount-ablation"
cd "$WORKTREE"
export PYTHONPATH="$WORKTREE:${PYTHONPATH:-}"

# Point uv at the MAIN checkout's simulator env, never the worktree's: the worktree has
# no envs/robocasa365/.venv, and uv would silently build an empty one (thousands of
# inodes on a WORK filesystem already at 93%) that lacks robocasa entirely. PYTHONPATH
# above is what makes the worktree's own code shadow the installed copy. See docs/envs.md.
ROBOCASA_ENV="$WORK/code/robot-PointAct/envs/robocasa365"
export HF_HOME="$SCRATCH/.cache/huggingface"

TASKS="${TASKS:?set TASKS=\"Task1 Task2\" via --export}"
echo "host=$(hostname) tasks=$TASKS"

# Downloading needs the network, so the usual offline flags must NOT be set here.
uv run --project "$ROBOCASA_ENV" --no-sync \
    python data_prep/robocasa365_to_lerobot/download_demos.py \
        --tasks $TASKS \
        --split target \
        --source human

echo "=== downloaded ==="
source experiments/13_robocasa365/task_paths.sh
for task in $TASKS; do
    dir=$(task_input_dir "$task")
    echo "$task -> $dir"
    du -sh "$dir" 2>/dev/null || echo "  MISSING"
    ls "$dir/lerobot/meta/info.json" >/dev/null 2>&1 \
        && echo "  episodes: $(python -c "import json;print(json.load(open('$dir/lerobot/meta/info.json'))['total_episodes'])")" \
        || echo "  no lerobot/meta/info.json -- check the download"
done
