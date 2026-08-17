#!/bin/bash
# Stages 2-3/3 on Jean Zay: pack the replay cache into a LeRobot dataset + points_3views LMDB,
# then compute the state/action normalization stats training needs. CPU-only.
#
#   sbatch --export=ALL,TASK=TurnOnMicrowave experiments/13_robocasa365/data_prep_convert_jeanzay.slurm
#
#SBATCH --job-name=convert
#SBATCH --account=rgx@cpu
#SBATCH --partition=cpu_p1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

set -euo pipefail

module purge
module load ffmpeg/6.1.1

# The checkout to run from is wherever this was submitted, exactly as the roi_sampling jobs
# already do it. A hardcoded worktree path outlives the worktree: this named
# .claude/worktrees/pointcount-ablation, which no longer exists on this cluster.
WORKTREE="${SLURM_SUBMIT_DIR:-$WORK/code/robot-PointAct}"
cd "$WORKTREE"
export PYTHONPATH="$WORKTREE:${PYTHONPATH:-}"

export SVT_LOG=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=TRUE
export HDF5_USE_FILE_LOCKING=FALSE
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export UV_OFFLINE=1

source experiments/13_robocasa365/task_paths.sh
TASK="${TASK:?set TASK=<task name> via --export}"
INPUT_DIR="${INPUT_DIR:-$(task_input_dir "$TASK")}"
OUTPUT_DIR="${OUTPUT_DIR:-$(task_output_dir "$TASK")}"

echo "host=$(hostname) task=$TASK"
echo "  in : $INPUT_DIR"
echo "  out: $OUTPUT_DIR"

uv run --project "$WORK/code/robot-PointAct" --no-sync \
    python data_prep/robocasa365_to_lerobot/convert.py \
        --cache-dir "$INPUT_DIR/replay_cache" \
        --output-dir "$OUTPUT_DIR" \
        --repo-id "$TASK" \
        --image-writer-threads 12

# Stage 3/3. Both flags are mandatory here, not stylistic:
#   --point_cloud_dir  position stats must be computed in the point-cloud frame;
#   --replace_zero_std these tasks are fixed-base, so the base-pose dims have zero variance and
#                      would divide by zero in the state normalization -> NaN on the first step.
uv run --project "$WORK/code/robot-PointAct" --no-sync \
    python data_prep/prepare_robot_state_action_stats.py \
        --dataset_dirs "$OUTPUT_DIR" \
        --output_file "$OUTPUT_DIR/robot_state_action_stats/rot6d.json" \
        --point_cloud_dir points_3views \
        --state_xyz_slice 0 3 --action_xyz_slice 0 3 \
        --state_rotation_slice 3 7 --action_rotation_slice 3 7 \
        --rotation_type quat --target_rotation_type rot6d \
        --replace_zero_std

# Converted episode -> source episode. Every stage that joins a source-indexed dump to a
# converted-indexed cache needs this: the geom oracle's target_positions lookup, the MolmoPoint
# depth lift, and eval_molmo_accuracy. Conversion drops episodes whose replay failed and
# renumbers the rest -- OpenDrawer's 514 became 496 -- so assuming identity reads a different
# episode's answer and looks entirely plausible. It had no wrapper and was a manual step, which
# is exactly how a new task ends up without one; it belongs here, where both directories are
# already in hand. Verifies itself against replay_length and fails the job if the map is wrong.
uv run --project "$WORK/code/robot-PointAct" --no-sync \
    python -m data_prep.robocasa365_to_lerobot.episode_index_map \
        --source-dir "$INPUT_DIR/lerobot" \
        --dataset-dir "$OUTPUT_DIR"

# Frame count feeds the steps-per-epoch check: the grid fixes 50K gradient steps, which is only
# comparable across tasks while their frames-per-epoch stay within ~10% of each other.
python - <<EOF
import json
info = json.load(open("$OUTPUT_DIR/meta/info.json"))
frames, eps = info["total_frames"], info["total_episodes"]
steps_per_epoch = frames / 128
print(f"=== $TASK: {eps} episodes, {frames} frames "
      f"({frames/eps:.0f}/ep) -> {steps_per_epoch:.0f} steps/epoch @ eff. batch 128; "
      f"50K steps = {50000/steps_per_epoch:.1f} epochs ===")
EOF
