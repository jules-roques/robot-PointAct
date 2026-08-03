#!/bin/bash
# Probe: how many points does a finer voxel grid actually put near the end effector?
#
#   sbatch experiments/13_robocasa365/voxel_probe_jeanzay.slurm
#
# Motivation. At the 1 cm grid the cache is built on, a frame's cloud holds ~21k points but
# only ~142 of them lie within 8 cm of the gripper (~9 within 4 cm). The eef-density draw at
# 4096 already selects ~95% of those 142, and at 8192 it selects all of them -- so neither a
# bigger point budget nor a sharper Gaussian can concentrate any further. The binding
# constraint is the voxel grid, not the sampler.
#
# What this does NOT settle by itself: whether a finer grid yields more points or merely
# stops merging points the render never resolved. The left/right cameras are 256^2 over a
# ~1.5 m scene (~6 mm pixel pitch), so from those two, 1 cm is close to the true sampling
# rate. The wrist camera sits ~20 cm from the gripper, so in the grasp region its pitch is
# ~1 mm and the merge is throwing detail away. This job measures which effect wins.
#
# Five episodes at two voxel sizes, into throwaway cache dirs. Nothing downstream reads them;
# voxel_probe_report.py counts points near the eef and the run dirs can then be deleted.
# Not a dataset build -- do not point convert.slurm at these.
#
# V100 like data_prep_replay_jeanzay.slurm: MuJoCo/EGL rendering, no model inference, and
# RoboCasa365 sim misbehaves on H100. Rendering dominates, so runtime is ~independent of the
# voxel size; 5 episodes with --point-labels is ~2 h against the 4 h asked for here.
#
#SBATCH --job-name=voxelprobe
#SBATCH --account=rgx@v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%A_%a.out
#SBATCH --error=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%A_%a.out

set -euo pipefail

module purge
module load ffmpeg/6.1.1

WORKTREE="$WORK/code/robot-PointAct/.claude/worktrees/pointcount-ablation"
cd "$WORKTREE"
export PYTHONPATH="$WORKTREE:${PYTHONPATH:-}"

# uv points at the MAIN checkout's simulator env, never the worktree's -- see the same note
# in data_prep_replay_jeanzay.slurm and docs/envs.md.
ROBOCASA_ENV="$WORK/code/robot-PointAct/envs/robocasa365"

export MUJOCO_GL=egl
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export UV_OFFLINE=1

source experiments/13_robocasa365/task_paths.sh
TASK="${TASK:-OpenDrawer}"
INPUT_DIR="${INPUT_DIR:-$(task_input_dir "$TASK")}"

# Same five episodes at both sizes, and the same five the 1 cm baseline is measured on, so the
# three columns of the report differ only in the grid.
EPISODES="${EPISODES:-0 1 2 3 4}"

VOXELS=(0.005 0.002)
VOXEL="${VOXELS[$SLURM_ARRAY_TASK_ID]}"
CACHE_DIR="$SCRATCH/voxel_probe/$TASK/v${VOXEL}"

echo "host=$(hostname) task=$TASK voxel=$VOXEL episodes=$EPISODES"
echo "cache_dir=$CACHE_DIR"
nvidia-smi -L

# --point-labels so the report can also count points on the handle itself, which is the
# anchor arm's target and the smallest structure in the scene (~71 points at 1 cm).
uv run --project "$ROBOCASA_ENV" --no-sync \
    python data_prep/robocasa365_to_lerobot/replay.py \
        --input-dir "$INPUT_DIR/lerobot" \
        --cache-dir "$CACHE_DIR" \
        --episodes $EPISODES \
        --voxel-size "$VOXEL" \
        --point-labels \
        --resume
