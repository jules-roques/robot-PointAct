#!/bin/bash
# Jean Zay counterpart to eval_grid.slurm: evaluate one ablation run at 20/30/40/50K as an
# array, one task per checkpoint.
#
#   sbatch --export=ALL,RUN=od-eef-n4096-s0 experiments/13_robocasa365/eval_grid_jeanzay.slurm
#
# A100 (rgx@a100, gpu_p5, qos_gpu_a100-t3): PTv3 uses FlashAttention so V100 is out, and
# RoboCasa365 does not render correctly on H100. Note there is no qos_gpu_a100-t4 -- a100 only
# has dev and t3, which is the likeliest cause of the "Invalid account/partition" once recorded
# in docs/clusters/jean-zay.md.
#
# Array tasks whose checkpoint does not exist yet exit 0, so this can be submitted with
# --dependency=afterany on the training job and resubmitted later for the stragglers.
#
#SBATCH --job-name=evalgrid
#SBATCH --account=rgx@a100
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --time=10:00:00
#SBATCH --array=0-3
#SBATCH --output=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%A_%a.out
#SBATCH --error=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%A_%a.out

set -euo pipefail

module load ffmpeg/6.1.1

REPO="${SLURM_SUBMIT_DIR:-$WORK/code/robot-PointAct}"
cd "$REPO"

RUN="${RUN:?set RUN=<run name, e.g. od-eef-n4096-s0> via --export}"
EXPRS_DIR="${EXPRS_DIR:-$SCRATCH/PointAct_exprs/robocasa365/ablation}"
CKPT_DIR="$EXPRS_DIR/$RUN"

# 50 trials on the intermediate points (enough to draw a curve), 150 at 50K (what it took to
# separate adjacent sampling arms in the earlier ablation).
STEPS=(20000 30000 40000 50000)
TRIALS=(50 50 50 150)
CKPT_STEP="${STEPS[$SLURM_ARRAY_TASK_ID]}"
export NUM_TRIALS="${TRIALS[$SLURM_ARRAY_TASK_ID]}"

if [ ! -d "$CKPT_DIR/checkpoint-$CKPT_STEP" ]; then
    echo "checkpoint-$CKPT_STEP not written yet under $CKPT_DIR; nothing to do."
    exit 0
fi

# Task and the text-context cache both come from the run's own archived data config, so an
# eval can never silently disagree with the training it measures.
# `root` in the data config is repo-root-relative (the robot_data symlink), so the absolute
# paths are just $REPO/<root>/<repo_id>/...
DATA_CONFIG="$CKPT_DIR/data_config.yaml"
read -r ENV_NAME TEXT_CTX <<EOF
$($WORK/code/robot-PointAct/.venv/bin/python -c "
import yaml
ds = yaml.safe_load(open('$DATA_CONFIG'))['lerobot_datasets'][0]
cache = ds.get('text_context_file')
print(ds['repo_id'],
      f\"$REPO/{ds['root']}/{ds['repo_id']}/{cache}\" if cache else '')")
EOF

echo "host=$(hostname) run=$RUN ckpt=$CKPT_STEP env=$ENV_NAME trials=$NUM_TRIALS"
nvidia-smi -L

export TEXT_CONTEXT_FILE="$TEXT_CTX"
bash experiments/13_robocasa365/eval_robocasa365.sh "$ENV_NAME" "$CKPT_DIR" "$CKPT_STEP" "rot6d" "${OPTS:-}"
