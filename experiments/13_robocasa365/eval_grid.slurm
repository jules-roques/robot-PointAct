#!/bin/bash
# Evaluate one ablation run at all four checkpoints, as a SLURM array (one task per step).
#
#   sbatch --export=ALL,RUN=od-eef-n4096-s0 experiments/13_robocasa365/eval_grid.slurm
#
# The array shape is deliberate: each checkpoint is evaluated the moment it exists, so evals
# overlap with the training that is still producing later checkpoints, on a different GPU.
# Array tasks whose checkpoint is not written yet exit 0 and are simply re-submitted later --
# they never block the ones that are ready.
#
# 50 trials at 20K/30K/40K, 150 at 50K: the intermediate points only have to draw a curve,
# while the headline arm comparison at 50K needs the resolution (n=150 was what separated
# adjacent sampling arms in the earlier ablation; n=50 only separated the endpoints).
#
# A100, not V100: PTv3 uses FlashAttention (Ampere+) and V100 has no bf16. Not H100 either --
# RoboCasa365 does not render correctly there. Available on both clusters; on Jean Zay use
# eval_grid_jeanzay.slurm, whose account/QoS differ (rgx@a100, qos_gpu_a100-t3; note there is
# no t4 QoS for a100).
#
#SBATCH --job-name=evalgrid
#SBATCH --account=willow
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/jroques/logs/robocasa365/%x_%A_%a.out
#SBATCH --error=/scratch/jroques/logs/robocasa365/%x_%A_%a.out

set -euo pipefail

REPO="$HOME/code/robot-PointAct"
cd "$REPO"

RUN="${RUN:?set RUN=<run name, e.g. od-eef-n4096-s0> via --export}"
EXPRS_DIR="${EXPRS_DIR:-$SCRATCH/PointAct_exprs/robocasa365/ablation}"
CKPT_DIR="$EXPRS_DIR/$RUN"

STEPS=(20000 30000 40000 50000)
TRIALS=(50 50 50 150)
CKPT_STEP="${STEPS[$SLURM_ARRAY_TASK_ID]}"
# Overridable so a smoke test can ask for a handful of trials; the client only
# writes results at the end, so a long default would produce nothing before a
# short walltime killed it.
export NUM_TRIALS="${NUM_TRIALS:-${TRIALS[$SLURM_ARRAY_TASK_ID]}}"

if [ ! -d "$CKPT_DIR/checkpoint-$CKPT_STEP" ]; then
    echo "checkpoint-$CKPT_STEP not written yet under $CKPT_DIR; nothing to do."
    exit 0
fi

# Task, rotation type and the sampling mode all come from the run's own archived config, so
# an eval can never silently disagree with the training it is measuring. Evaluating an
# eef-trained policy on uniformly drawn clouds is a train/test mismatch that reads as "the
# ablation does not help".
DATA_CONFIG="$CKPT_DIR/data_config.yaml"
ENV_NAME=$(/home/jroques/code/robot-PointAct/.venv/bin/python -c "
import yaml, sys
ds = yaml.safe_load(open('$DATA_CONFIG'))['lerobot_datasets'][0]
print(ds['repo_id'])")
ROT="${ROT:-rot6d}"

echo "host=$(hostname) run=$RUN ckpt=$CKPT_STEP env=$ENV_NAME trials=$NUM_TRIALS"
nvidia-smi -L

bash experiments/13_robocasa365/eval_robocasa365.sh "$ENV_NAME" "$CKPT_DIR" "$CKPT_STEP" "$ROT" "${OPTS:-}"
