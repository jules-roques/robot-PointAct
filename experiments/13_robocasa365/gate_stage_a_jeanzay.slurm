#!/bin/bash
# Jean Zay counterpart to gate_stage_a.slurm: mail the stage-A point-count x sampling table so
# the point count for stage B can be chosen by hand.
#
#   sbatch --dependency=afterok:<eval jobids> experiments/13_robocasa365/gate_stage_a_jeanzay.slurm
#
# Stage B is deliberately NOT launched from here -- choosing the point count is the judgement
# call this gate exists to hand back to a person.
#
# Caveat: HPC sites often block or silently drop mail to external addresses. If nothing
# arrives, the table is still in this job's log and in $SUMMARY_OUT. Verify delivery once with
#   sbatch --account=rgx@cpu --partition=cpu_p1 --mail-type=END \
#          --mail-user=jules28.roques@gmail.com --wrap='echo mail probe'
#
#SBATCH --job-name=gate-stageA
#SBATCH --account=rgx@cpu
#SBATCH --partition=cpu_p1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --time=00:15:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jules28.roques@gmail.com
#SBATCH --output=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/rgx/unw24jl/logs/robocasa365/%x_%j.out

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$WORK/code/robot-PointAct}"
cd "$REPO"

EXPRS_DIR="${EXPRS_DIR:-$SCRATCH/PointAct_exprs/robocasa365/ablation}"
SUMMARY_OUT="${SUMMARY_OUT:-$EXPRS_DIR/stage_a_summary.txt}"

# --mail-type=END mails this job's stdout tail, so printing the table here is what puts it in
# the message body.
"$WORK/code/robot-PointAct/.venv/bin/python" \
    experiments/13_robocasa365/summarize_stage_a.py \
    --exprs-dir "$EXPRS_DIR" \
    --out "$SUMMARY_OUT"

echo
echo "Full table also at: $SUMMARY_OUT"
echo "Then: python experiments/13_robocasa365/runs/generate_stage_b.py --npoints <N>"
