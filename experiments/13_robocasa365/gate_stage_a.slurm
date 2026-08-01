#!/bin/bash
# Stage gate: once every stage-A eval has finished, mail the point-count x sampling table so
# the point count for stage B can be chosen by hand.
#
# Submit it held on the stage-A eval jobs, so it fires exactly once when they are all done:
#   sbatch --dependency=afterok:<jobid1>:<jobid2>:... \
#          experiments/13_robocasa365/gate_stage_a.slurm
#
# (submit_stage_a.sh collects the job ids and does this for you.)
#
# Stage B is deliberately NOT launched from here. Choosing the point count is the judgement
# call this gate exists to hand back to a person.
#
# Caveat worth checking once before relying on it: HPC sites frequently block or silently drop
# mail to external addresses. Verify delivery with a throwaway job before trusting the gate --
#   sbatch --mail-type=END --mail-user=jules28.roques@gmail.com --wrap='echo mail probe'
# -- otherwise the summary still lands in the log and $SUMMARY_OUT, but nothing tells you.
#
#SBATCH --job-name=gate-stageA
#SBATCH --account=willow
#SBATCH --partition=cpu_homogen
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jules28.roques@gmail.com
#SBATCH --output=/scratch/jroques/logs/robocasa365/%x_%j.out
#SBATCH --error=/scratch/jroques/logs/robocasa365/%x_%j.out

set -euo pipefail

REPO="$HOME/code/robot-PointAct"
cd "$REPO"

EXPRS_DIR="${EXPRS_DIR:-$SCRATCH/PointAct_exprs/robocasa365/ablation}"
SUMMARY_OUT="${SUMMARY_OUT:-$EXPRS_DIR/stage_a_summary.txt}"

# SLURM mails the job's stdout tail with --mail-type=END, so printing the table here is what
# puts it in the message body.
/home/jroques/code/robot-PointAct/.venv/bin/python \
    experiments/13_robocasa365/summarize_stage_a.py \
    --exprs-dir "$EXPRS_DIR" \
    --out "$SUMMARY_OUT"

echo
echo "Full table also at: $SUMMARY_OUT"
