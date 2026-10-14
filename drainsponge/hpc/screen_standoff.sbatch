#!/bin/bash
# Standoff ring screening with the fast preview model: one design per array task (1 core,
# ~2-5 min each), then one job collects the results and draws the figure.
#   sbatch --array=0-45%20 hpc/screen_standoff.sbatch
#   sbatch --dependency=afterok:<array_job_id> hpc/screen_standoff.sbatch collect
#SBATCH --job-name=drainsponge-screen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source hpc/env_euler.sh
OUT=$SCRATCH/drainsponge/screening
if [ "${1:-}" = "collect" ]; then
    python scripts/optimize_standoff.py --out "$OUT" --figs "$OUT" --collect
else
    python scripts/optimize_standoff.py --out "$OUT" --index "$SLURM_ARRAY_TASK_ID" --workers 1
fi
