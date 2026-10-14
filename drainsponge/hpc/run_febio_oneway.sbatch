#!/bin/bash
# FEBio only: biphasic tissue loaded with the film pressures of the preview model (one-way).
#   sbatch hpc/run_febio_oneway.sbatch standoff closed
#SBATCH --job-name=drainsponge-febio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source hpc/env_euler.sh
DESIGN=${1:-standoff}; EDGE=${2:-closed}; RES=${3:-fine}
OUT=$SCRATCH/drainsponge/oneway_${DESIGN}_${EDGE}_${RES}
python scripts/build_cases.py --design "$DESIGN" --film-edge "$EDGE" --resolution "$RES" --out "$OUT"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$OUT/febio/oneway" && "$FEBIO_BIN" -i tissue.feb -silent
tail -5 tissue.log
