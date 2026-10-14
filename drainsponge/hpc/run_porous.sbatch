#!/bin/bash
# Stand-alone OpenFOAM Darcy-Brinkman case (rigid porous tissue, no FEBio): how much liquid
# comes through the film / slots and how much out of the sponge, for a frozen tissue shape.
#   sbatch hpc/run_porous.sbatch standoff closed      # design: current|standoff, edge: open|closed
#SBATCH --job-name=drainsponge-porous
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source hpc/env_euler.sh
DESIGN=${1:-standoff}; EDGE=${2:-closed}; RES=${3:-fine}
OUT=$SCRATCH/drainsponge/porous_${DESIGN}_${EDGE}_${RES}
python scripts/build_cases.py --design "$DESIGN" --film-edge "$EDGE" --resolution "$RES" --out "$OUT" --no-preview
cd "$OUT/openfoam/porous_initial" && NP=$SLURM_NTASKS bash Allrun
cat postProcessing/*/0/*.dat 2>/dev/null | tail -20 || true
