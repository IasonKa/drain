#!/bin/bash
# End-to-end check of the Euler setup (modules, Python env, FEBio build, OpenFOAM, restart
# chaining) on the tiny 'test' mesh. Run this before the production job.
#   sbatch hpc/test_euler.sbatch
#SBATCH --job-name=drainsponge-test
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source hpc/env_euler.sh
echo "== tools";  which simpleFoam decomposePar mpirun python "$FEBIO_BIN" || true
"$FEBIO_BIN" -info 2>&1 | head -5 || true
python -c "import numpy, scipy, meshio; print('python ok')"
echo "== mock-solver unit test (plumbing only)"
python tests/test_coupling.py "$SCRATCH/drainsponge/_unit_test" | tail -3
echo "== real solvers, 3 windows"
RUN=$SCRATCH/drainsponge/test_euler
rm -rf "$RUN"
export OMP_NUM_THREADS=$SLURM_NTASKS
python -m drainsponge.couple --config hpc/test_euler.json --out "$RUN" \
       --np "$SLURM_NTASKS" --threads "$SLURM_NTASKS" --febio-exe "$FEBIO_BIN"
CODE=$?
echo "exit code $CODE (0 = ok)"; tail -20 "$RUN/progress.log"
grep -h "restart" "$RUN/progress.log" | head -3
exit $CODE
