#!/bin/bash
# Check of the Euler set-up for the VOF cases, and calibration of the run-time estimate. Meshes one
# case (coarse) and runs interFoam for 20 ms from the short start: the drain interior is above the
# pool level from the start, so air reaches the bore at once and the time step is that of the
# two-phase part of the runs. Prints cells, seconds per step, time step and the projected wall time.
# Run this before the production jobs (once per case if you want both cell counts).
#   sbatch vof/hpc/test_vof.sbatch                 # rec_tilt10 (36 openings, meshes fastest)
#   sbatch vof/hpc/test_vof.sbatch usz_tilt10
#SBATCH --job-name=drainstop-vof-test
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source vof/hpc/env_euler.sh
CASE=${1:-rec_tilt10}
RUN=$SCRATCH/drainstop_vof/test_$CASE
NP=$SLURM_NTASKS
echo "== tools"
which blockMesh surfaceFeatureExtract snappyHexMesh topoSet setFields interFoam foamDictionary mpirun python || true
python -c "import numpy; print('numpy', numpy.__version__)" || echo "numpy missing: analyse_run.py needs it (see vof/README.md)"
mkdir -p logs "$(dirname "$RUN")"
if [ ! -f "vof/cases/$CASE/constant/triSurface/drain.stl" ]; then
    echo "== building the STL surfaces (not in git)"
    python vof/make_cases.py --no-estimate --only "$CASE" || { echo "make_cases.py failed"; exit 1; }
fi
rm -rf "$RUN" && cp -r "vof/cases/$CASE" "$RUN"
echo "== mesh ($CASE, coarse, short start)"
SECONDS=0
RES=coarse START=short NP=$NP bash "$RUN/Allmesh" || { echo "meshing failed"; tail -30 "$RUN/log.snappyHexMesh"; exit 1; }
echo "meshing took $SECONDS s"
echo "== interFoam, 20 ms of simulated time"
foamDictionary -entry endTime -set 0.02 "$RUN/system/controlDict" > /dev/null
foamDictionary -entry writeInterval -set 0.01 "$RUN/system/controlDict" > /dev/null
SECONDS=0
(cd "$RUN" && timeout 4500 mpirun -np "$NP" interFoam -parallel > log.interFoam.test 2>&1)
CODE=$?
echo "interFoam exit code $CODE after $SECONDS s (0 = ok, 124 = cut at 75 min)"
tail -8 "$RUN/log.interFoam.test"
python vof/analyse_run.py "$RUN" --timing
exit $CODE
