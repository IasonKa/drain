#!/bin/bash
# interFoam drainage run of one case of the drainstop VOF check, chained over 24 h jobs.
#
#   sbatch vof/hpc/run_vof.sbatch usz_tilt10                     # full start (3 mm above the drain top), coarse mesh
#   sbatch vof/hpc/run_vof.sbatch rec_tilt10 short               # short start (5 mm above the model's stop level)
#   sbatch vof/hpc/run_vof.sbatch rec_tilt10 full fine           # fine mesh
#   AIR_RHO_FACTOR=10 sbatch vof/hpc/run_vof.sbatch usz_tilt10   # 10x denser air: same stop level, ~3x fewer steps
#   RUN dir: $SCRATCH/drainstop_vof/<case>_<start>_<res>[_air<f>]   (scratch is purged after ~2 weeks)
#
# The first job copies vof/cases/<case> to the run directory and meshes it (Allmesh: snappyHexMesh on
# all tasks). vof/hpc/watch_run.py runs beside interFoam and stops it cleanly (stopAt writeNow):
#   - when the pool has stopped draining (-> STOPPED): the job reconstructs the last time and runs
#     vof/analyse_run.py, which writes $RUN/analysis/summary.txt;
#   - ~30 min before the time limit (-> WALLTIME): the job resubmits itself, at most MAX_CHAIN times,
#     and the next job continues from the latest written time.
# A run can also be stopped with scancel and restarted with the same sbatch line; it continues from
# the latest written time (field output every 0.05-0.1 s of simulated time).
#SBATCH --job-name=drainstop-vof
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source vof/hpc/env_euler.sh
CASE=${1:?usage: sbatch vof/hpc/run_vof.sbatch <usz_tilt10|rec_tilt10> [full|short] [coarse|fine]}
START=${2:-full}
RES=${3:-coarse}
AIR=${AIR_RHO_FACTOR:-1}
NAME=${CASE}_${START}_${RES}
[ "$AIR" != "1" ] && NAME=${NAME}_air$AIR
RUN=${RUN_DIR:-$SCRATCH/drainstop_vof/$NAME}
CHAIN=${CHAIN:-0}
MAX_CHAIN=${MAX_CHAIN:-12}
NP=$SLURM_NTASKS
mkdir -p logs "$(dirname "$RUN")"
echo "job $SLURM_JOB_ID chain $CHAIN: run $RUN, $NP cores, host $(hostname)"

if [ -f "$RUN/STOPPED" ] || [ -f "$RUN/FINISHED" ]; then
    echo "run already finished: $RUN (remove STOPPED/FINISHED there to continue it)"; exit 0
fi
if [ ! -f "$RUN/MESHED" ]; then
    # the STL surfaces are not in git: build them from the analytic geometry (~10 s, numpy)
    if [ ! -f "vof/cases/$CASE/constant/triSurface/drain.stl" ]; then
        python vof/make_cases.py --no-estimate --only "$CASE" || { echo "make_cases.py failed"; exit 1; }
    fi
    rm -rf "$RUN" && cp -r "vof/cases/$CASE" "$RUN" || exit 1
    SECONDS=0
    RES=$RES START=$START NP=$NP AIR_RHO_FACTOR=$AIR bash "$RUN/Allmesh" \
        || { echo "meshing failed - see $RUN/log.*"; tail -20 "$RUN/log.snappyHexMesh"; exit 1; }
    echo "meshing took $SECONDS s"
fi
rm -f "$RUN/WALLTIME"
foamDictionary -entry stopAt -set endTime "$RUN/system/controlDict" > /dev/null

# time left in this job (seconds) from Slurm, fallback 24 h
TL=$(squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null || echo "1-00:00:00")
WALL=$(python - "$TL" <<'PYEOF'
import sys
try:
    s = sys.argv[1].strip()
    d, s = s.split("-") if "-" in s else ("0", s)
    p = [int(x) for x in s.split(":")]
    while len(p) < 3:
        p.insert(0, 0)
    print(int(d) * 86400 + p[0] * 3600 + p[1] * 60 + p[2])
except Exception:
    print(86400)
PYEOF
)
python vof/hpc/watch_run.py "$RUN" --walltime-s "$WALL" --margin-s 1800 >> "$RUN/watch.log" 2>&1 &
WATCH=$!
(cd "$RUN" && mpirun -np "$NP" interFoam -parallel > "log.interFoam.$SLURM_JOB_ID" 2>&1)
CODE=$?
kill "$WATCH" 2>/dev/null
wait "$WATCH" 2>/dev/null
echo "interFoam exit code $CODE"
tail -3 "$RUN/watch.log"

if [ "$CODE" -ne 0 ]; then
    echo "interFoam failed - see $RUN/log.interFoam.$SLURM_JOB_ID"
    tail -30 "$RUN/log.interFoam.$SLURM_JOB_ID"
    exit "$CODE"
fi
if [ -f "$RUN/WALLTIME" ]; then
    if [ "$CHAIN" -lt "$MAX_CHAIN" ]; then
        sbatch --export=ALL,CHAIN=$((CHAIN + 1)),RUN_DIR="$RUN" "$SLURM_SUBMIT_DIR/vof/hpc/run_vof.sbatch" "$CASE" "$START" "$RES"
        echo "resubmitted (chain $((CHAIN + 1)) of max $MAX_CHAIN)"
    else
        echo "MAX_CHAIN reached - continue manually with: RUN_DIR=$RUN sbatch vof/hpc/run_vof.sbatch $CASE $START $RES"
    fi
    exit 0
fi
# the pool stopped draining (STOPPED), or endTime was reached first
[ -f "$RUN/STOPPED" ] || echo "endTime reached before the stop test fired" > "$RUN/FINISHED"
(cd "$RUN" && reconstructPar -latestTime > log.reconstructPar 2>&1) || true
python vof/analyse_run.py "$RUN" --plot || true
echo "run finished. Summary: $RUN/analysis/summary.txt. Archive: bash vof/hpc/archive_vof.sh $RUN"
