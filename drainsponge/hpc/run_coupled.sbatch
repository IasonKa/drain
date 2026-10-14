#!/bin/bash
# Coupled OpenFOAM <-> FEBio run of one design, chained over as many 24 h jobs as needed.
# Each job continues from the last committed coupling window (FEBio dump + OpenFOAM fields +
# interface history in $RUN) and stops cleanly ~30 min before its time limit; it then
# resubmits itself (at most MAX_CHAIN times) until all windows are done.
#
#   sbatch hpc/run_coupled.sbatch hpc/production_standoff.json            # run name from the config file
#   sbatch hpc/run_coupled.sbatch hpc/production_standoff.json my_run     # explicit run name
#   RUN dir: $SCRATCH/drainsponge/<name>   (scratch is purged after ~2 weeks: hpc/archive_results.sh)
#
# One node: OpenFOAM uses all tasks as MPI ranks, then FEBio uses the same cores as OpenMP
# threads (they run one after the other, never at the same time).
#SBATCH --job-name=drainsponge
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source hpc/env_euler.sh
CFG=${1:?usage: sbatch hpc/run_coupled.sbatch <config.json> [run_name]}
NAME=${2:-$(basename "$CFG" .json)}
RUN=${RUN_DIR:-$SCRATCH/drainsponge/$NAME}
CHAIN=${CHAIN:-0}
MAX_CHAIN=${MAX_CHAIN:-12}
mkdir -p "$RUN" logs
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
export OMP_NUM_THREADS=$SLURM_NTASKS
echo "job $SLURM_JOB_ID chain $CHAIN: run $RUN, $SLURM_NTASKS cores, ${WALL}s wall time, host $(hostname)"
python -m drainsponge.couple --config "$CFG" --out "$RUN" \
       --walltime-s "$WALL" --margin-s 1800 \
       --np "$SLURM_NTASKS" --threads "$SLURM_NTASKS" \
       --febio-exe "$FEBIO_BIN" ${EXTRA_ARGS:-}
CODE=$?
echo "coupler exit code $CODE"
if [ "$CODE" -eq 3 ]; then
    if [ "$CHAIN" -lt "$MAX_CHAIN" ]; then
        sbatch --export=ALL,CHAIN=$((CHAIN + 1)),RUN_DIR="$RUN" "$SLURM_SUBMIT_DIR/hpc/run_coupled.sbatch" "$CFG" "$NAME"
        echo "resubmitted (chain $((CHAIN + 1)) of max $MAX_CHAIN)"
    else
        echo "MAX_CHAIN reached - not resubmitting. Continue manually with: sbatch hpc/run_coupled.sbatch $CFG $NAME"
    fi
elif [ "$CODE" -eq 0 ]; then
    echo "run finished. Results: $RUN (vtk/*.pvd for ParaView). Archive: bash hpc/archive_results.sh $RUN"
fi
exit $CODE
