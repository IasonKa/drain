#!/bin/bash
# Environment for the drainstop VOF check on ETH Euler. Source it in every job script and after login:
#     source vof/hpc/env_euler.sh          (from drain/drainstop)
# Same software stack as drain/drainsponge/hpc/env_euler.sh (stack/2024-06, GCC 12.2.0 layer):
#   OpenFOAM v2306 (ESI edition: interFoam, snappyHexMesh, surfaceFeatureExtract, foamDictionary),
#   OpenMPI 4.1.6, Python 3.11.
# If a module name changed, check with:  module load stack/2024-06 && module avail openfoam

module purge >/dev/null 2>&1
module load stack/2024-06 gcc/12.2.0 openmpi/4.1.6 openfoam/2306 python/3.11.6 \
    || echo "WARNING: a module failed to load - check 'module avail' on Euler and edit vof/hpc/env_euler.sh"

# Some OpenFOAM module builds only set WM_PROJECT_DIR; then the OpenFOAM bashrc must be sourced.
if ! command -v interFoam >/dev/null 2>&1; then
    for f in "${WM_PROJECT_DIR:-/none}/etc/bashrc" "${FOAM_INST_DIR:-/none}/etc/bashrc" "${OPENFOAM_DIR:-/none}/etc/bashrc"; do
        if [ -f "$f" ]; then set +u; source "$f"; break; fi
    done
fi
command -v interFoam >/dev/null 2>&1 || echo "WARNING: interFoam not found after loading openfoam/2306"

# Python with numpy (+ matplotlib for the plots) for analyse_run.py; the run itself needs only the
# standard library (watch_run.py). Created once, see vof/README.md.
VENV=${DRAINSTOP_VENV:-$HOME/venvs/drainstop}
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"

# package root on PYTHONPATH (scripts are run from drain/drainstop)
export DRAINSTOP_ROOT=${DRAINSTOP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export PYTHONPATH=$DRAINSTOP_ROOT:$PYTHONPATH
