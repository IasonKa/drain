#!/bin/bash
# Environment for drainsponge on ETH Euler. Source it in every job script and after login:
#     source hpc/env_euler.sh
# Modules: Euler software stack 2024-06 (GCC 12.2.0 layer), see https://docs.hpc.ethz.ch/software/software-stack/
#   OpenFOAM v2306 (ESI edition: simpleFoam + fvOptions), OpenMPI 4.1.6, Python 3.11, CMake.
# If a module name changed, check with:  module load stack/2024-06 && module avail openfoam

module purge >/dev/null 2>&1
module load stack/2024-06 gcc/12.2.0 openmpi/4.1.6 openfoam/2306 python/3.11.6 cmake/3.27.7 \
    || echo "WARNING: a module failed to load - check 'module avail' on Euler and edit hpc/env_euler.sh"

# Some OpenFOAM module builds only set WM_PROJECT_DIR; then the OpenFOAM bashrc must be sourced.
if ! command -v simpleFoam >/dev/null 2>&1; then
    for f in "${WM_PROJECT_DIR:-/none}/etc/bashrc" "${FOAM_INST_DIR:-/none}/etc/bashrc" "${OPENFOAM_DIR:-/none}/etc/bashrc"; do
        if [ -f "$f" ]; then set +u; source "$f"; break; fi
    done
fi
command -v simpleFoam >/dev/null 2>&1 || echo "WARNING: simpleFoam not found after loading openfoam/2306"

# Python environment with numpy/scipy/meshio (created once by hpc/setup_euler.sh)
VENV=${DRAINSPONGE_VENV:-$HOME/venvs/drainsponge}
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"

# FEBio 4.13 built from source by hpc/build_febio.sbatch
FEBIO_PREFIX=${FEBIO_PREFIX:-$HOME/febio4}
[ -f "$FEBIO_PREFIX/env.sh" ] && source "$FEBIO_PREFIX/env.sh"
export FEBIO_BIN=${FEBIO_BIN:-$FEBIO_PREFIX/bin/febio4}

# package root on PYTHONPATH (scripts are run from the package root)
export DRAINSPONGE_ROOT=${DRAINSPONGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export PYTHONPATH=$DRAINSPONGE_ROOT:$PYTHONPATH
