#!/bin/bash
# One-time setup on Euler (run on a login node from the package root; light work only):
#   1. Python venv with numpy/scipy/meshio (pip install, ~1 min)
#   2. submits the FEBio 4.13 build as a batch job (compiling with many cores belongs in a job)
#
#   cd $HOME/drainsponge && bash hpc/setup_euler.sh
set -e
cd "$(dirname "$0")/.."
source hpc/env_euler.sh
VENV=${DRAINSPONGE_VENV:-$HOME/venvs/drainsponge}
if [ ! -f "$VENV/bin/activate" ]; then
    python -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install --upgrade pip
    pip install numpy scipy meshio
fi
python -c "import numpy, scipy, meshio; print('python env ok:', numpy.__version__, scipy.__version__)"
mkdir -p logs
if [ ! -x "${FEBIO_PREFIX:-$HOME/febio4}/bin/febio4" ]; then
    sbatch hpc/build_febio.sbatch
    echo "FEBio build submitted - check with: squeue -u $USER ; log in logs/"
else
    echo "FEBio already built: ${FEBIO_PREFIX:-$HOME/febio4}/bin/febio4"
fi
echo "next: sbatch hpc/test_euler.sbatch   (15-30 min end-to-end check on the 'test' mesh)"
