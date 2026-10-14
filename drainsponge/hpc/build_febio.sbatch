#!/bin/bash
# Build FEBio 4.13 (the version the drainfeb pilot ran with) from source, with the Pardiso
# solver (needed for friction / non-symmetric stiffness). MKL comes from PyPI, as in
# drainfeb/tools/build_febio_linux.sh (proven); compilers and CMake from the Euler stack.
# FEBio is MIT-licensed (github.com/febiosoftware/FEBio).
#   sbatch hpc/build_febio.sbatch          -> $HOME/febio4/bin/febio4 + $HOME/febio4/env.sh
#SBATCH --job-name=febio-build
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source hpc/env_euler.sh
PREFIX=${FEBIO_PREFIX:-$HOME/febio4}
BUILD=${TMPDIR:-/tmp}/febio_build_$$
TAG=${FEBIO_TAG:-v4.13}
python -m venv "$BUILD/venv" && source "$BUILD/venv/bin/activate"
pip install --quiet ninja mkl-static==2024.2.0 mkl-include==2024.2.0 intel-openmp==2024.2.0
MKLBASE=$BUILD/venv
git clone --depth 1 --branch "$TAG" https://github.com/febiosoftware/FEBio.git "$BUILD/src"
mkdir -p "$BUILD/src/cb" && cd "$BUILD/src/cb"
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMKLROOT="$MKLBASE" -DUSE_MKL=ON \
  -DUSE_HYPRE=OFF -DUSE_MMG=OFF -DUSE_LEVMAR=OFF -DUSE_FFTW=OFF -DUSE_ZLIB=OFF \
  -DSET_DEVCOMMIT=OFF -DUSE_STATIC_STDLIBS=OFF \
  -DCMAKE_EXE_LINKER_FLAGS="-fopenmp" -DCMAKE_SHARED_LINKER_FLAGS="-fopenmp"
ninja -j "${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "$PREFIX/lib"
cp -r bin "$PREFIX/"
cp -r lib/* "$PREFIX/lib/" 2>/dev/null || true
cp "$MKLBASE"/lib/libiomp5.so* "$PREFIX/lib/" 2>/dev/null || true
cat > "$PREFIX/env.sh" <<EOF
export LD_LIBRARY_PATH=$PREFIX/lib:\$LD_LIBRARY_PATH
export FEBIO_BIN=$PREFIX/bin/febio4
EOF
source "$PREFIX/env.sh"
"$PREFIX/bin/febio4" -info || true
rm -rf "$BUILD"
echo "FEBio $TAG installed in $PREFIX"
