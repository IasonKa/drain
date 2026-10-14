#!/bin/bash
# After a coupled run: replay the whole converged interface-pressure history in ONE FEBio run
# (0 -> t_end, all windows) to get a single continuous plot file for FEBio Studio.
#   sbatch hpc/replay_febio.sbatch $SCRATCH/drainsponge/production_standoff
#SBATCH --job-name=drainsponge-replay
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source hpc/env_euler.sh
RUN=${1:?usage: sbatch hpc/replay_febio.sbatch <run_dir>}
python scripts/make_replay.py "$RUN"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$RUN/replay" && "$FEBIO_BIN" -i replay.feb -silent
echo "open $RUN/replay/replay.xplt in FEBio Studio"
