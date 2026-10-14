#!/bin/bash
# Copy the results worth keeping from $SCRATCH (purged after ~2 weeks) to $HOME (50 GB quota):
# state, interface history, logs, VTU snapshots, converged OpenFOAM fields. FEBio dumps,
# sub-iteration directories and the OpenFOAM case are left out (they can be regenerated).
#   bash hpc/archive_results.sh $SCRATCH/drainsponge/production_standoff [dest_dir]
set -euo pipefail
RUN=${1:?run dir}
DEST=${2:-$HOME/drainsponge_results}
mkdir -p "$DEST"
NAME=$(basename "$RUN")
tar -czf "$DEST/$NAME.tar.gz" -C "$(dirname "$RUN")" \
    --exclude="*.dmp" --exclude="$NAME/febio/w*/it*" --exclude="$NAME/openfoam/processor*" \
    "$NAME/state.json" "$NAME/config.json" "$NAME/interface_history.npz" "$NAME/progress.log" \
    "$NAME/vtk" "$NAME/openfoam_windows" "$NAME/replay" 2>/dev/null || true
ls -lh "$DEST/$NAME.tar.gz"
