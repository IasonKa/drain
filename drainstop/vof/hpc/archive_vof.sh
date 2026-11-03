#!/bin/bash
# Copy the results worth keeping from $SCRATCH (purged after ~2 weeks) to $HOME (50 GB quota):
# case set-up, logs, monitors (postProcessing: volumes, flows, probes, free-surface VTK), the analysis
# and the reconstructed last time. The decomposed fields (processor*) are left out.
#   bash vof/hpc/archive_vof.sh $SCRATCH/drainstop_vof/usz_tilt10_full_coarse [dest_dir]
set -euo pipefail
RUN=${1:?run dir}
DEST=${2:-$HOME/drainstop_vof_results}
mkdir -p "$DEST"
NAME=$(basename "$RUN")
LAST=$(cd "$RUN" && ls -d [0-9]* 2>/dev/null | sort -g | tail -1 || true)
tar -czf "$DEST/$NAME.tar.gz" -C "$(dirname "$RUN")" \
    --exclude="$NAME/processor*" --exclude="$NAME/constant/triSurface" \
    "$NAME/case.json" "$NAME/case.env" "$NAME/system" "$NAME/constant" "$NAME/postProcessing" \
    $(cd "$(dirname "$RUN")" && ls -d "$NAME"/log.* "$NAME"/watch.log "$NAME"/analysis "$NAME"/STOPPED "$NAME"/FINISHED 2>/dev/null) \
    ${LAST:+"$NAME/$LAST"}
ls -lh "$DEST/$NAME.tar.gz"
