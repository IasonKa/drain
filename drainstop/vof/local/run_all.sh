#!/bin/bash
# Build, mesh, run and picture the fixed-level CFD cases (see vof/local/README.md).
#
#   source /usr/lib/openfoam/openfoam2606/etc/bashrc      # OpenFOAM environment (Linux)
#   bash vof/local/run_all.sh                             # the two main cases, all cores
#   NP=6 bash vof/local/run_all.sh rec_tilt10:8           # one case, 6 cores
#   AIR=10 bash vof/local/run_all.sh                      # 10x denser air: about 3x faster (see README)
#
# Cases are <template>:<pool level in mm>. A case whose run has FINISHED is not run again
# (FORCE=1 reruns it). Pictures and summary.json end up in vof/local/results/<case>/.
set -e
cd "$(dirname "$0")/../.."                                # drainstop/
PY=${PY:-python3}
command -v "$PY" > /dev/null || PY=python
NP=${NP:-$(nproc 2> /dev/null || echo 4)}
AIR=${AIR:-1}
CASES=${*:-"rec_tilt10:8 usz_tilt10:21"}

for t in rec_tilt10 usz_tilt10; do                        # drain surfaces (not in git), ~10 s
    [ -f vof/cases/$t/constant/triSurface/drain.stl ] || $PY vof/make_cases.py --no-estimate --only $t
done
for c in $CASES; do
    tmpl=${c%%:*}; h=${c##*:}
    name=${tmpl}_h${h}; [ "$AIR" != "1" ] && name=${name}_air${AIR}
    run=vof/local/runs/$name
    if [ -f "$run/FINISHED" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "== $name: already finished"
    else
        echo "== $name: building"
        $PY vof/local/make_local_case.py "$tmpl" "$h" --air "$AIR"
        echo "== $name: meshing and running on $NP cores ($(date))"
        NP=$NP "$run/Allrun"
        grep -E "^ *cells:" "$run/log.checkMesh" || true
        echo "== $name: finished ($(date))"
    fi
    $PY vof/local/render_cfd.py "$run"
    mkdir -p "vof/local/results/$name"
    cp "$run"/pictures/* "vof/local/results/$name/"
    cp "$run/model_state.json" "$run/log.checkMesh" "vof/local/results/$name/"
done
echo "Pictures are in vof/local/results/. Commit and push that folder to share them."
