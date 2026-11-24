#!/usr/bin/env bash
set -euo pipefail
cd "${1:?supply the experiment directory}"
mode=${2:-full}
[[ "$mode" == full || "$mode" == smoke ]]
mkdir -p lvis/logs "lvis/$mode"
exec 8>"lvis/$mode.lock"
flock -n 8 || exit 0
exec >>"lvis/logs/$mode.log" 2>&1
echo $$ >"lvis/$mode.pid"
: >"lvis/$mode.exit"
trap 'rc=$?; printf "%s\n" "$rc" > "lvis/$mode.exit"; echo "$(date -Iseconds) $mode exit $rc"' EXIT
source jittor/setup/runtime.sh
export NVIDIA_TF32_OVERRIDE=0 PYTHONUNBUFFERED=1
app="$PWD/jittor/src/lvis_tp.py"
pt="$PWD/env/bin/python"
jt="$PWD/jittor/env/bin/python"
cpu="$PWD/lvis/metric-env/bin/python"
extra=()
if [[ "$mode" == smoke ]]; then
    extra=(--smoke)
else
    echo "$(date -Iseconds) waiting for the two complete four-stage runs and LVIS preflight"
    while [[ ! -s jittor/chain.exit || ! -s lvis/smoke.exit ]]; do
        for status in jittor/chain.exit lvis/smoke.exit; do
            if [[ -s "$status" && "$(cat "$status")" != 0 ]]; then echo "Failed predecessor: $status"; exit 1; fi
        done
        sleep 10
    done
    [[ "$(cat jittor/chain.exit)" == 0 && "$(cat lvis/smoke.exit)" == 0 ]]
    [[ -s jittor/full/framework-comparison.json && -s runs/coco20k/summary.json && -s lvis/smoke/comparison.json ]]
fi
[[ -s lvis/inputs/manifest.json ]]
for engine in pytorch jittor; do
    runtime=$pt
    if [[ "$engine" == jittor ]]; then runtime=$jt; fi
    echo "$(date -Iseconds) $mode $engine"
    if [[ ! -s "lvis/$mode/$engine-inference.json" ]]; then
        "$runtime" "$app" infer --engine "$engine" "${extra[@]}"
    fi
    "$cpu" "$app" score --engine "$engine" "${extra[@]}"
done
"$cpu" "$app" compare "${extra[@]}"
