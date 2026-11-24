#!/usr/bin/env bash
# Start once per A800 run. All paths are on the instance's persistent training disk.
set -euo pipefail
cd "${1:?supply the PyTorch experiment directory}"
base=$PWD
mkdir -p jittor/logs
exec 9>jittor/chain.lock
flock -n 9 || { echo "Jittor chain already running"; exit 0; }
exec >>jittor/logs/chain.log 2>&1
trap 'rc=$?; echo "$(date -Iseconds) chain exit $rc"; echo "$rc" > jittor/chain.exit' EXIT
: >jittor/chain.exit
printf '%s\n' "$$" > jittor/chain.pid
echo "$(date -Iseconds) waiting for verified PyTorch completion and Jittor preflight"
while [[ ! -s runs/coco20k/summary.json || ! -s setup/full.exit ||
         ! -s jittor/logs/parity.json || ! -s jittor/checks/math-parity.json ||
         ! -s jittor/smoke/independent-validation.json ]]; do
    if [[ -s setup/full.exit && "$(cat setup/full.exit)" != 0 ]]; then
        echo "$(date -Iseconds) PyTorch exited with code $(cat setup/full.exit); inspect logs/full.log"
        exit 1
    fi
    sleep 10
done
[[ "$(cat setup/full.exit)" == 0 ]]
source jittor/setup/runtime.sh
export NVIDIA_TF32_OVERRIDE=0
jt="$base/jittor/env/bin/python"
cmd=("$jt" -u jittor/src/train.py --init jittor/init --data data/coco20k --out jittor/full --batch 8 --workers 2 --imgsz 640)
echo "$(date -Iseconds) PyTorch and Jittor preflight verified; starting Jittor COCO20k"
resume_run() {
    local attempt
    for attempt in 1 2 3; do
        if "$@"; then return 0; fi
        echo "$(date -Iseconds) attempt $attempt failed: $*"
        if [[ $attempt -lt 3 ]]; then sleep 20; fi
    done
    return 1
}
resume_run "${cmd[@]}"
echo "$(date -Iseconds) four stages finished; reloading checkpoints in a fresh process"
resume_run "${cmd[@]}" --verify-only
"$jt" jittor/src/compare.py "$base"
echo "$(date -Iseconds) Jittor and framework comparison complete"
