#!/usr/bin/env bash
# Relaunch with the same experiment root to resume the standalone A800 run.
set -euo pipefail
base=$(realpath "${1:?supply the A800 experiment directory}")
run="$base/native/coco20k"
py="$base/native/env/bin/python"
mkdir -p "$run"/{data,jittor,lvis/full}
exec 9>"$run/run.lock"
flock -n 9 || { echo "Standalone COCO20k run already active"; exit 0; }
exec >>"$run/run.log" 2>&1
printf '%s\n' "$$" >"$run/run.pid"
: >"$run/run.exit"
trap 'rc=$?; echo "$rc" >"$run/run.exit"; echo "$(date -Iseconds) exit $rc"' EXIT
echo "$(date -Iseconds) standalone COCO20k preflight"
unset PYTHONHOME PYTHONPATH
source "$base/jittor/setup/runtime.sh"
export NVIDIA_TF32_OVERRIDE=0 OPENBLAS_NUM_THREADS=2 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd "$run"

# Snapshot only the standalone entry points and use them for every resumed stage.
if [[ ! -d src ]]; then
    snapshot=$(mktemp -d "$run/src.XXXXXXXX")
    for file in train.py prepare.py model.py objective.py optimizer.py wire.py native_data.py geometry.py native_metrics.py lvis_tp.py compare.py; do
        cp "$base/native/src/$file" "$snapshot/"
    done
    cp -R "$base/native/src/splits" "$snapshot/"
    mv -T "$snapshot" src
fi
for pair in "coco:$base/coco" "jittor/init:$base/native/init" "lvis/inputs:$base/native/benchmark/lvis/inputs"; do
    path=${pair%%:*}
    source_path=${pair#*:}
    [[ -e "$path" ]] || ln -s "$source_path" "$path"
    [[ "$(realpath "$path")" == "$(realpath "$source_path")" ]]
done
"$py" - <<'PY'
import importlib.util
import sys
for name in ("torch", "torchvision", "ultralytics"):
    if importlib.util.find_spec(name) is not None:
        raise SystemExit(f"Unexpected package in {sys.executable}: {name}")
print("Standalone environment verified:", sys.executable)
PY
"$py" -m pip freeze > packages.txt
if [[ ! -f data/coco20k/split.json ]]; then
    "$py" src/prepare.py --profile full --coco-root coco --out data/coco20k
fi
cmp data/coco20k/split.json "$base/data/coco20k/split.json"
cmp data/coco20k/train.txt "$base/data/coco20k/train.txt"
cmp data/coco20k/val.txt "$base/data/coco20k/val.txt"

cmd=("$py" src/train.py --init jittor/init --data data/coco20k --out jittor/full --batch 8 --workers 2 --imgsz 640)
echo "$(date -Iseconds) starting TP, VP, PF, SEG"
if ! "${cmd[@]}"; then
    echo "$(date -Iseconds) training interrupted; retrying from the saved epoch"
    "${cmd[@]}"
fi
echo "$(date -Iseconds) independently reloading all best checkpoints"
"${cmd[@]}" --verify-only
"$py" src/compare.py "$base" --jittor-dir jittor/full
echo "$(date -Iseconds) evaluating the new TP best on LVIS minival"
"$py" src/lvis_tp.py infer --base "$run" --engine jittor
"$py" src/lvis_tp.py score --base "$run" --engine jittor
