#!/usr/bin/env bash
# Run the repaired TP experiment after the original standalone A800 pipeline.
set -euo pipefail
base=$(realpath "${1:?supply the experiment root}")
run=$(realpath "${2:?supply the deployed repaired run directory}")
old="$base/native/coco20k"
py="$base/native/env/bin/python"
cd "$run"
exec 9>run.lock
flock -n 9 || exit 0
if [[ -s run.exit && $(cat run.exit) == 0 ]]; then exit 0; fi
exec >>run.log 2>&1
status() {
    printf '%s %s\n' "$(date -Iseconds)" "$*" | tee status.tmp
    mv status.tmp status
}
on_exit() {
    local rc=$?
    if (( rc != 0 )); then status "failed (exit $rc)"; fi
    printf '%s\n' "$rc" > run.exit.tmp
    mv run.exit.tmp run.exit
}
trap on_exit EXIT
printf '%s\n' "$$" >run.pid
: >run.exit
status preflight
sha256sum --quiet -c SHA256SUMS
unset PYTHONPATH PYTHONHOME
source "$base/jittor/setup/runtime.sh"
export NVIDIA_TF32_OVERRIDE=0 OPENBLAS_NUM_THREADS=2 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
data="$old/data/coco20k"
mkdir -p jittor lvis
for pair in "coco:$old/coco" "jittor/init:$old/jittor/init" "lvis/inputs:$old/lvis/inputs"; do
    path=${pair%%:*}
    target=${pair#*:}
    if [[ ! -e "$path" ]]; then ln -s "$target" "$path"; fi
    [[ "$(realpath "$path")" == "$(realpath "$target")" ]]
done
"$py" - "$data" <<'PY'
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

data = Path(sys.argv[1])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
load = lambda p: json.loads(p.read_bytes())
split = load(data / 'split.json')
if split['profile'] != 'full':
    raise ValueError('Expected full COCO20k profile')
for key, n, expected in (
    ('train', 20000, '0791b2269c1f041746d159571b75b57f458bff00041d7d34b0b748dd6a17fa45'),
    ('val', 5000, '612097d209d870b260f9b4cb9c5ccdc32586da617ccf0f25c0627e60f8877285'),
):
    record = split['splits'][key]
    if record['images'] != n or record['manifest_sha256'] != expected or sha(data / (key+'.txt')) != expected:
        raise ValueError('Split mismatch: ' + key)
for name in ('torch','torchvision','ultralytics'):
    if importlib.util.find_spec(name):
        raise RuntimeError('Unexpected training dependency: ' + name)
for name in ('jittor','lvis','pycocotools'):
    if not importlib.util.find_spec(name):
        raise RuntimeError('Missing training/evaluation package: ' + name)
init = Path('jittor/init')
for name, expected in {
    'det.json':'37edbe23635c3d6f3f966bff65e4dfa20c4b66b6e5fd292e22e1d184236de9ad',
    'det.npz':'6047e8d4f5b3613cd15a8baf71e03fb7428f6eaec883f16c9d4f7aa86408f416',
    'raw-text.npy':'47466ad1fa74e5b8f7bafebfed19e20f940e99f92a0a7aa2e814a36931bd2e19'
}.items():
    if sha(init / name) != expected:
        raise ValueError('Initialization mismatch: '+name)
inputs = Path('lvis/inputs')
manifest = load(inputs / 'manifest.json')
if len(manifest['images']) != 4809 or len(manifest['names']) != 1203 or sha(inputs/'text.npy') != manifest['text_sha']:
    raise ValueError('Unexpected LVIS split or text cache')
for name, expected in manifest['asset_sha256'].items():
    if sha(inputs/name) != expected:
        raise ValueError('LVIS asset mismatch: '+name)
print('Preflight passed: code checksums, standalone environment, COCO20k/5k, training initialization and LVIS inputs')
PY
status "queued: waiting for $old/run.lock"
exec 8<"$old/run.lock"
flock -x 8
printf 'Predecessor exit: %s\n' "$(cat "$old/run.exit")"
# The new TP run starts from the same initialization even if a predecessor failed.
while true; do
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
    if [[ -z "$gpu_pids" ]]; then break; fi
    status "waiting for GPU processes: $gpu_pids"
    sleep 60
done
status "training TP: 30 epochs, batch 8, 640px"
cmd=("$py" src/train.py --init jittor/init --data "$data" --out jittor/full --batch 8 --workers 2 --imgsz 640 --device cuda --stages TP)
if ! "${cmd[@]}"; then
    [[ -s jittor/full/TP/training-state.npz ]] || exit 1
    status "retrying TP from saved epoch"
    "${cmd[@]}"
fi
status "reloading TP best and validating COCO val2017"
"${cmd[@]}" --verify-only
status "running LVIS minival inference"
"$py" src/lvis_tp.py infer --base "$run" --engine jittor --device cuda:0
status "scoring LVIS Fixed AP"
"$py" src/lvis_tp.py score --base "$run" --engine jittor
status "comparing COCO and LVIS metrics"
"$py" src/compare_repaired_tp.py --base "$base" --run "$run"
status complete
