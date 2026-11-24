#!/usr/bin/env bash
# Queue the three COCO20k branches behind the repaired TP training and evaluation.
set -euo pipefail
base=$(realpath "${1:?supply the experiment root}")
tp=$(realpath "${2:?supply the repaired TP run directory}")
run=$(realpath "${3:?supply a separate branch run directory}")
scripts=$(dirname "$(realpath "$0")")
[[ "$tp" != "$run" ]]
py="$base/native/env/bin/python"
src="$tp/src"
data="$base/native/coco20k/data/coco20k"
init="$tp/jittor/init"
out="$run/jittor/full"
cd "$run"
exec 9>>run.lock
flock -n 9 || { echo "Branch queue already active in $run"; exit 0; }
exec >>run.log 2>&1
status() {
    printf '%s %s\n' "$(date -Iseconds)" "$*" | tee status.tmp
    mv status.tmp status
}
on_exit() {
    local rc=$?
    if (( rc != 0 )); then status "${failure_detail:-failed (exit $rc), see run.log}"; fi
    printf '%s\n' "$rc" >run.exit.tmp
    mv run.exit.tmp run.exit
}
trap on_exit EXIT
echo "$$" >run.pid
: >run.exit
status preflight
[[ -x "$py" && -r "$scripts/compare.py" && -r "$tp/run.lock" ]]
(cd "$tp" && sha256sum --quiet -c SHA256SUMS)
(cd "$init" && sha256sum --quiet -c - <<'SUMS'
1836e27d184a8e3995d4d0698dda971fa23fc9cbdf2a491050ca2f0202ae137b  seg.json
e1113357bb8cb70bb4da0757db6c1fb517543ef175e15d7f7fcae64a7bbeef01  seg.npz
SUMS
)
unset PYTHONPATH PYTHONHOME
source "$base/jittor/setup/runtime.sh"
export NVIDIA_TF32_OVERRIDE=0 OPENBLAS_NUM_THREADS=2 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
status "queued: waiting for repaired TP training, COCO and LVIS ($tp/run.lock)"
exec 8<"$tp/run.lock"
flock -s 8
if [[ ! -s "$tp/run.exit" || $(cat "$tp/run.exit") != 0 ]]; then
    echo "Repaired TP exited without successful training and evaluation."
    exit 1
fi

# Freeze the evaluated best once. All three trainer calls read out/TP/best.npz.
# Configuration and history are copied for the final four-stage comparison.
mkdir -p "$out"
parent_sha=$("$py" - "$tp" "$out" <<'PY'
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

tp, out = map(Path, sys.argv[1:])
sys.path.insert(0, str(tp / "src"))
from compare_repaired_tp import jittor_run, lvis, read, sha

row = jittor_run(tp, True)
lvis(tp, "jittor", row)
if row != read(tp / "tp-comparison.json")["runs"]["repaired_native_jittor"]:
    raise ValueError("TP comparison does not describe the evaluated checkpoint")
if sha(tp / "lvis/full/jittor.npz") != row["lvis_provenance"]["predictions_sha256"]:
    raise ValueError("TP LVIS predictions differ from the completed evaluation")
source = tp / "jittor/full/TP"
snapshot = out / "TP"
files = ("best.npz", "verified.json", "epochs.json", "config.json")
if not snapshot.exists():
    with TemporaryDirectory(dir=out) as tmp:
        staged = Path(tmp) / "TP"
        staged.mkdir()
        for name in files:
            shutil.copy2(source / name, staged / name)
        shutil.copy2(source.parent / "independent-validation.json", staged / "independent-validation.json")
        staged.rename(snapshot)
for name in files:
    if sha(snapshot / name) != sha(source / name):
        raise ValueError(f"TP snapshot changed: {name}")
if sha(snapshot / "independent-validation.json") != sha(source.parent / "independent-validation.json"):
    raise ValueError("TP validation snapshot differs from the evaluated source")
print(row["checkpoint"]["sha256"])
PY
)
flock -u 8
exec 8<&-
echo "All three branches use evaluated TP best $parent_sha"
check_branch() {
    "$py" - "$src" "$out" "$parent_sha" "$1" "$2" <<'PY'
import json
import math
import sys
from pathlib import Path

src, out = map(Path, sys.argv[1:3])
signature, stage, report_path = sys.argv[3:]
sys.path.insert(0, str(src))
from compare_repaired_tp import sha

directory = out / stage
read = lambda path: json.loads(Path(path).read_bytes())
info = read(directory / "verified.json")
cfg = read(directory / "config.json")
reloaded = read(report_path)[stage]
if not (info["stage"] == cfg["stage"] == stage and info["epochs"] == cfg["epochs"] == 10
        and info["parent_sha"] == cfg["parent_sha"] == signature and info["images"] == cfg["nval"] == 5000
        and info["steps"] > 0 and cfg["ntrain"] == 20000 and cfg["batch"] == 8 and cfg["size"] == 640
        and cfg["data_backend"] == "native-v1" and not cfg["smoke"]):
    raise ValueError(f"Incomplete or different branch: {stage}")
for name in ("best", "last"):
    if sha(directory / (name + ".npz")) != info[name + "_sha"]:
        raise ValueError(f"Checkpoint changed: {stage}/{name}")
expected = {"best", "coco80"} if stage == "PF" else {"best"}
if reloaded.keys() != expected or info["metrics"].keys() != expected:
    raise ValueError(f"Unexpected validation variants: {stage}")
for variant in expected:
    metrics = reloaded[variant]
    needed = {"metrics/mAP50-95(B)"}
    if stage == "SEG":
        needed.add("metrics/mAP50-95(M)")
    if not needed.issubset(metrics) or metrics.keys() != info["metrics"][variant].keys():
        raise ValueError(f"Incomplete validation: {stage}/{variant}")
    for name, value in metrics.items():
        if not math.isclose(value, info["metrics"][variant][name], rel_tol=1e-4, abs_tol=1e-5):
            raise ValueError(f"Reload metric changed: {stage}/{variant}/{name}")
PY
}

cmd=("$py" "$src/train.py" --init "$init" --data "$data" --out "$out" --batch 8 --workers 2 --imgsz 640 --device cuda)
failed=()
for stage in VP PF SEG; do
    printf '%s  %s\n' "$parent_sha" "$out/TP/best.npz" | sha256sum --quiet -c -
    report="$out/$stage/independent-validation.json"
    if [[ -s "$report" ]] && check_branch "$stage" "$report"; then
        status "$stage already trained and independently verified"
        continue
    fi
    while true; do
        gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
        [[ -n "$gpu_pids" ]] || break
        status "waiting for GPU processes before $stage: $gpu_pids"
        sleep 15
    done
    trained=false
    for attempt in 1 2; do
        status "training $stage: 10 epochs from repaired TP best $parent_sha (attempt $attempt)"
        if "${cmd[@]}" --stages "$stage"; then trained=true; break; fi
    done
    verified=false
    if [[ "$trained" == true ]]; then
        for attempt in 1 2; do
            status "independently reloading $stage on COCO val2017 (attempt $attempt)"
            if "${cmd[@]}" --verify-only --stages "$stage" &&
               check_branch "$stage" "$out/independent-validation.json"; then
                mv "$out/independent-validation.json" "$report"
                verified=true
                break
            fi
        done
    fi
    if [[ "$verified" != true ]]; then
        failed+=("$stage")
        status "$stage failed; proceeding with the other TP branches"
    fi
done
if (( ${#failed[@]} )); then
    failure_detail="failed branches: ${failed[*]}; verified branches will be reused on restart"
    exit 1
fi
"$py" - "$out" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
report = {}
for stage in ("TP", "VP", "PF", "SEG"):
    report[stage] = json.loads((out / stage / "independent-validation.json").read_bytes())[stage]
report_path = out / "independent-validation.json"
tmp = report_path.with_suffix(".tmp")
tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
tmp.replace(report_path)
PY
"$py" "$scripts/compare.py" "$base" --jittor-dir "$out"
status complete
