"""Compare COCO20k TP checkpoints after independent COCO and LVIS evaluation."""
import argparse
import hashlib
import json
import math
from pathlib import Path


SPLITS = {
    "train": (20000, "0791b2269c1f041746d159571b75b57f458bff00041d7d34b0b748dd6a17fa45"),
    "val": (5000, "612097d209d870b260f9b4cb9c5ccdc32586da617ccf0f25c0627e60f8877285"),
}


def read(path):
    return json.loads(path.read_bytes())


def check(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def metric(value, scale):
    check(math.isfinite(value) and 0 <= value <= scale, f"Invalid metric on 0–{scale} scale: {value}")
    return value * (100 / scale)


def coco_scores(metrics):
    return {name: metric(metrics[f"metrics/m{name}(B)"], 1) for name in ("AP50-95", "AP50")}


def checkpoint(path, recorded, required):
    check(len(recorded) == 64 and all(c in "0123456789abcdef" for c in recorded),
          f"Invalid checkpoint SHA-256 at {path}")
    present = path.is_file()
    if present or required:
        check(sha(path) == recorded, f"Checkpoint differs from recorded SHA-256: {path}")
    return {"sha256": recorded, "file_sha256_verified": present}


def pytorch_baseline(base):
    directory = base / "runs/coco20k"
    summary = read(directory / "summary.json")
    for split, (count, signature) in SPLITS.items():
        record = summary["split"]["splits"][split]
        check((record["images"], record["manifest_sha256"]) == (count, signature), f"PyTorch {split} split")
    tp = summary["stages"]["TP"]
    val = tp["standalone_validation"]["best"]
    check(tp["epochs"] == 30 and val["images"] == 5000 and tp["optimizer_steps"] > 0, "Incomplete PyTorch TP")
    check(Path(val["weights"]).as_posix() == "TP/weights/best.pt", "Unexpected PyTorch TP validation weights")
    return {"epochs": 30, "optimizer_steps": tp["optimizer_steps"], "coco": coco_scores(val["metrics"]),
            "checkpoint": checkpoint(directory / val["weights"], tp["best_sha256"], False)}


def jittor_run(run, repaired):
    directory = run / "jittor/full"
    tp = directory / "TP"
    cfg, verified = read(tp / "config.json"), read(tp / "verified.json")
    for split, (count, signature) in SPLITS.items():
        check((cfg["n"+split], cfg[split+"_sha"]) == (count, signature), f"{run}: {split} split")
    check(cfg["stage"] == verified["stage"] == "TP" and cfg["epochs"] == verified["epochs"] == 30
          and cfg["size"] == 640 and cfg["batch"] == 8 and not cfg["smoke"]
          and verified["images"] == 5000 and verified["steps"] > 0, f"{run}: incomplete or different TP recipe")
    check(cfg["data_backend"] == "native-v1" and verified["parent_sha"] == cfg["parent_sha"],
          f"{run}: unexpected initialization/data backend")
    if repaired:
        check(cfg["musgd_revision"] == 2, f"{run}: expected repaired MuSGD")
    reloaded = read(directory / "independent-validation.json")["TP"]["best"]
    for name, value in verified["metrics"]["best"].items():
        check(math.isclose(value, reloaded[name], rel_tol=1e-4, abs_tol=1e-5), f"{run}: reload differs for {name}")
    return {"epochs": 30, "optimizer_steps": verified["steps"], "coco": coco_scores(reloaded),
            "initialization": cfg["init_sha"], "musgd_revision": cfg.get("musgd_revision", 1),
            "checkpoint": checkpoint(tp / "best.npz", verified["best_sha"], repaired)}


def lvis(run, engine, row, reference=None):
    root = run / "lvis"
    path = root / "inputs/manifest.json"
    manifest, info = read(path), read(root / "full" / (engine + "-metrics.json"))
    protocol = manifest["protocol"]
    check(info["manifest_sha"] == sha(path) and info["best_sha"] == row["checkpoint"]["sha256"]
          and info["engine"] == engine and info["protocol"] == protocol
          and info["images"] == 4809 and info["metric_scale"] == "0-100", f"{run}: LVIS/COCO provenance mismatch")
    expected = dict(size=640, batch=1, classes=1203, head="one2one", nms=False, confidence=0.001,
                    max_det=1000, detections_per_category=10000, metrics="LVIS Fixed AP bbox")
    check(all(protocol[k] == v for k,v in expected.items()), f"{run}: unexpected LVIS protocol")
    check(len(manifest["names"]) == 1203 and len(manifest["images"]) == 4809
          and len({r[0] for r in manifest["images"]}) == 4809 and manifest["training_overlap"] == 0,
          f"{run}: invalid LVIS image or category list")
    if reference is not None:
        for key in ("images", "names", "asset_sha256", "protocol"):
            check(manifest[key] == reference[key], f"{run}: LVIS {key} differs from PyTorch")
    scores = info["metrics"]
    row["lvis"] = {k:metric(scores[k], 100) for k in ("AP", "AP50", "AP75", "APr", "APc", "APf")}
    row["lvis_provenance"] = {"manifest_sha256": info["manifest_sha"], "predictions_sha256": info["prediction_sha"],
                              "text_sha256": info.get("text_sha", manifest["text_sha"])}
    return manifest


def compare(base, run, previous):
    check(run.resolve() != previous.resolve(), "Repaired and previous run directories are identical")
    pt = pytorch_baseline(base)
    manifest = lvis(base, "pytorch", pt)
    fixed = jittor_run(run, True)
    lvis(run, "jittor", fixed, manifest)
    rows = {"original_pytorch": pt, "repaired_native_jittor": fixed}
    missing = []
    if all(p.is_file() for p in (previous/"jittor/full/TP/verified.json", previous/"jittor/full/independent-validation.json")):
        old = jittor_run(previous, False)
        check(old["initialization"] == fixed["initialization"], "Standalone Jittor initializations differ")
        rows["previous_native_jittor"] = old
        if (previous/"lvis/full/jittor-metrics.json").is_file():
            lvis(previous, "jittor", old, manifest)
        else:
            missing.append("previous_native_jittor.lvis")
    else:
        missing.extend(["previous_native_jittor.coco", "previous_native_jittor.lvis"])
    deltas = {}
    for baseline, row in rows.items():
        if row is fixed:
            continue
        deltas[baseline] = {test: {k: value-row[test][k] for k,value in fixed[test].items()}
                            for test in ("coco", "lvis") if test in row}
    return {"metric_unit": "AP points, 0–100", "training_images": 20000, "coco_validation_images": 5000,
            "lvis_images": 4809, "runs": rows, "repaired_minus_baseline": deltas, "missing_optional": missing}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--previous-run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.base, args.run, args.previous_run or args.base/"native/coco20k")
    output = args.output or args.run/"tp-comparison.json"
    tmp = output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(output)
    print(output, json.dumps(result["repaired_minus_baseline"]))
