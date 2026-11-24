"""Reopen all stage artifacts in a fresh process and evaluate the saved models."""
import argparse
import json
import math
import os
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.yoloe.val import YOLOEDetectValidator, YOLOESegValidator
from ultralytics.nn.tasks import load_checkpoint

from train import STAGES, sha256


def evaluate(args):
    data, out, assets = (p.resolve() for p in (args.data, args.out, args.assets))
    os.chdir(assets)
    split = json.loads((data / "split.json").read_text())
    summary = dict(ultralytics=ultralytics.__version__, torch=torch.__version__,
                   device=(torch.cuda.get_device_name(int(args.device)) if args.device != "cpu" else "CPU"),
                   split=split, stages={})
    n_val = split["splits"]["val"]["images"]
    for stage in STAGES:
        run = out / stage
        record = json.loads((run / "verified.json").read_text())
        best = run / "weights/best.pt"
        parent = assets / "yolo26s-objv1-150.pt" if stage == "TP" else out / "TP/weights/best.pt"
        if sha256(best) != record["best_sha256"] or sha256(parent) != record["input_sha256"]:
            raise ValueError(f"Checkpoint identity mismatch: {stage}")
        last, _ = load_checkpoint(run / "weights/last.pt", device="cpu")
        if not all(p.isfinite().all() for p in last.parameters()):
            raise FloatingPointError(f"Invalid last.pt: {stage}")
        del last
        imgsz = yaml.safe_load((run / "args.yaml").read_text())["imgsz"]
        variants = [("best", best)]
        if stage == "PF":
            vocab = run / "weights/prompt-free-coco80.pt"
            if sha256(vocab) != record["vocabulary_sha256"]:
                raise ValueError("PF vocabulary checkpoint identity mismatch")
            variants.append(("coco80", vocab))
        checks = {}
        for variant, path in variants:
            task = "segment" if stage == "SEG" else "detect"
            cls = (DetectionValidator if stage == "PF" else
                   YOLOESegValidator if stage == "SEG" else YOLOEDetectValidator)
            validator = cls(save_dir=run, args=dict(data=str(data / f"{task}.yaml"), model=str(path),
                task=task, imgsz=imgsz, batch=args.batch, workers=0, device=args.device, nms=False,
                rect=False, verbose=False, plots=False, save_json=False, single_cls=stage == "PF" and variant == "best"))
            kwargs = {"load_vp": True, "refer_data": str(data / "reference.yaml")} if stage == "VP" else {}
            result = validator(model=str(path), **kwargs)
            metrics = {k: float(v) for k, v in result.items() if k.startswith("metrics/")}
            if validator.seen != n_val or not metrics or not all(map(math.isfinite, metrics.values())):
                raise RuntimeError(f"Incomplete saved-model validation: {stage}/{variant}")
            checks[variant] = dict(weights=str(path.relative_to(out)), images=validator.seen, metrics=metrics)
        summary["stages"][stage] = dict(epochs=record["epochs"], optimizer_steps=record["optimizer_steps"],
            changed_tensors=len(record["changed_parameters"]), train_seconds_including_validation=record["elapsed_seconds"],
            train_losses=record["train_losses"], initial_sha256=record["input_sha256"],
            best_sha256=record["best_sha256"], standalone_validation=checks)
    path = out / "summary.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(tmp, path)
    print(f"Validated all four stages. Summary at {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "assets", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    evaluate(parser.parse_args())
