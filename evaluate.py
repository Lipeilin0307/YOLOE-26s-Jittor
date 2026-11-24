"""Validate a released COCO 20k checkpoint directly on a prepared COCO split."""
import argparse
import json
from pathlib import Path

import jittor as jt

from fetch_weights import ASSETS, PRESETS, ROOT, fetch
from native_data import NativeWorker
from train import STAGES, YOLOE26, evaluate, stage_config


def run(args):
    jt.flags.use_cuda = int(args.device == "cuda")
    for key in PRESETS[args.stage]:
        fetch(key, check=True)
    cfg = stage_config(args, args.stage)
    parent = None if args.stage == "TP" else ROOT / ASSETS["TP"]["path"]
    model = YOLOE26(ROOT / "init", args.stage, parent=parent, checkpoint=ROOT / ASSETS[args.stage]["path"])
    worker = NativeWorker(cfg)
    try:
        metrics = {"best": evaluate(model, worker, cfg)}
        if args.stage == "PF":
            metrics["coco80"] = evaluate(model, worker, cfg, vocab=True)
    finally:
        worker.close()
    return {"stage": args.stage, "images": cfg["nval"], "size": cfg["size"],
            "val_sha256": cfg["val_sha"], "weights_sha256": ASSETS[args.stage]["sha256"], "metrics": metrics}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int)
    args = p.parse_args()
    args.out = Path(".")
    print(json.dumps(run(args), indent=2, allow_nan=False))
