"""Compare the native validator to saved legacy-worker metrics with identical weights/images."""
import argparse
import importlib.util
import json
from pathlib import Path

import jittor as jt
import numpy as np

from model import YOLOE26
from native_data import NativeWorker
from train import STAGES, digest, evaluate, save_json, stage_config


def audit(args):
    dependencies = {key: importlib.util.find_spec(key) is not None for key in ("torch","torchvision","ultralytics")}
    if any(dependencies.values()):
        raise ValueError(f"Clean-environment check failed: {dependencies}")
    report = dict(packages_present=dependencies, stages={})
    for stage in args.stages:
        directory = args.out / stage
        reference = json.loads((directory / "verified.json").read_bytes())
        weights = directory / "best.npz"
        if digest(weights) != reference["best_sha"]:
            raise ValueError(f"Unverified reference checkpoint for {stage}")
        config = stage_config(args, stage)
        parent = args.out / "TP/best.npz" if stage != "TP" else None
        model = YOLOE26(args.init,stage,parent,weights)
        worker = NativeWorker(config)
        try:
            stage_report = {}
            for variant, expected in reference["metrics"].items():
                got = evaluate(model,worker,config,vocab=variant == "coco80")
                if set(got) != set(expected):
                    raise ValueError(f"Metric keys differ at {stage}/{variant}")
                stage_report[variant] = {key: dict(reference=expected[key],native=value,delta=value-expected[key])
                                         for key,value in got.items()}
            report["stages"][stage] = dict(checkpoint_sha=reference["best_sha"],images=config["nval"],
                                           metrics=stage_report)
            save_json(args.report,report)
            for variant, values in stage_report.items():
                for key, entry in values.items():
                    if not np.isclose(entry["native"], entry["reference"], atol=args.atol, rtol=0):
                        raise ValueError(f"{stage}/{variant}/{key} delta {entry['delta']} exceeds {args.atol}")
        finally:
            worker.close()
        del model
        jt.gc()
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("init","data","out","report"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--batch",type=int,default=8)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--imgsz",type=int)
    p.add_argument("--stages", choices=STAGES, nargs="+", default=STAGES)
    p.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance on the 0–1 metric scale")
    args = p.parse_args()
    jt.flags.use_cuda = 1
    audit(args)
