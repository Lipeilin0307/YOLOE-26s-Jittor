"""Measure held-out AP after recalibrating BN at fixed weights, using only the 200 training images."""
import argparse
import json
from pathlib import Path

import numpy as np

from yoloe11.training_replay import batches


def run(args):
    import torch
    from yoloe11.train_reference import read_model, validate
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    raw = torch.from_numpy(np.load(args.init / "raw-text.npy"))
    with np.load(args.init / "det.npz") as source:
        keys = {k for k in source if not k.endswith(("num_batches_tracked", "running_mean", "running_var"))}
    result = {}
    for name in ("initial", "pytorch", "jittor"):
        model, _ = read_model(args.assets, keys)
        if name == "pytorch":
            state = torch.load(args.pytorch, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=False)
        elif name == "jittor":
            with np.load(args.jittor) as state:
                for k, v in model.state_dict().items():
                    if k in state:
                        v.copy_(torch.from_numpy(state[k]).reshape_as(v))
        original = validate(model, raw, args.replay)
        model.eval()
        for layer in model.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.reset_running_stats()
                layer.momentum = None
                layer.train()
        for batch in batches(args.replay, "train"):
            prompts = raw[0][torch.from_numpy(batch["text_ids"]).long()]
            model.predict(torch.from_numpy(batch["img"]).float() / 255, tpe=prompts)
        calibrated = validate(model, raw, args.replay)
        result[name] = dict(original=original, recalibrated=calibrated)
        print(name, original, calibrated, flush=True)
    args.out.write_text(json.dumps(result, indent=2)+"\n")


def frozen_native(args):
    import jittor as jt
    from native_metrics import Evaluator
    from yoloe11.detector import YOLOE11TP, image_tensor, suppress
    jt.flags.use_cuda = 0
    model = YOLOE11TP(args.init)
    model.training = True
    with jt.no_grad():
        for batch in batches(args.replay, "train"):
            x = image_tensor(batch["img"])
            outputs = model.forward(x, model.raw[0][jt.array(batch["text_ids"])])
            jt.sync(list(outputs.values()))
            jt.sync_all()
        model.training = False
        evaluator = Evaluator(single=False, segment=False)
        for batch in batches(args.replay, "val"):
            x = image_tensor(batch["img"])
            outputs = model.forward(x, model.raw)
            evaluator.update(batch, suppress(*model.decoded(outputs, *x.shape[-2:])))
    result = evaluator.result()
    if result["seen"] != 200:
        raise ValueError("Incomplete validation")
    args.out.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("init", "replay", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("assets", "pytorch", "jittor"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--native-frozen", action="store_true")
    args = parser.parse_args()
    if not args.native_frozen and not all((args.assets, args.pytorch, args.jittor)):
        parser.error("PyTorch recalibration requires --assets, --pytorch and --jittor")
    (frozen_native if args.native_frozen else run)(args)
