"""One-time, CPU-only conversion of the baseline initialization and frozen text features."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics.nn.tasks import YOLOEModel, YOLOESegModel, load_checkpoint
from ultralytics.utils.torch_utils import init_seeds


def graph(model):
    result = {}
    fields = ("stride", "padding", "output_padding", "dilation", "groups", "kernel_size",
              "momentum", "eps", "add", "n", "num_heads", "head_dim", "key_dim", "scale",
              "scale_factor", "mode", "c", "d", "f")
    for path, m in model.named_modules(remove_duplicate=False):
        attrs = {"kind": type(m).__name__, "children": list(m._modules)}
        for key in fields:
            value = getattr(m, key, None)
            if isinstance(value, (int, float, bool, str, list, tuple)):
                attrs[key] = value
        result[path] = attrs
    return result


def run(args):
    out, assets, data = (p.resolve() for p in (args.out, args.assets, args.data))
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(assets)
    names = yaml.safe_load((data / "detect.yaml").read_text())["names"]
    torch.set_num_threads(2)
    weights, _ = load_checkpoint(assets / "yolo26s-objv1-150.pt")
    for key, cls, cfg in (("det", YOLOEModel, "yoloe-26s.yaml"), ("seg", YOLOESegModel, "yoloe-26s-seg.yaml")):
        init_seeds(20261121, deterministic=True)
        model = cls(cfg, nc=80, verbose=False)
        if key == "det":
            model.load(weights)
        model.eval()
        (out / (key + ".json")).write_text(json.dumps(graph(model)))
        np.savez(out / (key + ".npz"), **{k: v.detach().numpy() for k, v in model.state_dict().items()})
        if key == "det":
            raw = model.get_text_pe(list(names) + ["object"], without_reprta=True).cpu()
            np.save(out / "raw-text.npy", raw.numpy())
        else:
            raw = torch.from_numpy(np.load(out / "raw-text.npy"))
        # Independent fixed-input oracle for both heads and for the visual-prompt encoder.
        x = torch.from_numpy(np.random.default_rng(79).random((2, 3, 96, 96), dtype=np.float32))
        vp = torch.zeros(2, 2, 12, 12)
        vp[:, 0, 1:8, 2:7] = 1
        vp[:, 1, 8:, 5:] = 1
        with torch.no_grad():
            head = model.model[-1]
            head.training = True  # raw dual-head outputs, BN remains in evaluation mode
            p = model.predict(x, tpe=raw[:, :3])
            fixture = {"image": x, "visual": vp, "tpe": head.get_tpe(raw[:, :3]),
                       "vpe": model.get_visual_pe(x, vp), "text": raw[:, :3]}
            for branch, rec in p.items():
                for field in ("boxes", "scores", "mask_coefficient", "proto"):
                    if field in rec:
                        fixture[f"{branch}.{field}"] = rec[field]
            np.savez(out / (key + "-oracle.npz"), **{k: v.numpy() for k,v in fixture.items()})
    (out / "names.json").write_text(json.dumps(names))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for field in ("assets", "data", "out"):
        parser.add_argument("--"+field, required=True, type=Path)
    run(parser.parse_args())
