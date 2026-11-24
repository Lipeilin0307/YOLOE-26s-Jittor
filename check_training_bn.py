"""Compare local 26s training BN, its backward pass and running statistics on the same stem activation."""
import argparse
import json
from pathlib import Path

import numpy as np


KEY = "model.0.bn"
BUFFERS = ("weight", "bias", "running_mean", "running_var")


def reference(args):
    import torch
    from torch.nn import functional as F

    torch.set_num_threads(2)
    spec = json.loads((args.init / "det.json").read_text())
    with np.load(args.batch) as batch, np.load(args.init / "det.npz") as weights:
        image = torch.from_numpy(batch["img"]).float() / 255
        conv = spec["model.0.conv"]
        x = F.conv2d(image, torch.from_numpy(weights["model.0.conv.weight"]),
                     stride=conv["stride"], padding=conv["padding"]).requires_grad_()
        initial = {name: weights[KEY + "." + name] for name in BUFFERS}
    weight, bias, mean, var = (torch.from_numpy(initial[k].copy()) for k in BUFFERS)
    weight.requires_grad_()
    bias.requires_grad_()
    momentum, eps = (spec[KEY][k] for k in ("momentum", "eps"))
    y = F.batch_norm(x, mean, var, weight, bias, True, momentum, eps)
    dy = np.random.default_rng(26).standard_normal(tuple(x.shape), dtype=np.float32)
    y.backward(torch.from_numpy(dy))
    np.savez(args.fixture, x=x.detach().numpy(), dy=dy, y=y.detach().numpy(), dx=x.grad.numpy(),
             dw=weight.grad.numpy(), db=bias.grad.numpy(), mean=mean.numpy(), var=var.numpy(),
             momentum=momentum, eps=eps, **initial)


def check(args):
    import jittor as jt
    from model import YOLOE26

    jt.flags.use_cuda = 0
    model = YOLOE26.__new__(YOLOE26)
    model.stage = "TP"
    model.training = True
    with np.load(args.fixture) as data:
        model.spec = {KEY: {k: float(data[k]) for k in ("momentum", "eps")}}
        model.state = {KEY + "." + k: jt.array(data[k]).stop_grad() for k in BUFFERS}
        w, b = (model.state[KEY + "." + k].start_grad() for k in ("weight", "bias"))
        x = jt.array(data["x"]).start_grad()
        y = model.bn(KEY, x)
        dx, dw, db = jt.grad((y * jt.array(data["dy"]).stop_grad()).sum(), [x, w, b])
        values = dict(y=y, dx=dx, dw=dw, db=db, mean=model.state[KEY + ".running_mean"],
                      var=model.state[KEY + ".running_var"])
        errors = {}
        for key, value in values.items():
            native, target = value.numpy().astype(np.float64), data[key].astype(np.float64)
            reference_delta = target - data["running_" + key] if key in ("mean", "var") else target
            errors[key] = float(np.linalg.norm(native - target) / max(np.linalg.norm(reference_delta), 1e-20))
        print(json.dumps(errors, indent=2), flush=True)
        limits = dict(y=1e-6, dx=2e-6, dw=1e-5, db=1e-5, mean=2e-5, var=2e-5)
        for key, tolerance in limits.items():
            if not np.isfinite(errors[key]) or errors[key] > tolerance:
                raise AssertionError(f"{key} relative L2 {errors[key]:.8g} > {tolerance}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="engine", required=True)
    ref = commands.add_parser("torch")
    ref.add_argument("--init", type=Path, required=True)
    ref.add_argument("--batch", type=Path, required=True, help="NPZ containing the NCHW uint8 training batch in img")
    for command in (ref, commands.add_parser("jittor")):
        command.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    (reference if args.engine == "torch" else check)(args)
