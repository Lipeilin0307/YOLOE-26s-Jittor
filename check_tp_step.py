"""Replay one complete 26s TP minibatch with the official and native trainers on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


def load_batch(path):
    with np.load(path) as source:
        idx = source["batch_idx"].reshape(-1) < 2
        batch = {key: source[key][idx] for key in ("cls", "bboxes", "batch_idx")}
        # The paired image resize is done once with integer 2x2 area averaging.
        pixels = source["img"][:2]
        b, c, h, w = pixels.shape
        batch["img"] = ((pixels.reshape(b,c,h//2,2,w//2,2).sum((3,5)) + 2) // 4).astype(np.uint8)
        batch["text_ids"] = source["text_ids"][:2]
    return batch


def reference(args, batch):
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.nn.tasks import YOLOEModel
    from ultralytics.utils.loss import E2ELoss

    torch.set_num_threads(args.threads)
    model = YOLOEModel("yoloe-26s.yaml", nc=80, verbose=False)
    with np.load(args.init / "det.npz") as initial:
        model.load_state_dict({k:torch.from_numpy(initial[k]) for k in initial})
    model.args = get_cfg(overrides={"epochs":1, "overlap_mask":False})
    BaseTrainer._model_train(SimpleNamespace(model=model, freeze_layer_names=["model.23.savpe."]))
    for key, param in model.named_parameters():
        param.requires_grad_(".savpe." not in key)
    params = {k:v for k,v in model.named_parameters() if v.requires_grad}
    opt = BaseTrainer.build_optimizer(SimpleNamespace(), model, name="MuSGD", lr=.00125, momentum=.9, decay=.0007)
    raw = torch.from_numpy(np.load(args.init / "raw-text.npy"))
    texts = raw[0][torch.from_numpy(batch["text_ids"]).long()]
    outputs = model.predict(torch.from_numpy(batch["img"]).float()/255, tpe=texts)
    data = {k:torch.from_numpy(batch[k]) for k in ("cls","bboxes","batch_idx")}
    loss, terms = E2ELoss(model)(outputs, data)
    loss = loss.sum()
    loss.backward()
    values = {"loss":np.array(loss.item()), **{k:np.array(v.item()) for k,v in terms.items()}}
    for branch, rec in outputs.items():
        for field in ("scores","boxes"):
            values[f"{branch}.{field}"] = rec[field].detach().numpy()
    for key, param in params.items():
        if param.grad is None:
            raise ValueError(f"Unconnected parameter {key}")
        values["grad."+key] = param.grad.detach().numpy().copy()
    torch.nn.utils.clip_grad_norm_(params.values(), 10.)
    opt.step()
    with np.load(args.init / "det.npz") as initial:
        for key, param in params.items():
            values["update."+key] = param.detach().numpy() - initial[key]
        for key, buf in model.named_buffers():
            if key.endswith(("running_mean","running_var")):
                values["bn."+key] = buf.numpy() - initial[key]
    np.savez(args.fixture, **values)


class Comparison:
    def __init__(self, fixture):
        self.fixture = fixture
        self.groups = {}
        self.checked = set()

    def add(self, key, native, *groups):
        ref = self.fixture[key].astype(np.float64).ravel()
        x = np.asarray(native, dtype=np.float64).ravel()
        if x.shape != ref.shape or not np.isfinite(x).all() or not np.isfinite(ref).all():
            raise AssertionError(key)
        self.checked.add(key)
        d = x - ref
        for group in groups:
            out = self.groups.setdefault(group, [0., 0., 0., 0])
            out[0] += d.dot(d)
            out[1] += ref.dot(ref)
            out[2] = max(out[2], float(np.abs(d).max()))
            out[3] += 1

    def summary(self, expected=None):
        expected = set(self.fixture.files) if expected is None else expected
        if self.checked != expected:
            raise AssertionError(sorted(self.checked ^ expected))
        return {group:dict(relative_L2=float(np.sqrt(diff/ref)) if ref else None,
                           max_abs=maximum, reference_L2=float(np.sqrt(ref)), tensors=count)
                for group,(diff,ref,maximum,count) in self.groups.items()}


def area(key):
    index = int(key.split(".")[1])
    if index < 11:
        return "backbone"
    return "neck" if index < 23 else key.split(".")[2]


def fixed_gradients(args, batch):
    """Replace only the backward result with the official unclipped gradient."""
    import jittor as jt
    from model import YOLOE26
    from optimizer import Optimizer

    jt.flags.use_cuda = 0
    model = YOLOE26(args.init, "TP")
    opt = Optimizer(model, 1, len(batch["img"]), 1, smoke=True)
    with np.load(args.fixture) as fixture, np.load(args.init / "det.npz") as initial:
        compare = Comparison(fixture)
        grads = [jt.array(fixture["grad."+k]).reshape(v.shape).stop_grad() for k,v in model.trainable.items()]
        with patch.object(jt, "grad", return_value=grads):
            opt.backward(None, 0, 0)
        assert opt.steps == 1
        for key, param in model.trainable.items():
            compare.add("update."+key, param.numpy()-initial[key], "update", "update."+area(key))
        summary = compare.summary({k for k in fixture if k.startswith("update.")})
        report = json.loads(args.report.read_text())
        report["official_gradient_control"] = summary
        args.report.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(summary["update"]), flush=True)


def native(args, batch):
    import jittor as jt
    from model import YOLOE26
    from objective import objective
    from optimizer import Optimizer
    from train import image_tensor

    jt.flags.use_cuda = 0
    model = YOLOE26(args.init, "TP")
    model.training = True
    opt = Optimizer(model, 1, len(batch["img"]), 1, smoke=True)
    with np.load(args.fixture) as fixture, np.load(args.init / "det.npz") as initial:
        compare = Comparison(fixture)
        outputs = model.forward(image_tensor(batch), raw=model.raw[0][jt.array(batch["text_ids"])])
        loss, terms = objective(outputs, batch, "TP", 0, 1)
        losses = {}
        for k,v in {"loss":loss, **{key+"_loss":v for key,v in terms.items()}}.items():
            losses[k] = {"pytorch":float(fixture[k]), "jittor":float(v.item())}
            compare.add(k, v.item(), k)
        for branch, rec in outputs.items():
            for field in ("scores","boxes"):
                key = f"{branch}.{field}"
                compare.add(key, rec[field].numpy(), key)
        original_grad = jt.grad

        def capture_grads(*params, **kwargs):
            grads = original_grad(*params, **kwargs)
            jt.sync(grads)
            for key, grad in zip(model.trainable, grads):
                compare.add("grad."+key, grad.numpy(), "gradient", "gradient."+area(key))
            return grads

        with patch.object(jt, "grad", capture_grads):
            opt.backward(loss, 0, 0)
        assert opt.steps == 1
        for key, param in model.trainable.items():
            compare.add("update."+key, param.numpy()-initial[key], "update", "update."+area(key))
        for key, buf in model.state.items():
            if key.endswith(("running_mean","running_var")):
                compare.add("bn."+key, buf.numpy()-initial[key], "bn_increment", "bn_increment."+key.rsplit(".",1)[1])
        report = dict(model="YOLOE-26s, TP, local WSL CPU, one instrumented update",
                      batch_sha256=hashlib.sha256(args.batch.read_bytes()).hexdigest(),
                      init_sha256=hashlib.sha256((args.init/"det.npz").read_bytes()).hexdigest(),
                      input_shape=list(batch["img"].shape), objects=len(batch["cls"]),
                      alignment="Same initialization, real RGB pixels, labels, per-image prompt order, FP32, "
                                "dual-head E2E loss, clip at 10 and MuSGD smoke settings with no warmup. "
                                "Native gradients are materialized before the optimizer update.",
                      losses=losses, comparison=compare.summary())
        args.report.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps({k:report["comparison"][k] for k in ("gradient","update","bn_increment")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=["torch","jittor","fixed-gradients"])
    for key in ("init","batch","fixture"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.engine != "torch" and not args.report:
        parser.error("--report is required for Jittor runs")
    {"torch":reference, "jittor":native, "fixed-gradients":fixed_gradients}[args.engine](args, load_batch(args.batch))
