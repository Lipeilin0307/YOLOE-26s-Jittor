"""On the first training batch, replay individual official operators with their actual incoming gradients."""
import argparse
import json
from pathlib import Path

import numpy as np

from yoloe11.check_training_step import difference, summary
from yoloe11.training_replay import batches


BLOCKS = ("0", "1", "2.cv1", "2.m.0.cv1", "3", "4.cv1", "6.cv1", "8.cv1",
          "9.cv1", "9.cv2", "10.cv1", "13.cv1", "16.cv1", "23.cv2.0.0")
KEYS = [f"model.{block}.{op}" for block in BLOCKS for op in ("conv","bn")]


def reference(args, batch):
    import torch
    from ultralytics.utils.loss import v8DetectionLoss
    from yoloe11.train_reference import read_model
    torch.set_num_threads(4)
    with np.load(args.init/"det.npz") as src:
        names = {k for k in src if not k.endswith(("running_mean","running_var","num_batches_tracked"))}
    model, _ = read_model(args.assets, names)
    model.train()
    modules = dict(model.named_modules())
    hooks = []
    def hook(name):
        def capture(module, inputs, out):
            x = inputs[0].detach()
            np.save(args.cache/(name+".x.npy"), x.numpy())
            np.save(args.cache/(name+".y.npy"), out.detach().numpy())
            def save_grad(dy):
                np.save(args.cache/(name+".dy.npy"), dy.detach().numpy())
            out.register_hook(save_grad)
        return capture
    for key in KEYS:
        hooks.append(modules[key].register_forward_hook(hook(key)))
    raw = torch.from_numpy(np.load(args.init/"raw-text.npy"))
    x = torch.from_numpy(batch["img"]).float()/255
    pred = model.predict(x,tpe=raw[0][torch.from_numpy(batch["text_ids"]).long()])
    labels = {k:torch.from_numpy(batch[k]) for k in ("cls","bboxes","batch_idx")}
    v8DetectionLoss(model)(pred,labels)[0].sum().backward()
    for h in hooks:
        h.remove()
    for key in KEYS:
        x = torch.from_numpy(np.load(args.cache/(key+".x.npy"))).requires_grad_()
        dy = torch.from_numpy(np.load(args.cache/(key+".dy.npy")))
        dx = torch.autograd.grad(modules[key](x), x, dy)[0]
        np.save(args.cache/(key+".dx.npy"), dx.numpy())


def native(args, batch):
    import jittor as jt
    from yoloe11.detector import YOLOE11TP, image_tensor
    model = YOLOE11TP(args.init)
    model.training = True
    if args.legacy_bn:
        from model import YOLOE26
        model.bn = YOLOE26.bn.__get__(model)
    report = {}
    layer_method = model.layer
    def capture(path, x):
        out = layer_method(path, x)
        if path in KEYS:
            report[path] = dict(end_to_end_output=error(path, "y", out))
        return out
    def error(path, suffix, var):
        ref = np.load(args.cache/(path+"."+suffix+".npy"))
        return summary(difference(var.numpy(),ref))["relative_l2"]
    model.layer = capture
    image = jt.array(batch["img"]).float32()/255 if args.legacy_input else image_tensor(batch["img"])
    official = np.load(args.cache/"model.0.conv.x.npy")
    diff = image.numpy()-official
    report["input"] = dict(different=int(np.count_nonzero(diff)), max_abs=float(np.abs(diff).max()))
    model.forward(image, model.raw[0][jt.array(batch["text_ids"])])
    model.layer = layer_method
    for key in KEYS:
        x = jt.array(np.load(args.cache/(key+".x.npy"))).start_grad()
        dy = jt.array(np.load(args.cache/(key+".dy.npy")))
        y = model.layer(key,x)
        dx = jt.grad((y*dy).sum(), x)
        report[key].update(local_output=error(key,"y",y), local_input_grad=error(key,"dx",dx))
    args.report.write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("engine",choices=("reference","jittor"))
    p.add_argument("--legacy-bn",action="store_true")
    p.add_argument("--legacy-input",action="store_true")
    for key in ("init","assets","cache","replay","report"):
        p.add_argument("--"+key,required=key not in ("assets","report"),type=Path)
    args=p.parse_args()
    args.cache.mkdir(parents=True,exist_ok=True)
    (reference if args.engine=="reference" else native)(args,next(batches(args.replay,"train")))
