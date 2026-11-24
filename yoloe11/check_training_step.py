"""Compare the first TP batch, its gradients, and an identical-head-output loss probe."""
import argparse
import json
from pathlib import Path

import numpy as np

from yoloe11.training_replay import batches


def difference(actual, expected):
    a = np.asarray(actual).astype(np.float64).ravel()
    b = np.asarray(expected).astype(np.float64).ravel()
    return np.array([a @ a, b @ b, a @ b, (a-b) @ (a-b)])


def summary(sums):
    aa, bb, ab, err = sums
    return dict(native_l2=float(np.sqrt(aa)), ref_l2=float(np.sqrt(bb)),
                relative_l2=float(np.sqrt(err / max(bb, 1e-30))),
                cosine=float(ab / max(np.sqrt(aa*bb), 1e-30)))


def reference(args, batch):
    import torch
    from ultralytics.utils.loss import v8DetectionLoss
    from yoloe11.train_reference import read_model
    torch.set_num_threads(4)
    with np.load(args.init / "det.npz") as det:
        names = {k for k in det if not k.endswith(("num_batches_tracked", "running_var", "running_mean"))}
    model, params = read_model(args.assets, names)
    model.train()
    raw = torch.from_numpy(np.load(args.init / "raw-text.npy"))
    prompts = raw[0][torch.from_numpy(batch["text_ids"]).long()]
    pred = model.predict(torch.from_numpy(batch["img"]).float() / 255, tpe=prompts)
    labels = {k: torch.from_numpy(batch[k]) for k in ("cls", "bboxes", "batch_idx")}
    for k in ("boxes", "scores"):
        pred[k].retain_grad()
    loss, terms = v8DetectionLoss(model)(pred, labels)
    loss.sum().backward()
    output = {k: v.grad.numpy() for k,v in params.items()}
    for k in ("boxes", "scores"):
        output[k] = pred[k].detach().numpy()
        output["grad_" + k] = pred[k].grad.numpy()
    for k, v in terms.items():
        output[k] = v.item()
    for k, v in model.named_buffers():
        if k.endswith(("running_mean", "running_var")) and k.rsplit(".", 1)[0]+".weight" in params:
            output[k] = v.numpy()
    np.savez(args.fixture, **output)


def native(args, batch):
    import jittor as jt
    from yoloe11.detector import YOLOE11TP, image_tensor
    jt.flags.use_cuda = 0
    model = YOLOE11TP(args.init)
    model.training = True
    raw = model.raw[0][jt.array(batch["text_ids"])]
    pred = model.forward(image_tensor(batch["img"]), raw)
    loss, terms = model.loss(pred, batch)
    named = model.trainable
    groups = {}
    with np.load(args.fixture) as ref:
        forward = {k:summary(difference(pred[k].numpy(), ref[k])) for k in pred}
        grads = jt.grad(loss, list(named.values()))
        jt.sync(grads)
        for (key, _), grad in zip(named.items(), grads):
            group = key.split(".")[2] if key.startswith("model.23.") else "backbone_neck"
            groups[group] = groups.get(group, np.zeros(4)) + difference(grad.numpy(), ref[key])
        stats = {}
        for k, v in model.state.items():
            if k.endswith(("running_var", "running_mean")):
                group = k.split(".")[-1]
                stats[group] = stats.get(group, np.zeros(4)) + difference(v.numpy(), ref[k])
        forced = {k:jt.array(ref[k]).start_grad() for k in pred}
        fixed_loss, fixed_terms = model.loss(forced, batch)
        fixed_grads = jt.grad(fixed_loss, list(forced.values()))
        report = dict(first_batch_output=forward, first_batch_gradients={k:summary(v) for k,v in groups.items()},
                      global_gradient=summary(sum(groups.values())), running_stats_first_step={k:summary(v) for k,v in stats.items()},
                      native_loss={k:float(v.item()) for k,v in terms.items()},
                      identical_head_outputs=dict(reference_loss={k:float(ref[k+"_loss"]) for k in terms},
                          native_loss={k:float(v.item()) for k,v in fixed_terms.items()},
                          gradients={k:summary(difference(g.numpy(),ref["grad_"+k])) for k,g in zip(forced,fixed_grads)}))
    args.report.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("engine", choices=("reference", "jittor"))
    for name in ("init", "replay", "fixture"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--assets", type=Path)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    (reference if args.engine=="reference" else native)(args, next(batches(args.replay, "train")))
