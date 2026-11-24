"""Replay all 50 updates and compare complete, layer-grouped gradients, updates and BN buffers."""
import argparse
import json
from pathlib import Path

import numpy as np

from yoloe11.training_replay import batches


SHARED_STEPS = (1, 2, 3, 4, 5, 10, 20, 50)


def restore(flat, keys, tensors, jt):
    offset = 0
    for key in keys:
        tensor = tensors[key]
        end = offset + tensor.numel()
        tensor.update(jt.array(flat[offset:end].reshape(tuple(tensor.shape))))
        offset = end
    assert offset == len(flat)


def concat(arrays):
    return np.concatenate([a.reshape(-1) for a in arrays])


def groups(names, params):
    result = {}
    pos = 0
    for key in names:
        group = ".".join(key.split(".")[:2])
        count = params[key].numel()
        result.setdefault(group, []).append(slice(pos, pos+count))
        pos += count
    return result


def compare(a, b, layout):
    result = {}
    total = np.zeros(3)
    for name, slices in layout.items():
        values = np.zeros(3)
        for span in slices:
            x, y = a[span].astype("float64"), b[span].astype("float64")
            d = x-y
            values += [d@d, y@y, x@x]
        total += values
        result[name] = values
    result["all"] = total
    return {k:dict(rel=float(np.sqrt(v[0]/max(v[1],1e-30))),
                   abs=float(np.sqrt(v[0])), ref_norm=float(np.sqrt(v[1]))) for k,v in result.items()}


def run(args):
    ref = args.engine == "reference"
    with np.load(args.init / "det.npz") as init:
        names = sorted(k for k in init if not k.endswith(("running_mean","running_var","num_batches_tracked")))
        bn_names = sorted(k for k in init if k.endswith(("running_mean","running_var")))
    if ref:
        import torch
        from ultralytics.utils.loss import v8DetectionLoss
        from yoloe11.train_reference import read_model
        torch.set_num_threads(4)
        torch.manual_seed(20261121)
        model, params = read_model(args.assets, set(names))
        model.train()
        buffers = dict(model.named_buffers())
        raw = torch.from_numpy(np.load(args.init / "raw-text.npy"))
        optimizer = torch.optim.SGD(params.values(), lr=.0005, momentum=.9)
        criterion = v8DetectionLoss(model)
        def array(v):
            return v.detach().numpy()
    else:
        import jittor as jt
        from yoloe11.detector import YOLOE11TP, image_tensor
        jt.flags.use_cuda = 0
        jt.set_global_seed(20261121)
        model = YOLOE11TP(args.init)
        model.training = True
        if args.legacy_bn:
            from model import YOLOE26
            model.bn = YOLOE26.bn.__get__(model)
        params, buffers = model.trainable, model.state
        optimizer = jt.optim.SGD(list(params.values()), lr=.0005, momentum=.9)
        def array(v):
            return v.numpy()
    layout = groups(names, params)
    bn_layout = groups(bn_names, buffers)
    def collect(named, keys):
        return concat([array(named[k]) for k in keys])
    previous, previous_bn = collect(params, names), collect(buffers, bn_names)
    records = []
    for step, batch in enumerate(batches(args.replay, "train"), 1):
        if args.shared_start and step in SHARED_STEPS:
            start = args.cache / f"start-{step:02}.npz"
            if ref:
                velocity = (concat([array(optimizer.state[params[k]]["momentum_buffer"]) for k in names])
                            if step > 1 else np.zeros_like(previous))
                np.savez(start, params=previous, buffers=previous_bn, velocity=velocity)
            else:
                with np.load(start) as values:
                    previous, previous_bn = values["params"], values["buffers"]
                    restore(previous, names, params, jt)
                    restore(previous_bn, bn_names, buffers, jt)
                    velocities = dict(zip(params, optimizer.param_groups[0]["values"]))
                    restore(values["velocity"], names, velocities, jt)
                jt.sync_all()
        elif args.shared_start and not ref:
            continue
        if ref:
            optimizer.zero_grad(set_to_none=True)
            labels = {k:torch.from_numpy(batch[k]) for k in ("cls","bboxes","batch_idx")}
            outputs = model.predict(torch.from_numpy(batch["img"]).float()/255,
                                    tpe=raw[0][torch.from_numpy(batch["text_ids"]).long()])
            loss, terms = criterion(outputs, labels)
            loss.sum().backward()
            grad = concat([array(params[k].grad) for k in names])
            torch.nn.utils.clip_grad_norm_(params.values(), 10.)
            optimizer.step()
        else:
            image = jt.array(batch["img"]).float32()/255 if args.legacy_input else image_tensor(batch["img"])
            outputs = model.forward(image,
                                    model.raw[0][jt.array(batch["text_ids"])])
            loss, terms = model.loss(outputs, batch)
            terms = {k:v.item() for k,v in terms.items()}
            optimizer.backward(loss)
            if args.reference_gradients:
                with np.load(args.cache / f"step-{step:02}.npz") as source:
                    by_name = {k:optimizer.find_grad(params[k]) for k in names}
                    restore(source["gradient"], names, by_name, jt)
            grad = concat([array(optimizer.find_grad(params[k])) for k in names])
            if args.stable_clip:
                from yoloe11.train import clip_gradients
                clip_gradients(optimizer)
            else:
                optimizer.clip_grad_norm(10.)
            if args.reference_gradients:
                postclip = concat([array(optimizer.find_grad(params[k])) for k in names])
                postclip_norm = float(np.linalg.norm(postclip.astype(np.float64)))
            optimizer.step()
            jt.sync_all()
        current, bn = collect(params, names), collect(buffers, bn_names)
        arrays = dict(gradient=grad, update=current-previous, bn=bn, bn_update=bn-previous_bn)
        previous, previous_bn = current, bn
        path = args.cache / f"step-{step:02}.npz"
        row = dict(step=step, loss={k.removesuffix("_loss"):float(v) for k,v in terms.items()})
        if args.reference_gradients:
            row["postclip_norm"] = postclip_norm
        if ref:
            if args.shared_start and path.exists():
                with np.load(path) as saved:
                    for key in arrays:
                        np.testing.assert_array_equal(arrays[key], saved[key])
            else:
                np.savez(path, **arrays)
        else:
            with np.load(path) as data:
                for key, a in arrays.items():
                    row[key] = compare(a, data[key], bn_layout if key.startswith("bn") else layout)
        records.append(row)
        if step in (1, 2) or step % 10 == 0:
            print(step, row["loss"], flush=True)
    if len(records) != (len(SHARED_STEPS) if args.shared_start and not ref else 50):
        raise ValueError("Incomplete replay")
    args.report.write_text(json.dumps(records, separators=(",",":"))+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=("reference","jittor"))
    parser.add_argument("--shared-start", action="store_true",
                        help="Save/restore official parameters, BN buffers and SGD momentum at selected steps")
    parser.add_argument("--legacy-bn", action="store_true", help="Diagnose the pre-fix native BN")
    parser.add_argument("--legacy-input", action="store_true", help="Use the original float32 image division")
    clipping = parser.add_mutually_exclusive_group()
    clipping.add_argument("--stable-clip", dest="stable_clip", action="store_true", help="Use the corrected CPU clipping (default)")
    clipping.add_argument("--legacy-clip", dest="stable_clip", action="store_false", help="Replay the original float32 global norm")
    parser.set_defaults(stable_clip=True)
    parser.add_argument("--reference-gradients", action="store_true",
                        help="Isolate the optimizer with exact official gradients at shared-start steps")
    for key in ("assets","init","replay","cache","report"):
        parser.add_argument("--"+key, required=key!="assets", type=Path)
    args = parser.parse_args()
    if args.reference_gradients and (not args.shared_start or args.engine!="jittor"):
        parser.error("--reference-gradients requires jittor --shared-start")
    args.cache.mkdir(exist_ok=True, parents=True)
    run(args)
