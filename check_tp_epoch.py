"""Train and independently reload YOLOE-26s TP on identical 200-image COCO replays."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from native_metrics import Evaluator
from yoloe11.training_replay import batches


def sha(path):
    with path.open("rb") as src:
        result = hashlib.sha256()
        for block in iter(lambda: src.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def torch_model(args):
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import YOLOEModel

    model = YOLOEModel("yoloe-26s.yaml", nc=80, verbose=False)
    with np.load(args.init / "det.npz") as src:
        model.load_state_dict({k:torch.from_numpy(src[k]) for k in src}, strict=True)
    model.args = get_cfg(overrides={"epochs":1, "overlap_mask":False})
    model.model[-1].end2end = True
    return model


def trace_loss(trace, batch, loss, terms):
    row = dict(dual_loss_per_image=float(loss.item()) / len(batch["img"]),
               **{k.removesuffix("_loss"):float(v.item()) for k,v in terms.items()})
    if not np.isfinite(list(row.values())).all():
        raise FloatingPointError(row)
    trace.append(row)
    if len(trace) in (1,2) or len(trace) % 10 == 0:
        print(len(trace), row, flush=True)


def torch_run(args, ckpt):
    import torch
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.utils.loss import E2ELoss
    from ultralytics.utils.torch_utils import ModelEMA

    torch.set_num_threads(args.threads)
    torch.manual_seed(20261121)
    model = torch_model(args)
    raw = torch.from_numpy(np.load(args.init / "raw-text.npy"))
    if args.action == "validate":
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
        model.eval()
        evaluator = Evaluator(single=False, segment=False)
        with torch.inference_mode():
            for batch in batches(args.replay, "val"):
                pred, _ = model.predict(torch.from_numpy(batch["img"]).float()/255., tpe=raw[:, :80])
                evaluator.update(batch, pred.numpy())
        return evaluator.result()
    BaseTrainer._model_train(SimpleNamespace(model=model, freeze_layer_names=["model.23.savpe."]))
    for key, value in model.named_parameters():
        value.requires_grad_(".savpe." not in key)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = BaseTrainer.build_optimizer(SimpleNamespace(), model, name="MuSGD", lr=.00125, momentum=.9, decay=.0007)
    criterion = E2ELoss(model)
    ema = ModelEMA(model)
    trace = []
    for batch in batches(args.replay, "train"):
        prompts = raw[0][torch.from_numpy(batch["text_ids"]).long()]
        outputs = model.predict(torch.from_numpy(batch["img"]).float()/255., tpe=prompts)
        labels = {k:torch.from_numpy(batch[k]) for k in ("batch_idx","cls","bboxes")}
        loss, terms = criterion(outputs, labels)
        loss = loss.sum()
        trace_loss(trace, batch, loss, terms)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10., error_if_nonfinite=True)
        opt.step()
        ema.update(model)
    if ema.updates != 50:
        raise AssertionError(ema.updates)
    torch.save(ema.ema.state_dict(), ckpt)
    return trace


def jittor_run(args, ckpt):
    import jittor as jt
    from model import YOLOE26
    from objective import detections, objective
    from optimizer import Optimizer
    from train import image_tensor, save_arrays

    jt.flags.use_cuda = 0
    jt.set_global_seed(20261121)
    if args.action == "validate":
        model = YOLOE26(args.init, "TP", checkpoint=ckpt)
        evaluator = Evaluator(single=False, segment=False)
        with jt.no_grad():
            pe = model.text(model.raw[:, :80])
            for batch in batches(args.replay, "val"):
                x = image_tensor(batch)
                out = model.forward(x, pe=pe, both=False)["one2one"]
                evaluator.update(batch, detections(out, *x.shape[-2:]))
        return evaluator.result()
    model = YOLOE26(args.init, "TP")
    model.training = True
    opt = Optimizer(model, epochs=1, batch=4, batches=50, smoke=True)
    trace = []
    for i, batch in enumerate(batches(args.replay, "train")):
        prompts = model.raw[0][jt.array(batch["text_ids"])]
        outputs = model.forward(image_tensor(batch), raw=prompts)
        loss, terms = objective(outputs, batch, "TP", epoch=0, epochs=1)
        trace_loss(trace, batch, loss, terms)
        opt.backward(loss, 0, i)
    if opt.steps != 50:
        raise AssertionError(opt.steps)
    save_arrays(ckpt, opt.ema)
    return trace


def run(args):
    meta = json.loads((args.replay / "replay.json").read_bytes())
    if any(meta[key]["images"] != 200 or meta[key]["steps"] != 50 for key in ("train","val")):
        raise ValueError("Expected exactly 200 train and validation images, batch 4")
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = args.out / ("pytorch-ema.pt" if args.engine == "pytorch" else "jittor-ema.npz")
    train_path = args.out / f"{args.engine}-train.json"
    if args.action == "validate":
        if sha(ckpt) != json.loads(train_path.read_bytes())["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed since training")
    elif ckpt.exists() or train_path.exists():
        raise ValueError(f"Choose a fresh output directory for {args.engine}")
    values = (torch_run if args.engine == "pytorch" else jittor_run)(args, ckpt)
    if args.action == "validate":
        if values["seen"] != 200 or not np.isfinite(list(values["metrics"].values())).all():
            raise ValueError("Incomplete validation")
        write(args.out / f"{args.engine}-val.json", dict(**values, checkpoint_sha256=sha(ckpt)))
        print(values, flush=True)
        return
    if len(values) != 50:
        raise ValueError("Incomplete training")
    report = dict(framework=args.engine, device="cpu", checkpoint="FP32 EMA", checkpoint_sha256=sha(ckpt),
                  init_sha256=sha(args.init/"det.npz"), raw_text_sha256=sha(args.init/"raw-text.npy"),
                  seed=20261121, size=320, batch=4, optimizer="MuSGD", lr=.00125, momentum=.9, decay=.0007,
                  clip_grad_norm=10, warmup=0, optimizer_steps=len(values), replay=meta, trace=values,
                  mean_loss={k:float(np.mean([s[k] for s in values])) for k in values[0]})
    if args.engine == "jittor":
        report["implementation_sha256"] = {f:sha(Path(__file__).with_name(f)) for f in
                                           ("model.py","objective.py","optimizer.py","train.py")}
    write(train_path, report)


def compare(directory, output):
    trained = {e:json.loads((directory / f"{e}-train.json").read_bytes()) for e in ("pytorch","jittor")}
    val = {e:json.loads((directory / f"{e}-val.json").read_bytes()) for e in trained}
    config = {}
    for key in ("init_sha256","raw_text_sha256","seed","device","size","batch","optimizer","lr","momentum",
                "decay","clip_grad_norm","warmup","optimizer_steps","checkpoint","replay"):
        if trained["pytorch"][key] != trained["jittor"][key]:
            raise ValueError(f"Paired configuration differs at {key}")
        config[key] = trained["pytorch"][key]
    for engine in trained:
        if val[engine]["checkpoint_sha256"] != trained[engine]["checkpoint_sha256"] or val[engine]["seen"] != 200:
            raise ValueError(f"Checkpoint or validation count mismatch for {engine}")

    def pair(pt, native):
        return dict(pytorch=pt, jittor=native, delta_jittor_minus_pytorch=native-pt)

    def losses(rows):
        return {k:pair(rows["pytorch"][k],rows["jittor"][k]) for k in rows["pytorch"]}

    aps = {name:pair(100*val["pytorch"]["metrics"][key],100*val["jittor"]["metrics"][key])
           for name,key in (("AP50","metrics/mAP50(B)"),("AP50-95","metrics/mAP50-95(B)"))}
    report = dict(experiment="YOLOE-26s TP, one epoch, COCO 200 train / 200 val",
                  config=config, checkpoints={e:r["checkpoint_sha256"] for e,r in trained.items()},
                  mean_loss=losses({e:r["mean_loss"] for e,r in trained.items()}),
                  first_batch_loss=losses({e:r["trace"][0] for e,r in trained.items()}),
                  bbox_AP_0_to_100=aps, validation_head="one2one", max_det=300, conf=.001,
                  validation_evaluator="native_metrics.Evaluator",
                  jittor_source_sha256=trained["jittor"]["implementation_sha256"])
    write(output, report)
    print(json.dumps(aps, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=("pytorch", "jittor", "pair"))
    parser.add_argument("action", choices=("train", "validate", "compare"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--init", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.engine == "pair" and args.action == "compare" and args.report:
        compare(args.out, args.report)
    elif args.engine in ("pytorch","jittor") and args.action != "compare" and args.init and args.replay:
        run(args)
    else:
        parser.error("Use {pytorch,jittor} {train,validate} with --init and --replay, or pair compare with --report")
