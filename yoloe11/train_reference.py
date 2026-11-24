"""Run the official Ultralytics 11s detection criterion on the original Jittor smoke batches."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import ultralytics
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.nms import non_max_suppression

from native_metrics import Evaluator
from yoloe11.export import check_asset, DETECT_SHA
from yoloe11.training_replay import batches


def read_model(assets, trainable_keys):
    model, _ = load_checkpoint(assets / "yoloe-11s-seg.pt", device="cpu")
    model = model.float()
    model.args = IterableSimpleNamespace(**model.args)
    model.model[-1].nc = 80
    params = {}
    for name, tensor in model.named_parameters():
        tensor.requires_grad_(name in trainable_keys)
        if tensor.requires_grad:
            params[name] = tensor
    if set(params) != trainable_keys:
        raise ValueError("Official model and Jittor trainable tensors differ")
    return model, params


@torch.inference_mode()
def validate(model, raw, replay):
    model.eval()
    evaluator = Evaluator(single=False, segment=False)
    for batch in batches(replay, "val"):
        det, _ = model.predict(torch.from_numpy(batch["img"]).float() / 255, tpe=raw)
        if isinstance(det, tuple):
            det = det[0]
        kept = non_max_suppression(det[:, :84].contiguous(), conf_thres=.001, iou_thres=.7,
                                   nc=80, multi_label=True, max_det=300)
        evaluator.update(batch, [d.numpy() for d in kept])
    result = evaluator.result()
    if result["seen"] != 200 or not np.isfinite(list(result["metrics"].values())).all():
        raise ValueError("Incomplete validation")
    return result


def run(args):
    if ultralytics.__version__ != "8.4.216":
        raise ValueError("Use the pinned Ultralytics reference")
    torch.set_num_threads(4)
    check_asset(args.assets / "yoloe-11s-seg.pt", DETECT_SHA)
    record = json.loads(args.jittor_run.read_bytes())
    torch.manual_seed(record["seed"])
    replay = json.loads((args.replay / "replay.json").read_bytes())
    if replay["train"].get("seed", 20261121) != record["seed"]:
        raise ValueError("Replay seed differs from the native run")
    if "train_arrays_sha256" in record and record["train_arrays_sha256"] != replay["train"]["arrays_sha256"]:
        raise ValueError("Augmented training tensors differ between frameworks")
    for split, values in replay.items():
        if record["config"][split + "_sha"] != values["manifest_sha256"]:
            raise ValueError(f"Wrong {split} manifest")
    check_asset(args.init / "raw-text.npy", record["config"]["raw_sha"])
    check_asset(args.init / "det.npz", record["config"]["init_sha"])
    with np.load(args.init / "det.npz") as initial:
        keys = set(initial.files)
    trainable = {k for k in keys if not k.endswith(("num_batches_tracked", "running_var", "running_mean"))}
    if len(trainable) != record["trainable_tensors"]:
        raise ValueError("Unexpected trainable parameter set")
    model, params = read_model(args.assets, trainable)
    raw = torch.from_numpy(np.load(args.init / "raw-text.npy"))
    baseline = validate(model, raw, args.replay)
    criterion = v8DetectionLoss(model)
    optimizer = torch.optim.SGD(params.values(), lr=record["lr"], momentum=.9)
    model.train()
    trace = []
    for step, batch in enumerate(batches(args.replay, "train"), 1):
        labels = {k: torch.from_numpy(batch[k]) for k in ("batch_idx", "cls", "bboxes")}
        prompts = raw[0][torch.from_numpy(batch["text_ids"]).long()]
        pred = model.predict(torch.from_numpy(batch["img"]).float() / 255, tpe=prompts)
        losses, components = criterion(pred, labels)
        optimizer.zero_grad(set_to_none=True)
        losses.sum().backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params.values(), 10., error_if_nonfinite=True))
        optimizer.step()
        row = {k: float(components[k + "_loss"]) for k in ("box", "cls", "dfl")}
        trace.append(dict(step=step, loss=row, preclip_grad_norm=grad_norm))
        if step in (1, 2) or step % 10 == 0:
            print(step, row, "gradient norm", grad_norm, flush=True)
    if len(trace) != record["steps"]:
        raise ValueError("Different number of optimizer steps")
    checkpoint = args.out / "tp-epoch1.pt"
    torch.save({k:v.detach().cpu() for k, v in model.state_dict().items() if k in keys}, checkpoint)
    result = validate(model, raw, args.replay)
    del criterion, optimizer, model, params, pred, losses
    # As a control, replay the same inputs starting at the official initialization,
    # with train-mode BN updates and all learnable tensors held fixed.
    control, _ = read_model(args.assets, trainable)
    control.train()
    with torch.no_grad():
        for batch in batches(args.replay, "train"):
            prompts = raw[0][torch.from_numpy(batch["text_ids"]).long()]
            control.predict(torch.from_numpy(batch["img"]).float() / 255, tpe=prompts)
    bn_only = validate(control, raw, args.replay)
    report = dict(ultralytics=ultralytics.__version__, torch=torch.__version__, device="cpu",
                  epochs=1, seed=record["seed"], train_images=200, steps=len(trace), size=320, batch=4,
                  optimizer="SGD", lr=record["lr"], momentum=.9, weight_decay=0., clip_grad_norm=10.,
                  warmup=False, ema=False, trainable_tensors=len(trainable), gains=dict(box=7.5, cls=.5, dfl=1.5),
                  checkpoint=checkpoint.name, checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  release_sha256=DETECT_SHA, replay=replay, steps_detail=trace,
                  mean_loss={k:np.mean([s["loss"][k] for s in trace]) for k in trace[0]["loss"]},
                  baseline=baseline, epoch1=result, bn_only_no_sgd=bn_only)
    (args.out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Baseline, epoch1, BN only", baseline, result, bn_only, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "init", "replay", "jittor-run", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if any((args.out / f).exists() for f in ("result.json", "tp-epoch1.pt")):
        parser.error("Use a fresh output directory")
    run(args)
