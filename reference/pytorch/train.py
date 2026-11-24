"""Train YOLOE-26s TP, VP, PF and SEG and evaluate reloaded weights."""
import argparse
import csv
import hashlib
import json
import math
import os
import time
from copy import copy
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.yoloe.train import YOLOEPEFreeTrainer, YOLOETrainerFromScratch, YOLOEVPTrainer
from ultralytics.models.yolo.yoloe.train_seg import YOLOESegTrainerFromScratch
from ultralytics.models.yolo.yoloe.val import YOLOEDetectValidator
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import init_seeds


SEED = 20261121
STAGES = ("TP", "VP", "PF", "SEG")


class ReferenceValidator(YOLOEDetectValidator):
    """Build class exemplars from training images even during per-epoch validation."""
    def get_vpe_dataloader(self, data):
        # This short-lived loader is discarded after extracting prototypes, also on WSL.
        # Keep it synchronous so a pinned-memory thread cannot outlive its workers.
        workers = self.args.workers
        try:
            self.args.workers = 0
            return super().get_vpe_dataloader(data)
        finally:
            self.args.workers = workers

    @torch.no_grad()
    def get_model(self, model=None, trainer=None, **kwargs):
        if trainer is None:
            return super().get_model(model, refer_data=self.reference, load_vp=True)
        self.device = trainer.device
        self.args.quantize = None
        ema = trainer.ema.ema.eval()
        data = check_det_dataset(self.reference)
        embeddings = self.get_visual_pe(self.get_vpe_dataloader(data), ema)
        ema.set_classes(list(data["names"].values()), embeddings)
        return ema


class VisualTrainer(YOLOEVPTrainer):
    def get_validator(self):
        validator = ReferenceValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args),
                                       _callbacks=self.callbacks)
        validator.reference = self.reference
        return validator


def freeze_layers(model, stage):
    head_idx = len(model.model) - 1
    head = model.model[-1]
    if stage == "TP":
        return [f"{head_idx}.savpe"]
    keep = {"VP": {"savpe", "cv4", "one2one_cv4"},
            "PF": {"cv3", "one2one_cv3"},
            "SEG": {"cv5", "one2one_cv5", "proto"}}[stage]
    frozen = list(range(head_idx)) + [f"{head_idx}.{k}" for k, _ in head.named_children() if k not in keep]
    if stage == "PF":
        for branch in keep:
            for i, tower in enumerate(getattr(head, branch)):
                frozen.extend(f"{head_idx}.{branch}.{i}.{j}" for j in range(len(tower)-1))
    return frozen


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class WeightAudit:
    def __init__(self, n_train, n_val):
        self.n_train, self.n_val = n_train, n_val

    def start(self, trainer):
        if (len(trainer.train_loader.dataset), len(trainer.test_loader.dataset)) != (self.n_train, self.n_val):
            raise RuntimeError("Image counts differ from prepared manifests")
        self.before = {k: v.detach().cpu().clone() for k, v in trainer.model.named_parameters()}

    def finish(self, trainer):
        changed, trainable = [], []
        for k, param in trainer.model.named_parameters():
            value = param.detach().cpu()
            if not value.isfinite().all():
                raise FloatingPointError(k)
            if param.requires_grad:
                trainable.append(k)
            if not torch.equal(self.before[k], value):
                if not param.requires_grad:
                    raise RuntimeError(f"Frozen parameter changed: {k}")
                changed.append(k)
        if not changed:
            raise RuntimeError("No model parameter updated")
        self.result = dict(changed_parameters=changed, trainable_tensors=len(trainable),
                           optimizer_steps=int(trainer.ema.updates))


def deploy_pf(tp_path, pf_path, output, data_yaml, args):
    """Pair TP recognition weights with the learned PF objectness projections."""
    tp, _ = load_checkpoint(tp_path)
    pf, ckpt = load_checkpoint(pf_path)
    names = yaml.safe_load(data_yaml.read_text())["names"]
    tp.eval().get_vocab(names)  # fuse both one-to-many and one-to-one text classifiers offline
    head = tp.model[-1]
    vocab = torch.nn.ModuleList(tower[-1] for tower in head.cv3)
    o2o = torch.nn.ModuleList(tower[-1] for tower in head.one2one_cv3)
    pf.eval().set_vocab(vocab, names, one2one_vocab=o2o)
    pf.args["single_cls"] = False
    torch.save({**ckpt, "model": pf.half(), "ema": None,
                "train_args": {**ckpt["train_args"], "single_cls": False}}, output)
    validator = DetectionValidator(args=dict(model=str(output), data=str(data_yaml), imgsz=args.imgsz,
        device=args.device, batch=args.batch, workers=args.workers, plots=False, save_json=False,
        nms=False, verbose=False, project=str(output.parent.parent), name="vocab-val", exist_ok=True))
    metrics = validator(model=str(output))
    return int(validator.seen), {k: float(v) for k, v in metrics.items() if k.startswith("metrics/")}


def train_stage(stage, args, split):
    start = time.monotonic()
    task = "segment" if stage == "SEG" else "detect"
    model_yaml = "yoloe-26s-seg.yaml" if stage == "SEG" else "yoloe-26s.yaml"
    data_yaml = args.data / f"{task}.yaml"
    tp_path = args.out / "TP/weights/best.pt"
    source = args.assets / "yolo26s-objv1-150.pt" if stage == "TP" else tp_path
    if not source.is_file():
        raise FileNotFoundError(source)
    if (args.out / stage).exists():
        raise FileExistsError(f"Use a new run directory or select unfinished stages: {args.out / stage}")
    data = {s: {"yolo_data": [str(data_yaml)]} for s in ("train", "val")}
    num_epochs = 1 if split["profile"] == "smoke" else (30 if stage == "TP" else 10)
    is_mu = stage in ("TP", "SEG")
    options = dict(model=model_yaml, data=data, task=task, epochs=num_epochs, device=args.device,
        imgsz=args.imgsz, batch=args.batch, workers=args.workers, seed=SEED, deterministic=True,
        optimizer="MuSGD" if is_mu else "AdamW", lr0=0.00125 if is_mu else 0.002,
        lrf=0.5 if is_mu else 0.01, momentum=0.9, weight_decay=0.0007 if is_mu else 0.025,
        project=str(args.out), name=stage, save=True, val=True, plots=False, save_json=False,
        single_cls=stage == "PF", nms=False, amp=False, verbose=False)
    if split["profile"] == "smoke":
        options.update(warmup_epochs=0, mosaic=0, close_mosaic=0, nbs=args.batch)
    elif stage == "TP":
        options["close_mosaic"] = 2
    trainer_cls = {"TP": YOLOETrainerFromScratch, "VP": VisualTrainer,
                   "PF": YOLOEPEFreeTrainer, "SEG": YOLOESegTrainerFromScratch}[stage]
    trainer = trainer_cls(overrides=options)
    trainer.reference = str(args.data / "reference.yaml")
    weights, _ = load_checkpoint(source)
    trainer.model = trainer.get_model(cfg=model_yaml, weights=weights, verbose=False)
    trainer.args.freeze = freeze_layers(trainer.model, stage)
    YAML.save(trainer.save_dir / "args.yaml", vars(trainer.args))
    n_train, n_val = (split["splits"][s]["images"] for s in ("train", "val"))
    audit = WeightAudit(n_train, n_val)
    trainer.add_callback("on_train_start", audit.start)
    trainer.add_callback("on_train_epoch_end", lambda t: audit.finish(t) if t.epoch == t.epochs-1 else None)
    trainer.train()  # includes standalone final validation from best.pt
    best, last = trainer.best, trainer.last
    if not best.is_file() or not last.is_file() or trainer.validator.seen != n_val:
        raise RuntimeError(f"{stage}: missing weights or incomplete validation")
    metrics = {k: float(v) for k, v in trainer.metrics.items() if k.startswith("metrics/")}
    with (trainer.save_dir / "results.csv").open() as f:
        history = list(csv.DictReader(f, skipinitialspace=True))
    losses = {k: float(v) for k, v in history[-1].items() if k.startswith("train/")}
    result = dict(stage=stage, epochs=len(history), train_images=n_train, validation_images=n_val,
        input_checkpoint=str(source), input_sha256=sha256(source), best_sha256=sha256(best),
        best=str(best), last=str(last), reload_metrics=metrics, train_losses=losses, **audit.result)
    if stage == "PF":
        deploy = best.with_name("prompt-free-coco80.pt")
        seen, retrieval_metrics = deploy_pf(tp_path, best, deploy, data_yaml, args)
        if seen != n_val or not retrieval_metrics:
            raise RuntimeError("Incomplete PF vocabulary evaluation")
        result.update(vocabulary_checkpoint=str(deploy), vocabulary_sha256=sha256(deploy),
                      vocabulary_metrics=retrieval_metrics)
    values = [*losses.values(), *metrics.values(), *result.get("vocabulary_metrics", {}).values()]
    if not losses or not metrics or len(history) != num_epochs or not all(map(math.isfinite, values)):
        raise RuntimeError(f"{stage}: incomplete history or non-finite loss/metrics")
    result["elapsed_seconds"] = round(time.monotonic() - start, 2)
    (trainer.save_dir / "verified.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"VERIFIED {stage}: {len(audit.result['changed_parameters'])} updated tensors, "
          f"{n_val} validation images, {metrics}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Output directory of prepare.py")
    parser.add_argument("--assets", type=Path, required=True, help="Directory with detector .pt and MobileCLIP2 .ts")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stages", choices=STAGES, nargs="+", default=list(STAGES))
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    args.data, args.assets, args.out = (p.resolve() for p in (args.data, args.assets, args.out))
    split = json.loads((args.data / "split.json").read_text())
    args.imgsz = args.imgsz or (320 if split["profile"] == "smoke" else 640)
    if not (args.assets / "mobileclip2_b.ts").is_file():
        raise FileNotFoundError(args.assets / "mobileclip2_b.ts")
    os.chdir(args.assets)  # Ultralytics resolves the TorchScript text encoder relative to cwd
    print(ultralytics.__version__, torch.__version__, flush=True)
    for stage in args.stages:
        init_seeds(SEED, deterministic=True)
        train_stage(stage, args, split)
