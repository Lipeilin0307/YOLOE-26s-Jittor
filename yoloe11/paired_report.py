"""Summarize the local three-seed, two-learning-rate TP comparison in AP points."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch


SEEDS = (20261121, 20261122, 20261123)
RATES = ("0.0005", "0.00005")


def read(path):
    return json.loads(path.read_bytes())


def scores(record):
    if record["seen"] != 200:
        raise ValueError("Incomplete validation")
    return {k:100*record["metrics"][f"metrics/{k}(B)"] for k in ("mAP50", "mAP50-95")}


def check_weights(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Changed checkpoint {path}")


def displacement(source, updated):
    groups = set()
    changed = 0
    squared = 0.
    for key, initial in source.items():
        delta = np.asarray(updated[key]).reshape(initial.shape).astype(np.float64) - initial
        squared += np.square(delta).sum()
        if np.any(delta):
            changed += 1
            groups.add(key.split(".")[2] if key.startswith("model.23.") else "image")
    if not {"image", "reprta", "cv2", "cv3"}.issubset(groups):
        raise ValueError(f"Incomplete training updates: {sorted(groups)}")
    return dict(changed_tensors=changed, parameter_delta_l2=float(np.sqrt(squared)), changed_groups=sorted(groups))


def run(args):
    torch.set_num_threads(1)
    archive = read(args.previous_reference / "result.json")
    control = read(args.previous_native / "epoch1.json")
    with np.load(args.init / "det.npz") as initial:
        source = {k:initial[k] for k in initial if not k.endswith(("running_mean","running_var","num_batches_tracked"))}
    if len(source) != 271:
        raise ValueError("Unexpected trainable set")
    rows = []
    for seed in SEEDS:
        for rate in RATES:
            prior = seed == SEEDS[0] and rate == RATES[0]
            base = args.runs / str(seed) / ("lr-" + rate)
            jpath = args.previous_native if prior else base / "jittor"
            ppath = args.previous_reference if prior else base / "pytorch"
            j, p = read(jpath/"epoch1.json"), read(ppath/"result.json")
            if not (j["config"] == control["config"] and j["seed"] == p["seed"] == seed and
                    j["lr"] == p["lr"] == float(rate) and j["train_images"] == p["train_images"] == 200 and
                    j["steps"] == p["steps"] == 50 and j["trainable_tensors"] == p["trainable_tensors"] == 271 and
                    j["reduction_dtype"] == "float64" and j["input_normalization"] == "wide_cpu_divide" and
                    p["replay"]["val"]["arrays_sha256"] == archive["replay"]["val"]["arrays_sha256"]):
                raise ValueError(f"Training configuration mismatch for {seed}/{rate}")
            for split in ("train","val"):
                if p["replay"][split]["manifest_sha256"] != j["config"][split+"_sha"]:
                    raise ValueError("Manifest mismatch")
            if not prior and j["train_arrays_sha256"] != p["replay"]["train"]["arrays_sha256"]:
                raise ValueError("Training replay mismatch")
            check_weights(jpath/"tp-epoch1.npz", j["checkpoint_sha"])
            check_weights(ppath/"tp-epoch1.pt", p["checkpoint_sha256"])
            with np.load(jpath/"tp-epoch1.npz") as final:
                jdelta = displacement(source, final)
            pdelta = displacement(source,torch.load(ppath/"tp-epoch1.pt",map_location="cpu",weights_only=True))
            js, ps = scores(j["validation"]), scores(p["epoch1"])
            rows.append(dict(seed=seed,lr=rate,archived_baseline=prior,train_sha=p["replay"]["train"]["arrays_sha256"],
                             val_sha=p["replay"]["val"]["arrays_sha256"], pytorch=ps,jittor=js,
                             delta_AP=js["mAP50-95"]-ps["mAP50-95"], no_SGD_BN_only=scores(p["bn_only_no_sgd"]),
                             initial=scores(p["baseline"]), jittor_update=jdelta,pytorch_update=pdelta,
                             mean_loss=dict(jittor=j["mean_loss"],pytorch=p["mean_loss"]),
                             checkpoints=dict(jittor=j["checkpoint_sha"],pytorch=p["checkpoint_sha256"])))
    if any(len({r["train_sha"] for r in rows if r["seed"]==s}) != 1 for s in SEEDS):
        raise ValueError("Learning rate arms used different training inputs")
    if len({r["train_sha"] for r in rows}) != len(SEEDS):
        raise ValueError("Seeds did not generate distinct training batches")
    if any(r["initial"] != rows[0]["initial"] for r in rows):
        raise ValueError("Pretraining validation changed across runs")
    for seed in SEEDS:
        first, second = [r for r in rows if r["seed"] == seed]
        if first["no_SGD_BN_only"] != second["no_SGD_BN_only"]:
            raise ValueError("BN-only controls differ for the same inputs")
    summary = {}
    for rate in RATES:
        subset = [r for r in rows if r["lr"] == rate]
        gaps = [r["delta_AP"] for r in subset]
        summary[rate] = dict(pytorch_mean_AP=mean(r["pytorch"]["mAP50-95"] for r in subset),
                             jittor_mean_AP=mean(r["jittor"]["mAP50-95"] for r in subset),
                             mean_signed_difference=mean(gaps), paired_difference_sample_sd=stdev(gaps),
                             mean_abs_difference=mean(map(abs,gaps)), max_abs_difference=max(map(abs,gaps)),
                             pairs_within_one_AP=sum(abs(g)<=1 for g in gaps))
    report = dict(model="YOLOE-11s text-prompt bbox detection, local CPU, 1 epoch, COCO 200 train + 200 validation",
                  seeds=SEEDS, rates=RATES, criterion="Every paired raw mAP50-95 gap <= 1 AP point; all detector groups updated",
                  optimizer="SGD, momentum .9, global norm clip 10, batch 4, 320px, no EMA/warmup",
                  units="AP points on a 0-100 scale; delta = Jittor - PyTorch",summary=summary, runs=rows)
    args.report.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("init","runs","previous-reference","previous-native","report"):
        parser.add_argument("--"+key,required=True,type=Path)
    run(parser.parse_args())
