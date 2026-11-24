"""Compare native 26s optimizer groups against the pinned Ultralytics build_optimizer without training."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


STAGES = ("TP", "VP", "PF", "SEG")


def official_groups():
    import torch
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.nn.tasks import YOLOEModel, YOLOESegModel

    torch.set_num_threads(2)
    result = {}
    for stage in STAGES:
        mu = stage in ("TP", "SEG")
        cls, cfg = (YOLOESegModel, "yoloe-26s-seg.yaml") if stage == "SEG" else (YOLOEModel, "yoloe-26s.yaml")
        model = cls(cfg, nc=80, verbose=False).eval()
        if stage == "PF":
            head = model.model[-1]
            del head.savpe
            # PF trains only the output convolution in each fused classification tower.
            head.fuse(torch.zeros(1, 1, head.embed))
            active = {id(p) for branch in (head.cv3, head.one2one_cv3) for tower in branch for p in tower[-1].parameters()}
        params = {id(p): name for name, p in model.named_parameters() if stage != "PF" or id(p) in active}
        trainer = object.__new__(BaseTrainer)
        lr = .00125 if mu else .002
        decay = .0007 if mu else .025
        opt = trainer.build_optimizer(model, name="MuSGD" if mu else "AdamW", lr=lr, momentum=.9, decay=decay)
        groups = {}
        for group in opt.param_groups:
            for p in group["params"]:
                if id(p) in params:
                    name = params[id(p)]
                    if name in groups:
                        raise AssertionError(f"Duplicate official optimizer parameter {name}")
                    groups[name] = [group["param_group"], group["lr"] / lr, group["weight_decay"]]
        if set(groups) != set(params.values()):
            raise AssertionError(f"Incomplete official optimizer for {stage}")
        result[stage] = groups
    return result


def compare(init, reference):
    from model import parameter_enabled
    from optimizer import parameter_group

    stats = {}
    for stage in STAGES:
        mu = stage in ("TP", "SEG")
        prefix = "seg" if stage == "SEG" else "det"
        spec = json.loads((init / (prefix + ".json")).read_bytes())
        official = reference[stage]
        native = {}
        with np.load(init / (prefix + ".npz")) as state:
            for name in state.files:
                if name.endswith((".running_mean", ".running_var", ".num_batches_tracked")):
                    continue
                if stage == "PF" and not parameter_enabled(stage, name):
                    continue
                group, multiplier = parameter_group(name, state[name].ndim, spec, mu)
                decay = (.0007 if mu else .025) if group in ("weight", "muon") else 0.
                native[name] = [group, multiplier, decay]
        errors = [k for k in sorted(official.keys() | native.keys()) if official.get(k) != native.get(k)]
        if errors:
            raise AssertionError(f"{stage}: {[(key, official.get(key), native.get(key)) for key in errors[:10]]}")
        stats[stage] = dict(checked=len(native), active=sum(parameter_enabled(stage, name) for name in native),
                            groups=dict(Counter(v[0] for v in native.values())),
                            boosted=sum(v[1] == 3 for v in native.values()), mismatches=0)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=("pytorch", "jittor"))
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--init", type=Path, default=Path(__file__).parent / "init")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.engine == "pytorch":
        args.reference.write_text(json.dumps(official_groups(), indent=2) + "\n")
    else:
        stats = compare(args.init, json.loads(args.reference.read_bytes()))
        print(json.dumps(stats, indent=2))
        if args.report:
            args.report.write_text(json.dumps(stats, indent=2) + "\n")
