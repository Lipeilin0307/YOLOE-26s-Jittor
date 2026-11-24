"""Replace only the native first-step stem weight and momentum in an otherwise official training trajectory."""
import argparse
import json
from pathlib import Path

import numpy as np

from yoloe11.training_replay import batches


STEM = "model.0.conv.weight"


def capture(args):
    import jittor as jt
    from yoloe11.detector import YOLOE11TP, image_tensor
    from yoloe11.train import clip_gradients
    jt.flags.use_cuda = 0
    jt.set_global_seed(20261121)
    model = YOLOE11TP(args.init)
    model.training = True
    optimizer = jt.optim.SGD(list(model.trainable.values()), lr=.0005, momentum=.9)
    batch = next(batches(args.replay, "train"))
    outputs = model.forward(image_tensor(batch["img"]), model.raw[0][jt.array(batch["text_ids"])])
    loss, terms = model.loss(outputs,batch)
    for v in terms.values():
        v.item()
    optimizer.backward(loss)
    clip_gradients(optimizer)
    optimizer.step()
    jt.sync_all()
    velocity = dict(zip(model.trainable, optimizer.param_groups[0]["values"]))
    np.savez(args.stem, weight=model.trainable[STEM].numpy(), velocity=velocity[STEM].numpy())


def roll_out(args):
    import torch
    from ultralytics.utils.loss import v8DetectionLoss
    from yoloe11.train_reference import read_model, validate
    torch.set_num_threads(4)
    torch.manual_seed(20261121)
    with np.load(args.init / "det.npz") as src:
        names = {k for k in src if not k.endswith(("num_batches_tracked","running_mean","running_var"))}
    raw = torch.from_numpy(np.load(args.init / "raw-text.npy"))
    report = {}
    with np.load(args.stem) as saved:
        native_weight, native_velocity = saved["weight"], saved["velocity"]
    for change_stem in (False,True):
        model, params = read_model(args.assets,names)
        model.train()
        opt = torch.optim.SGD(params.values(),lr=.0005,momentum=.9)
        criterion = v8DetectionLoss(model)
        initial = params[STEM].detach().numpy().copy()
        steps = []
        for step,batch in enumerate(batches(args.replay,"train"), 1):
            img = torch.from_numpy(batch["img"]).float()/255
            prompts = raw[0][torch.from_numpy(batch["text_ids"]).long()]
            labels = {k:torch.from_numpy(batch[k]) for k in ("cls","bboxes","batch_idx")}
            opt.zero_grad(set_to_none=True)
            loss, terms = criterion(model.predict(img,tpe=prompts), labels)
            loss.sum().backward()
            torch.nn.utils.clip_grad_norm_(params.values(),10.)
            opt.step()
            steps.append({k:float(v) for k,v in terms.items()})
            if step == 1 and change_stem:
                p = params[STEM]
                update = p.detach().numpy()-initial
                diff = native_weight-p.detach().numpy()
                velocity = opt.state[p]["momentum_buffer"]
                report["injected"] = dict(parameter=STEM, weight_max_abs=float(np.abs(diff).max()),
                    relative_update_l2=float(np.linalg.norm(diff)/np.linalg.norm(update)),
                    relative_velocity_l2=float(np.linalg.norm(native_velocity-velocity.numpy())/np.linalg.norm(velocity.numpy())))
                with torch.no_grad():
                    p.copy_(torch.from_numpy(native_weight))
                    velocity.copy_(torch.from_numpy(native_velocity))
        val = validate(model,raw,args.replay)
        name = "replace_first_stem" if change_stem else "unaltered"
        report[name] = dict(validation=val, loss=steps)
        print(name, val, flush=True)
    args.report.write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("engine",choices=("capture","roll-out"))
    for key in ("init","replay","stem","assets","report"):
        p.add_argument("--"+key, type=Path, required=key not in ("assets","report"))
    args=p.parse_args()
    (capture if args.engine=="capture" else roll_out)(args)
