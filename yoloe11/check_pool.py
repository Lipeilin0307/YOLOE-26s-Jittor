"""Probe the third shared-start step with the official SPPF max-pool winners (diagnostics only)."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from yoloe11.trace_training import compare, concat, groups, restore
from yoloe11.training_replay import batches


def probe(args):
    batch = next(itertools.islice(batches(args.replay,"train"), 2, 3))
    with np.load(args.init/"det.npz") as original:
        names = sorted(k for k in original if not k.endswith(("running_var","running_mean","num_batches_tracked")))
        bn_names = sorted(k for k in original if k.endswith(("running_var","running_mean")))
    with np.load(args.cache/"start-03.npz") as start:
        weights, buffers = start["params"], start["buffers"]
    if args.engine=="reference":
        import torch
        import torch.nn.functional as F
        from yoloe11.train_reference import read_model
        torch.set_num_threads(4)
        model, params = read_model(args.assets,set(names))
        model.train()
        with torch.no_grad():
            for flat, keys, dest in ((weights,names,params),(buffers,bn_names,dict(model.named_buffers()))):
                offset=0
                for k in keys:
                    v=dest[k]
                    v.copy_(torch.from_numpy(flat[offset:offset+v.numel()]).reshape_as(v))
                    offset+=v.numel()
            record={}
            def capture(_, inputs, x):
                record["input"]=x.detach().numpy().copy()
                for n in range(3):
                    x, idx = F.max_pool2d(x,5,stride=1,padding=2,return_indices=True)
                    record[f"idx{n}"]=idx.numpy().astype("int32")
            h=model.model[9].cv1.register_forward_hook(capture)
            raw=torch.from_numpy(np.load(args.init/"raw-text.npy"))
            model.predict(torch.from_numpy(batch["img"]).float()/255,
                          tpe=raw[0][torch.from_numpy(batch["text_ids"]).long()])
            h.remove()
            np.savez(args.fixture,**record)
        return
    import jittor as jt
    from jittor import nn
    from yoloe11.detector import YOLOE11TP, image_tensor
    jt.flags.use_cuda=0
    jt.set_global_seed(20261121)
    with np.load(args.fixture) as src:
        pool_input=src["input"]
        winners=[src[f"idx{n}"] for n in range(3)]
    def native_indices(x):
        return nn.max_pool2d(x,5,stride=1,padding=2,return_indices=True)[1].numpy()
    x=jt.array(pool_input)
    common_mismatches=[]
    for idx in winners:
        common_mismatches.append(int(np.count_nonzero(native_indices(x)!=idx)))
        x=nn.max_pool2d(x,5,stride=1,padding=2)
    report=dict(common_input_winner_mismatches=common_mismatches, entries_per_pool=winners[0].size)
    with np.load(args.cache/"step-03.npz") as ref:
        expected=ref["gradient"]
    for force in (False,True):
        model=YOLOE11TP(args.init)
        model.training=True
        restore(weights,names,model.trainable,jt)
        restore(buffers,bn_names,model.state,jt)
        layer=model.layer
        mismatches=[]
        def pooled(key, x):
            y=layer(key,x)
            if key=="model.9.m":
                indices=winners[len(mismatches)]
                mismatches.append(int(np.count_nonzero(native_indices(x)!=indices)))
                if force:
                    shape=x.shape
                    chosen=jt.gather(x.reshape(shape[0],shape[1],-1),2,
                                     jt.array(indices).reshape(shape[0],shape[1],-1)).reshape(shape)
                    return y.detach()+(chosen-chosen.detach())
            return y
        model.layer=pooled
        outputs=model.forward(image_tensor(batch["img"]),model.raw[0][jt.array(batch["text_ids"])])
        loss,_=model.loss(outputs,batch)
        grads=jt.grad(loss,[model.trainable[k] for k in names])
        jt.sync(grads)
        diff=compare(concat([g.numpy() for g in grads]),expected,groups(names,model.trainable))
        report["official_indices" if force else "native_indices"]=dict(winner_mismatches=mismatches,
                                                                     gradient=diff)
        del model, grads, outputs, loss
    args.out.write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("engine",choices=("reference","jittor"))
    for key in ("init","replay","cache","fixture","assets","out"):
        p.add_argument("--"+key, type=Path,required=key not in ("assets","out"))
    probe(p.parse_args())
