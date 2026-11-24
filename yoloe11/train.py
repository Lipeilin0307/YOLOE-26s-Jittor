"""One-epoch YOLOE-11s text-prompt detection acceptance run and fresh-process validation."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import jittor as jt
import numpy as np

from native_data import NativeWorker
from train import digest, save_arrays, save_json
from yoloe11.detector import YOLOE11TP, image_tensor, suppress
from yoloe11.training_replay import update_digest


def clip_gradients(optimizer, max_norm=10.):
    """Use a wide reduction for the global L2 norm on CPU, retaining float32 SGD buffers."""
    if jt.flags.use_cuda:
        optimizer.clip_grad_norm(max_norm)
        return
    grads = [g for group in optimizer.param_groups
             for p,g in zip(group["params"], group["grads"]) if not p.is_stop_grad()]
    squared = jt.stack([g.float64().sqr().sum() for g in grads]).sum()
    scale = (max_norm/(squared.sqrt()+1e-6)).minimum(1.).float32()
    for g in grads:
        g.update(g*scale)


def validate(model, worker, legacy_input=False):
    model.training=False
    with jt.no_grad():
        pe=model.text(model.raw)
        for batch in worker.batches("val",0,200):
            x=(jt.array(batch["img"]).float32()/255. if legacy_input else image_tensor(batch["img"]))
            rec=model.head_outputs(model.image_features(x),pe)
            boxes,scores=model.decoded(rec,*x.shape[-2:])
            worker.send(suppress(boxes,scores))
    result=worker.read()
    if result["seen"]!=200 or not all(np.isfinite(v) for v in result["metrics"].values()):
        raise ValueError("Incomplete or nonfinite validation")
    return result


def run(args):
    if any(importlib.util.find_spec(k) is not None for k in ("torch","torchvision","ultralytics")):
        raise RuntimeError("Use the independent Jittor environment")
    if args.batch < 1 or args.size < 32 or args.size % 32 or args.lr <= 0:
        raise ValueError("Expected positive batch/lr and an image size divisible by 32")
    parity=json.loads(args.parity.read_bytes())
    if parity["init_sha"]!=digest(args.init/"det.npz") or parity["raw_sha"]!=digest(args.init/"raw-text.npy"):
        raise ValueError("Initialization differs from the parity-tested release")
    split=json.loads((args.data/"split.json").read_bytes())["splits"]
    if any(split[k]["images"]!=200 for k in ("train","val")):
        raise ValueError("Expected the fixed 200 train + 200 validation images")
    args.out.mkdir(parents=True,exist_ok=True)
    ckpt=args.out/"tp-epoch1.npz"
    info=args.out/"epoch1.json"
    cfg=dict(stage="TP",data=str(args.data),size=args.size,batch=args.batch,workers=2,smoke=True,epochs=1,seed=args.seed,
             train_sha=split["train"]["manifest_sha256"],val_sha=split["val"]["manifest_sha256"])
    worker=NativeWorker(cfg)
    identity=dict(train_sha=cfg["train_sha"],val_sha=cfg["val_sha"],init_sha=parity["init_sha"],
                  raw_sha=parity["raw_sha"],graph_sha=digest(args.init/"det.json"),size=args.size,batch=args.batch)
    try:
        if args.verify_only:
            record=json.loads(info.read_bytes())
            if digest(ckpt)!=record["checkpoint_sha"] or record["config"]!=identity:
                raise ValueError("Checkpoint, data or settings changed")
            result=validate(YOLOE11TP(args.init,checkpoint=ckpt),worker,
                            legacy_input="input_normalization" not in record)
            for k,v in result["metrics"].items():
                np.testing.assert_allclose(v,record["validation"]["metrics"][k],rtol=0,atol=1e-6)
            save_json(args.out/"reloaded.json",dict(checkpoint_sha=record["checkpoint_sha"],**result))
            print("Fresh-process validation",result,flush=True)
            return
        if info.exists() or ckpt.exists():
            raise FileExistsError("Use a fresh training output directory or --verify-only")
        model=YOLOE11TP(args.init)
        opt=jt.optim.SGD(list(model.trainable.values()),lr=args.lr,momentum=.9)
        model.training=True
        steps=seen=0
        totals={}
        train_digest=hashlib.sha256()
        start=time.monotonic()
        for batch in worker.batches("train",0,200):
            update_digest(train_digest,batch)
            x=image_tensor(batch["img"])
            raw=model.raw[0][jt.array(batch["text_ids"])]
            loss,terms=model.loss(model.forward(x,raw),batch)
            values={k:float(v.item()) for k,v in terms.items()}
            if not all(np.isfinite(v) for v in values.values()):
                raise FloatingPointError(values)
            opt.backward(loss)
            clip_gradients(opt)
            opt.step()
            jt.sync_all()
            steps+=1
            seen+=len(batch["img"])
            for k,v in values.items():
                totals[k]=totals.get(k,0.)+v
            if steps<3 or steps%10==0 or seen==200:
                print("step",steps,"images",seen,"loss",values,"seconds",round(time.monotonic()-start,1),flush=True)
        if seen!=200:
            raise RuntimeError(f"Trained on {seen}/200")
        with np.load(args.init/"det.npz") as initial:
            changed=[k for k,v in model.trainable.items() if not np.array_equal(initial[k],v.numpy())]
        groups={"image":any(not k.startswith(model.head+".") for k in changed),
                "box":any(k.startswith(model.head+".cv2.") for k in changed),
                "class":any(k.startswith(model.head+".cv3.") for k in changed),
                "text_projection":any(k.startswith(model.head+".reprta.") for k in changed)}
        if not all(groups.values()):
            raise RuntimeError(f"Missing parameter updates: {groups}")
        save_arrays(ckpt,model.state)
        result=validate(model,worker)
        save_json(info,dict(config=identity,epochs=1,steps=steps,train_images=seen,
                            mean_loss={k:v/steps for k,v in totals.items()},changed_tensors=len(changed),
                            changed_groups=groups,trainable_tensors=len(model.trainable),
                            optimizer="SGD",lr=args.lr,seed=args.seed,train_arrays_sha256=train_digest.hexdigest(),
                            device=args.device,jittor=jt.__version__,
                            reduction_dtype="float64" if args.device=="cpu" else "float32",
                            input_normalization="wide_cpu_divide" if args.device=="cpu" else "float32_divide",
                            checkpoint_sha=digest(ckpt),validation=result))
        print("Saved and validated",result,flush=True)
    finally:
        worker.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("init","data","parity","out"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    p.add_argument("--size",type=int,default=320)
    p.add_argument("--batch",type=int,default=4)
    p.add_argument("--lr",type=float,default=.0005)
    p.add_argument("--seed",type=int,default=20261121)
    p.add_argument("--verify-only",action="store_true")
    a=p.parse_args()
    jt.flags.use_cuda=int(a.device=="cuda")
    jt.set_global_seed(a.seed)
    run(a)
