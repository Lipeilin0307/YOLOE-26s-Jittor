"""Train, checkpoint, resume and independently reload YOLOE-26s with Jittor."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import jittor as jt
import numpy as np
from model import YOLOE26
from objective import objective, detections
from optimizer import Optimizer
import wire


STAGES=("TP","VP","PF","SEG")


def save_arrays(path,arrays):
    tmp=path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        values={k:(v.numpy() if isinstance(v,jt.Var) else v) for k,v in arrays.items()}
        if any(np.issubdtype(v.dtype,np.floating) and not np.isfinite(v).all() for v in values.values()):
            raise FloatingPointError(f"Non-finite state for {path}")
        np.savez(f,**values)
    os.replace(tmp,path)


def save_json(path,data):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    os.replace(tmp,path)


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()


class Worker:
    def __init__(self,config,python):
        env={**os.environ,"CUDA_VISIBLE_DEVICES":"","OMP_NUM_THREADS":"2"}
        env.pop("LD_PRELOAD",None)
        self.proc=subprocess.Popen([python,"-u",str(Path(__file__).with_name("data_worker.py"))],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,env=env)
        self.send(config)

    def send(self,value):
        wire.write(self.proc.stdin,value)

    def read(self):
        try: return wire.read(self.proc.stdout)
        except EOFError as exc: raise RuntimeError(f"CPU data/evaluation worker exited ({self.proc.poll()})") from exc

    def batches(self,mode,epoch,expected):
        self.send((mode,epoch))
        total,count=self.read()
        if count!=expected: raise RuntimeError(f"{mode} has {count} images, expected {expected}")
        seen=0
        for _ in range(total):
            batch=self.read(); seen+=len(batch["img"]); yield batch
        if seen!=count: raise RuntimeError(f"{mode} processed {seen}/{count} images")

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try: self.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.proc.terminate(); self.proc.wait()


def image_tensor(batch):
    pixels = jt.array(batch["img"]).stop_grad()
    if jt.flags.use_cuda:
        return pixels.float32() / 255.
    # Avoid the CPU compiler's approximate float32 reciprocal on byte inputs.
    return (pixels.float64() / 255.).float32()


def exemplars(model,worker,config):
    sums=np.zeros((80,512),np.float64)
    counts=np.zeros(80,np.int64)
    for batch in worker.batches("ref",0,config["ntrain"]):
        z=model.visual(model.image_features(image_tensor(batch)),jt.array(batch["visuals"]).stop_grad()).numpy()
        labels=batch["cls"].ravel().astype(int); idx=batch["batch_idx"].ravel()
        for b in range(len(z)):
            cls=np.unique(labels[idx==b])
            sums[cls]+=z[b,:len(cls)]; counts[cls]+=1
    sums/=np.maximum(counts[:,None],1)
    sums/=np.maximum(np.linalg.norm(sums,axis=1,keepdims=True),1.e-12)
    return jt.array(sums[None].astype("float32")).stop_grad()


def evaluate(model,worker,config,vocab=False,epoch_validation=False):
    model.training=False
    with jt.no_grad():
        pe=exemplars(model,worker,config) if model.stage=="VP" else None
        if model.stage in ("TP","SEG"): pe=model.text(model.raw[:,:80])
        mode="vocab" if vocab else ("epoch_val" if epoch_validation else "val")
        for batch in worker.batches(mode,0,config["nval"]):
            x=image_tensor(batch)
            rec=model.forward(x,pe=pe,retrieve=vocab,both=False)["one2one"]
            worker.send(detections(rec,*x.shape[-2:]))
    result=worker.read()
    if result["seen"]!=config["nval"] or not all(np.isfinite(v) for v in result["metrics"].values()):
        raise RuntimeError("Validation failed")
    return result["metrics"]


def stage_config(args,stage):
    split=json.loads((args.data/"split.json").read_text()); smoke=split["profile"]=="smoke"
    return dict(stage=stage,data=str(args.data),out=str(args.out/stage),smoke=smoke,
        ntrain=split["splits"]["train"]["images"],nval=split["splits"]["val"]["images"],
        train_sha=split["splits"]["train"]["manifest_sha256"],val_sha=split["splits"]["val"]["manifest_sha256"],
        size=args.imgsz or (320 if smoke else 640),batch=args.batch,workers=args.workers,
        epochs=1 if smoke else (30 if stage=="TP" else 10))


def checkpoint_resume(path,model,opt,next_epoch,best):
    state={}
    for prefix,group in (("net",model.state),("ema",opt.ema),("vel",opt.vel),("second",opt.second),("acc",opt.acc)):
        state.update({prefix+"/"+k:v for k,v in group.items()})
    state["clock"]=np.array([next_epoch,opt.steps,opt.last,best],np.float64)
    save_arrays(path,state)


def restore(path,model,opt):
    with np.load(path) as z:
        for prefix,group in (("net",model.state),("ema",opt.ema),("vel",opt.vel),("second",opt.second)):
            for k,v in group.items(): v.assign(jt.array(z[prefix+"/"+k]))
        for k in z.files:
            if k.startswith("acc/"): opt.acc[k[4:]]=jt.array(z[k]).stop_grad()
        epoch,steps,last,best=z["clock"]
    opt.steps,opt.last=int(steps),int(last)
    for v in model.trainable.values():
        v.start_grad()
    return int(epoch),float(best)


def train_stage(args,stage,verify_only=False):
    cfg=stage_config(args,stage); out=args.out/stage; out.mkdir(exist_ok=True,parents=True)
    parent=args.out/"TP"/"best.npz" if stage!="TP" else None
    identity={**cfg,"parent_sha":digest(parent or args.init/"det.npz")}
    if not args.reference_python:
        identity["data_backend"]="native-v1"
        prefix="seg" if stage=="SEG" else "det"
        identity["init_sha"]={name:digest(args.init/name)
                              for name in (prefix+".json",prefix+".npz","raw-text.npy")}
    if not jt.flags.use_cuda:
        identity["device"] = "cpu"
    if stage in ("TP", "SEG"):
        identity["musgd_revision"] = 2
    if stage == "SEG":
        identity["seg_lr_groups"] = "detector-head-only"
    pf_vocab_assignment = stage == "PF" and getattr(args, "pf_p5_assignment", "object") == "tp-vocab"
    if pf_vocab_assignment:
        identity["pf_p5_assignment"] = "tp-vocab"
    pf_p5_bias_lr = getattr(args, "pf_p5_bias_warmup_lr", 0.1) if stage == "PF" else 0.1
    if not np.isfinite(pf_p5_bias_lr) or pf_p5_bias_lr < 0:
        raise ValueError("PF P5 bias warmup learning rate must be finite and nonnegative")
    if pf_p5_bias_lr != 0.1:
        identity["pf_p5_bias_warmup_lr"] = pf_p5_bias_lr
    batches = (cfg["ntrain"] + cfg["batch"] - 1) // cfg["batch"]
    if not cfg["smoke"] and (64 % cfg["batch"] or round(64 / cfg["batch"]) > cfg["epochs"] * batches):
        identity["accumulation"] = "rounded-and-capped"
    config_file=out/"config.json"
    if config_file.exists():
        saved = json.loads(config_file.read_text())
        expected = identity.copy()
        if verify_only:
            # Optimizer settings do not enter validation of archived checkpoints.
            for key in ("pf_p5_assignment", "pf_p5_bias_warmup_lr"):
                expected.pop(key, None)
                if key in saved:
                    expected[key] = saved[key]
            for key in ("seg_lr_groups", "accumulation", "musgd_revision"):
                if key not in saved:
                    expected.pop(key, None)
        if saved != expected:
            raise ValueError("Changed run configuration. Use a new output directory for training with the revised settings.")
    elif not verify_only:
        save_json(config_file,identity)
    best_path=out/"best.npz"; last_path=out/"last.npz"; resume=out/"training-state.npz"
    verified=out/"verified.json"
    if verify_only or verified.exists():
        record=json.loads(verified.read_text())
        if (record["best_sha"]!=digest(best_path) or record["last_sha"]!=digest(last_path)
                or record["parent_sha"]!=identity["parent_sha"]):
            raise ValueError(f"Checkpoint identity mismatch for {stage}")
        if record["epochs"]!=cfg["epochs"] or record["steps"]<1 or record["images"]!=cfg["nval"]:
            raise ValueError(f"Incomplete run for {stage}")
        if not verify_only:
            return
    if args.reference_python:
        worker=Worker(cfg,args.reference_python)
    else:
        from native_data import NativeWorker
        worker=NativeWorker(cfg)
    try:
        if verify_only:
            model=YOLOE26(args.init,stage,parent,best_path)
            metrics={"best":evaluate(model,worker,cfg)}
            if stage=="PF": metrics["coco80"]=evaluate(model,worker,cfg,True)
            for variant,values in metrics.items():
                previous=record["metrics"][variant]
                for name,value in values.items():
                    if not np.isclose(value,previous[name],rtol=1e-4,atol=1e-5):
                        raise ValueError(f"Changed reload metric {stage}/{variant}/{name}: {value} vs {previous[name]}")
            return metrics
        model=YOLOE26(args.init,stage,parent)
        n=(cfg["ntrain"]+cfg["batch"]-1)//cfg["batch"]
        opt=Optimizer(model,cfg["epochs"],cfg["batch"],n,cfg["smoke"],pf_p5_bias_lr)
        start,best=restore(resume,model,opt) if resume.exists() else (0,-1.)
        history_file=out/"epochs.json"
        history=json.loads(history_file.read_text())[:start] if history_file.exists() else []
        sample={k:v.numpy().copy() for k,v in model.trainable.items()} if start==0 else {}
        for epoch in range(start,cfg["epochs"]):
            model.training=True; started=time.monotonic(); totals={}
            for i,batch in enumerate(worker.batches("train",epoch,cfg["ntrain"])):
                x=image_tensor(batch)
                raw=model.raw[0][jt.array(batch["text_ids"])] if "text_ids" in batch else None
                vis=jt.array(batch["visuals"]).stop_grad() if "visuals" in batch else None
                out_heads=model.forward(x,raw=raw,visual=vis,pf_vocab_assignment=pf_vocab_assignment)
                loss,details=objective(out_heads,batch,stage,epoch,cfg["epochs"])
                numbers={k:float(v.item()) for k,v in details.items()}
                if not all(np.isfinite(v) for v in numbers.values()): raise FloatingPointError((stage,epoch,i,numbers))
                opt.backward(loss,epoch,i)
                for k,v in numbers.items(): totals[k]=totals.get(k,0.)+v
                if i<2 or (i+1)%100==0:
                    print(f"{stage} {epoch+1}/{cfg['epochs']} batch {i+1}/{n} loss {numbers} elapsed {time.monotonic()-started:.1f}s",flush=True)
            model.training=False
            online=model.state; model.state=opt.ema
            try: metrics=evaluate(model,worker,cfg,epoch_validation=True)
            finally: model.state=online
            fitness=metrics.get("fitness",metrics["metrics/mAP50-95(B)"])
            save_arrays(last_path,opt.ema)
            if fitness>=best:
                best=fitness; save_arrays(best_path,opt.ema)
            changed=sum(not np.array_equal(v,model.trainable[k].numpy()) for k,v in sample.items()) if sample else None
            if sample and changed==0: raise RuntimeError("No trainable tensor changed")
            history.append(dict(epoch=epoch+1,seconds=time.monotonic()-started,steps=opt.steps,changed_tensors=changed,
                train={k:v/n for k,v in totals.items()},metrics=metrics))
            save_json(history_file,history)
            checkpoint_resume(resume,model,opt,epoch+1,best)
            print(stage,history[-1],flush=True)
        del model,opt
        model=YOLOE26(args.init,stage,parent,best_path)
        metrics={"best":evaluate(model,worker,cfg)}
        if stage=="PF": metrics["coco80"]=evaluate(model,worker,cfg,True)
        save_json(verified,dict(stage=stage,epochs=len(history),steps=history[-1]["steps"],images=cfg["nval"],
            best_sha=digest(best_path),last_sha=digest(last_path),parent_sha=identity["parent_sha"],metrics=metrics))
        print("VERIFIED",stage,metrics,flush=True)
    finally:
        worker.close()


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("init","data","out"): p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--reference-python", help="Optional legacy CPU oracle for comparison")
    p.add_argument("--batch",type=int,default=8); p.add_argument("--workers",type=int,default=2)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--imgsz",type=int); p.add_argument("--verify-only",action="store_true")
    p.add_argument("--stages",choices=STAGES,nargs="+",default=list(STAGES))
    p.add_argument("--pf-p5-assignment", choices=("object", "tp-vocab"), default="object",
                   help="Score stride-32 one-to-one PF training assignments with the object head or the frozen TP vocabulary")
    p.add_argument("--pf-p5-bias-warmup-lr", type=float, default=0.1,
                   help="Initial warmup learning rate of the stride-32 one-to-one PF object bias")
    args=p.parse_args()
    args.data,args.init,args.out=(x.resolve() for x in (args.data,args.init,args.out))
    if args.reference_python:
        args.reference_python=os.path.abspath(args.reference_python)
    jt.flags.use_cuda=int(args.device == "cuda"); jt.set_global_seed(20261121)
    report={}
    for stage in args.stages:
        report[stage]=train_stage(args,stage,args.verify_only)
        jt.gc()
    if args.verify_only: save_json(args.out/"independent-validation.json",report)
