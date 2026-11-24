"""Compare dual-head losses, derivatives and optimizer steps on identical nonempty targets."""
import argparse
import json
from pathlib import Path
import numpy as np


def inputs(stage):
    rng=np.random.default_rng(823); n=189; nc=1 if stage=="PF" else 3
    batch=dict(img=np.zeros((2,3,96,96),np.uint8),cls=np.array([[0],[1],[2]],np.float32)%nc,
        batch_idx=np.array([0,0,1]),bboxes=np.array([[.4,.46,.42,.45],[.54,.57,.27,.31],[.65,.44,.08,.1]],np.float32))
    batch["masks"]=np.zeros((3,24,24),np.uint8)
    for i,(cx,cy,w,h) in enumerate(batch["bboxes"]):
        batch["masks"][i,int((cy-h/2)*24):int((cy+h/2)*24),int((cx-w/2)*24):int((cx+w/2)*24)]=1
    outputs={}
    for name in ("one2many","one2one"):
        rec={"boxes":rng.uniform(.3,3.2,(2,4,n)).astype("float32"),
             "scores":rng.normal(-3,1,(2,nc,n)).astype("float32")}
        if stage=="SEG":
            rec.update(proto=rng.normal(0,.1,(2,32,24,24)).astype("float32"),
                       mask_coefficient=rng.normal(0,.1,(2,32,n)).astype("float32"))
        outputs[name]=rec
    return batch,outputs


def torch_fixture(stage):
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import YOLOEModel,YOLOESegModel
    from ultralytics.utils.loss import E2ELoss,TVPDetectLoss,v8DetectionLoss,v8SegmentationLoss
    b,raw=inputs(stage)
    model=YOLOESegModel("yoloe-26s-seg.yaml",nc=3,verbose=False) if stage=="SEG" else YOLOEModel("yoloe-26s.yaml",nc=1 if stage=="PF" else 3,verbose=False)
    model.args=get_cfg(overrides={"epochs":30,"overlap_mask":False})
    criterion=E2ELoss(model,{"SEG":v8SegmentationLoss,"VP":TVPDetectLoss}.get(stage,v8DetectionLoss))
    tensors={}; outputs={}
    for branch,rec in raw.items():
        outputs[branch]={}
        for k,v in rec.items():
            x=torch.from_numpy(v).requires_grad_()
            tensors[branch+"."+k]=x; outputs[branch][k]=x
        outputs[branch]["feats"]=[torch.zeros((2,1,d,d)) for d in (12,6,3)]
    total,_=criterion(outputs,{k:torch.from_numpy(v) for k,v in b.items()})
    total=total.sum(); total.backward()
    return {"loss":np.array([total.item()]),**{k:(v.grad.numpy() if v.grad is not None else np.zeros(v.shape,np.float32)) for k,v in tensors.items()}}


def jittor_fixture(stage):
    import jittor as jt
    from objective import objective
    batch,raw=inputs(stage); tensors={}; outputs={}
    for branch,rec in raw.items():
        outputs[branch]={}
        for k,v in rec.items():
            x=jt.array(v).start_grad()
            tensors[branch+"."+k]=x; outputs[branch][k]=x
    loss,_=objective(outputs,batch,stage,0,30)
    grads=jt.grad(loss,list(tensors.values()))
    return {"loss":np.array([loss.item()]),**{k:g.numpy() for k,g in zip(tensors,grads)}}


def optimizer_fixture(engine,stage):
    rng=np.random.default_rng(109)
    a=rng.normal(0,.15,(6,4,3,3)).astype("float32")
    g=rng.normal(0,.2,a.shape).astype("float32")
    mu=stage=="TP"; k="model.1.conv.weight"
    if engine=="torch":
        import torch
        from ultralytics.optim.muon import MuSGD
        p=torch.nn.Parameter(torch.from_numpy(a.copy()))
        opt=(MuSGD([{"params":[p],"use_muon":True}],lr=.00125,momentum=.9,nesterov=True,
                   weight_decay=.0007,muon=.2,sgd=1.) if mu else
             torch.optim.AdamW([p],lr=.002,betas=(.9,.999),weight_decay=.025))
        for _ in range(3):
            p.grad=torch.from_numpy(g.copy()); opt.step()
        return p.detach().numpy()-a
    import jittor as jt
    from types import SimpleNamespace
    from optimizer import Optimizer
    p=jt.array(a).start_grad(); group={k:p}
    opt=Optimizer(SimpleNamespace(stage=stage,trainable=group,state=group,spec={}),1,8,3,True)
    for i in range(3): opt.backward((p*jt.array(g).stop_grad()).sum(),0,i)
    return p.numpy()-a


def check_resume(directory):
    """Resume across an epoch boundary with two of eight gradients accumulated."""
    import tempfile
    from types import SimpleNamespace
    import jittor as jt
    from optimizer import Optimizer
    from train import checkpoint_resume,restore
    def fresh():
        p=jt.array(np.array([0.25,-0.5],np.float32)).start_grad()
        params={"model.0.weight":p}
        net=SimpleNamespace(stage="PF",trainable=params,state=params,spec={})
        opt=Optimizer(net,10,8,10,False)
        opt.last=39
        return p,net,opt
    def epoch(p,opt,e):
        for i in range(10):
            opt.backward((p*p*(1+i/10)).sum(),e,i)
    p,net,opt=fresh()
    epoch(p,opt,4)
    assert opt.steps==1 and opt.acc
    with tempfile.TemporaryDirectory(dir=directory) as scratch:
        path=Path(scratch)/"resume.npz"
        checkpoint_resume(path,net,opt,5,0.7)
        q,net2,opt2=fresh()
        assert restore(path,net2,opt2)==(5,0.7)
    epoch(p,opt,5); epoch(q,opt2,5)
    assert opt.steps==opt2.steps==2 and opt.last==opt2.last
    pairs=[(p,q)]
    for name in ("ema","vel","second","acc"):
        first,second=getattr(opt,name),getattr(opt2,name)
        pairs.extend((v,second[k]) for k,v in first.items())
    for a,b in pairs: np.testing.assert_array_equal(a.numpy(),b.numpy())
    return 0.


def check_accumulation_schedule():
    from types import SimpleNamespace
    import jittor as jt
    from optimizer import Optimizer

    # Ultralytics ramps to the rounded nominal count, capped by total training batches.
    for batch, epochs, batches, expected in (
        (24, 4, 6, [0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21]),
        (4, 1, 1, [0]),
    ):
        param = jt.array([0.25]).start_grad()
        group = {"model.0.weight": param}
        optimizer = Optimizer(SimpleNamespace(stage="PF", state=group, trainable=group, spec={}),
                              epochs, batch, batches, False)
        observed = []
        for i in range(epochs * batches):
            optimizer.backward((param * param).sum(), i // batches, i % batches)
            if optimizer.last == i:
                observed.append(i)
        assert observed == expected, (observed, expected)
    return 0.


def check_bfloat16_rounding():
    import jittor as jt
    from optimizer import rounded_bfloat16

    # Both halfway cases, a value just above halfway, a signed zero and non-finites.
    x = np.array([1.00390625, 1.01171875, 1.004, -1.01171875, -0., np.inf, -np.inf, np.nan], np.float32)
    expected = np.array([1., 1.015625, 1.0078125, -1.015625, -0., np.inf, -np.inf, np.nan], np.float32)
    actual = rounded_bfloat16(jt.array(x)).numpy()
    np.testing.assert_array_equal(actual, expected)
    assert np.signbit(actual[4])
    return 0.


def check_muon_buckets():
    from optimizer import muon_transposes

    shapes = {"model.0.weight": (128, 64, 1, 1), "model.1.weight": (64, 64, 1, 1),
              "model.2.weight": (4, 64, 1, 1), "model.23.cv3.0.weight": (64, 64, 1, 1)}
    flips = muon_transposes({k:np.empty(s, np.float32) for k,s in shapes.items()}, {})
    # Two compatible weight matrices share direction. Large row-ratio and head LR split the others.
    assert flips == {"model.0.weight":True, "model.1.weight":True, "model.2.weight":False,
                     "model.23.cv3.0.weight":False}
    return 0.


def check_pf_assignment_gradient():
    """Only the target location may come from the teacher; BCE must use student logits."""
    import jittor as jt
    from objective import anchors, branch_loss

    h = w = 128
    centers, strides = anchors(h, w)
    # Give every anchor the same target box so only the assignment scores break the tie.
    lo, hi = np.array([24, 24], np.float32), np.array([104, 104], np.float32)
    distances = np.concatenate((centers - lo / strides, hi / strides - centers), axis=1)
    boxes = jt.array(distances.T[None]).stop_grad()
    batch = dict(img=np.empty((1, 3, h, w), np.uint8), cls=np.zeros((1, 1), np.float32),
                 batch_idx=np.zeros(1, np.int32), bboxes=np.array([[.5, .5, .625, .625]],np.float32))
    candidates = np.flatnonzero((strides[:, 0] == 32) & ((centers * strides > lo) & (centers * strides < hi)).all(1))
    first, second = candidates[:2]
    student = np.full((1, 1, len(centers)), -3., np.float32)
    student[0, 0, first] = 1.
    ranks = student.copy()
    ranks[0, 0, second] = 4.
    errors = []
    for scores, selected in ((None, first), (ranks, second)):
        logits = jt.array(student).start_grad()
        record = dict(boxes=boxes, scores=logits)
        if scores is not None:
            record["assignment_scores"] = jt.array(scores).stop_grad()
        _, terms = branch_loss(record, batch, "PF", True)
        gradient = jt.grad(terms["cls"], logits).numpy()
        expected = 1 / (1 + np.exp(-student))
        expected[0, 0, selected] -= 1
        expected *= .5
        np.testing.assert_allclose(gradient, expected, atol=2e-6)
        errors.append(float(np.max(np.abs(gradient-expected))))
    return max(errors)


def check_pf_bias_warmup():
    from types import SimpleNamespace
    import jittor as jt
    from optimizer import Optimizer

    p5 = "model.23.one2one_cv3.2.2.bias"
    names = (p5, "model.23.one2one_cv3.1.2.bias", "model.23.cv3.2.2.bias")
    errors = []
    for rate in (0.1, 0.01):
        params = {key: jt.zeros((1,)).start_grad() for key in names}
        model = SimpleNamespace(stage="PF", trainable=params, state=params, spec={})
        opt = Optimizer(model, epochs=10, batch=8, batches=2500, smoke=False, pf_p5_bias_warmup_lr=rate)
        opt.backward(sum(v.sum() for v in params.values()), 0, 0)
        values = np.array([v.item() for v in params.values()])
        expected = np.array([-rate, -.1, -.1])
        np.testing.assert_allclose(values, expected, atol=2e-6)
        errors.append(float(np.max(np.abs(values-expected))))
    return max(errors)


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("engine",choices=["torch","jittor"]); p.add_argument("out",type=Path)
    p.add_argument("--device", choices=["cpu","cuda"], default="cuda", help="Jittor device for the parity check")
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True); stats={}
    if args.engine=="jittor":
        import jittor as jt
        jt.flags.use_cuda=int(args.device=="cuda")
    else:
        import torch
        torch.set_num_threads(2)
    for stage in ("TP","VP","PF","SEG"):
        out=torch_fixture(stage) if args.engine=="torch" else jittor_fixture(stage)
        if stage in ("TP","PF"): out["optimizer"]=optimizer_fixture(args.engine,stage)
        path=args.out/(stage+"-math.npz")
        if args.engine=="torch": np.savez(path,**out); continue
        with np.load(path) as ref:
            for k,x in out.items():
                scale=3e-5 if k=="optimizer" else 3e-4
                delta=float(np.max(np.abs(x-ref[k])))
                print(stage,k,delta,flush=True)
                np.testing.assert_allclose(x,ref[k],rtol=.002,atol=scale)
                stats[stage+"."+k]=delta
    if args.engine=="jittor":
        stats["accumulation_resume"]=check_resume(args.out)
        stats["accumulation_schedule"]=check_accumulation_schedule()
        stats["bfloat16_rounding"]=check_bfloat16_rounding()
        stats["muon_buckets"]=check_muon_buckets()
        stats["pf_assignment_gradient"]=check_pf_assignment_gradient()
        stats["pf_p5_bias_warmup"]=check_pf_bias_warmup()
        (args.out/"math-parity.json").write_text(json.dumps(stats,indent=2))
