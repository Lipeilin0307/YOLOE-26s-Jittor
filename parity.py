import argparse
import json
from pathlib import Path
import numpy as np
import jittor as jt
from model import YOLOE26, retrieve_vocab


def check(init,device):
    jt.flags.use_cuda=int(device=="cuda")
    results={}
    rng=np.random.default_rng(95)
    z=rng.normal(size=(2,4,2,3)).astype("float32")
    weight=rng.normal(size=(7,4,1,1)).astype("float32")
    bias=rng.normal(size=(7,)).astype("float32")
    dense=np.einsum("oc,bcn->bon",weight[:,:,0,0],z.reshape(2,4,6))+bias[None,:,None]
    for label, obj in (("empty",np.full((2,6),-20.)),("full",np.ones((2,6))),
                       ("mixed",np.array([[2,-20,-20,2,-20,-20],[-20,-20,2,2,-20,-20]]))):
        expected=np.where(obj[:,None,:]>-6.90675,dense,-1e5)
        actual=retrieve_vocab(jt.array(z),jt.array(obj.astype("float32")),jt.array(weight),jt.array(bias)).numpy()
        np.testing.assert_allclose(actual,expected,atol=1e-5)
        results["PF.vocabulary."+label]=float(np.max(np.abs(actual-expected)))
    for name,stage in (("det","TP"),("seg","SEG")):
        model=YOLOE26(init,stage)
        with np.load(init/(name+"-oracle.npz")) as f, jt.no_grad():
            x=jt.array(f["image"]); visual=jt.array(f["visual"]); raw=jt.array(f["text"])
            feats=model.image_features(x)
            got={"tpe":model.text(raw),"vpe":model.visual(feats,visual)}
            for branch,rec in model.forward(x,raw=raw).items():
                got.update({branch+"."+k:v for k,v in rec.items()})
            for key,v in got.items():
                value=v.numpy(); ref=f[key]
                max_err=float(np.abs(value-ref).max())
                print(name,key,value.shape,max_err,flush=True)
                np.testing.assert_allclose(value,ref,rtol=0.002,atol=0.003)
                results[name+"."+key]=max_err
    return results


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--init",type=Path,required=True); p.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    p.add_argument("--report",type=Path,required=True)
    a=p.parse_args(); a.report.write_text(json.dumps(check(a.init,a.device),indent=2))
