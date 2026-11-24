"""Compare the pretrained detector on the same 200 validation images in separate environments."""
import argparse
import json
from pathlib import Path

import numpy as np

from native_data import NativeWorker


def reference_predict(init, assets, weights=None):
    import torch
    from ultralytics.nn.tasks import load_checkpoint
    from ultralytics.utils.nms import non_max_suppression
    from yoloe11.export import check_asset, DETECT_SHA
    torch.set_num_threads(2)
    path=assets/"yoloe-11s-seg.pt"
    check_asset(path,DETECT_SHA)
    model,_=load_checkpoint(path,device="cpu")
    model=model.float().eval()
    if weights is not None:
        state=model.state_dict()
        with np.load(init/"det.npz") as original, np.load(weights) as updated:
            expected={k for k in original.files if not k.endswith("num_batches_tracked")}
            if set(updated.files)!=expected:
                raise ValueError("Checkpoint does not match the 11s TP detector")
            for key in updated.files:
                state[key].copy_(torch.from_numpy(updated[key]).reshape_as(state[key]))
    raw=torch.from_numpy(np.load(init/"coco-oracle.npy"))
    model.model[-1].nc=raw.shape[1]
    @torch.inference_mode()
    def predict(images):
        out,_=model.predict(torch.from_numpy(images).float()/255.,tpe=raw)
        out=out[0] if isinstance(out,tuple) else out
        kept=non_max_suppression(out[:,:4+raw.shape[1]].contiguous(),conf_thres=.001,iou_thres=.7,
                                 nc=raw.shape[1],multi_label=True,max_det=300)
        return [row.numpy() for row in kept]
    return predict


def native_predict(init, weights=None):
    import jittor as jt
    from yoloe11.detector import YOLOE11TP, image_tensor, suppress
    jt.flags.use_cuda=0
    model=YOLOE11TP(init,checkpoint=weights)
    with jt.no_grad():
        pe=model.text(model.raw)
    def predict(images):
        with jt.no_grad():
            rec=model.head_outputs(model.image_features(image_tensor(images)),pe)
            return suppress(*model.decoded(rec,*images.shape[-2:]))
    return predict


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("engine",choices=("reference","jittor"))
    for name in ("init","data","out"):
        p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--assets",type=Path)
    p.add_argument("--weights",type=Path,help="Optional native trained checkpoint")
    args=p.parse_args()
    if args.engine=="reference" and args.assets is None:
        p.error("reference requires --assets")
    predict=(reference_predict(args.init,args.assets,args.weights) if args.engine=="reference"
             else native_predict(args.init,args.weights))
    splits=json.loads((args.data/"split.json").read_bytes())["splits"]
    cfg=dict(stage="TP",data=str(args.data),size=320,batch=4,workers=2,epochs=1,smoke=True,
             val_sha=splits["val"]["manifest_sha256"])
    worker=NativeWorker(cfg)
    try:
        for batch in worker.batches("val",0,200):
            worker.send(predict(batch["img"]))
        record=worker.read()
        assert record["seen"]==200 and all(np.isfinite(x) for x in record["metrics"].values())
        record.update(engine=args.engine,val_sha=cfg["val_sha"],size=320,conf=.001,iou=.7)
        args.out.write_text(json.dumps(record,indent=2)+"\n")
        print(record,flush=True)
    finally:
        worker.close()
