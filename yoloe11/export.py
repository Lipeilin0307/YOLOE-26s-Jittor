"""Convert the official unfused YOLOE-11s TP checkpoint and generate independent oracle fixtures."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import xywh2xyxy
from export import graph

DETECT_SHA = "8e439445c87338b79d9ce21dec109f4621e26df67e94d26ea1a98c1e64dce3e3"
TEXT_SHA = "a67804d1b0f07b8b9a20c1761ec0847f34660f5fa338ec70e8f3fce68ed95e54"


def check_asset(path, expected):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"Unexpected release asset: {path}")


@torch.inference_mode()
def export(args):
    import clip
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    src = args.assets / "yoloe-11s-seg.pt"
    text_src = args.assets / "mobileclip_blt.ts"
    check_asset(src, DETECT_SHA)
    check_asset(text_src, TEXT_SHA)
    torch.set_num_threads(2)
    encoder = torch.jit.load(str(text_src), map_location="cpu").eval()
    state = encoder.text_encoder.state_dict()
    np.savez(out / "mobileclip-blt.npz", **{k:v.numpy() for k,v in state.items()})
    names = yaml.safe_load((args.data/"detect.yaml").read_bytes())["names"]
    names = [names[i] for i in range(len(names))]
    prompts = [["chair", "person", "potted plant"], ["potted plant", "armchair", "chair", "person"]]
    def encode(words):
        return torch.cat([encoder(t).float() for t in clip.tokenize(words, truncate=True).split(8)])[None]
    np.save(out/"coco-oracle.npy", encode(names).numpy())
    raw_prompts = [encode(words) for words in prompts]
    del encoder, state
    gc.collect()
    model, _ = load_checkpoint(src, device="cpu")
    model = model.float().eval()
    head = model.model[-1]
    if head.end2end or head.reg_max != 16 or head.is_fused:
        raise ValueError("Expected the official 16-bin, unfused 11s text prompt detector")
    prefix = f"model.{len(model.model)-1}"
    omit = tuple(prefix+"."+k+"." for k in ("proto","cv5","savpe","dfl"))
    params = {k:v.numpy() for k,v in model.state_dict().items() if not k.startswith(omit)}
    np.savez(out/"det.npz", **params)
    spec = graph(model)
    for node in spec.values():
        if node["kind"] == "SPPF":
            node.setdefault("n",3)
            node.setdefault("add",False)
    (out/"det.json").write_text(json.dumps(spec))
    ids = (args.data/"val.txt").read_text().splitlines()
    fixtures = []
    for i,(words,raw) in enumerate(zip(prompts,raw_prompts)):
        path = args.data/"detect/images/val"/Path(ids[i]).name
        bgr = cv2.imread(str(path))
        h,w = bgr.shape[:2]
        target = (256,320) if i else (320,320)
        ratio = min(target[0]/h,target[1]/w)
        nh,nw = round(h*ratio),round(w*ratio)
        rgb = np.full((*target,3),114,np.uint8)
        y,x = (target[0]-nh)//2,(target[1]-nw)//2
        rgb[y:y+nh,x:x+nw] = cv2.resize(bgr,(nw,nh))[:,:,::-1]
        image = torch.from_numpy(rgb.transpose(2,0,1).copy())[None].float()/255
        head.nc = len(words)
        pred,rec = model.predict(image,tpe=raw)
        if isinstance(pred,tuple):
            pred = pred[0]
        objects = pred[:,:4+len(words)].clone()
        boxes = xywh2xyxy(objects[:,:4].transpose(1,2)).numpy()
        kept = non_max_suppression(objects.clone(),conf_thres=.001,iou_thres=.7,nc=len(words),
                                   multi_label=True,max_det=300)[0]
        if not len(kept):
            raise ValueError("Empty reference fixture")
        np.savez(out/f"oracle-{i}.npz",image=image.numpy(),raw=raw.numpy(),pe=head.get_tpe(raw).numpy(),
                 distributions=rec["boxes"].numpy(),logits=rec["scores"].numpy(),xyxy=boxes,
                 nms=kept.numpy(),**{f"p{j+3}":a.numpy() for j,a in enumerate(rec["feats"][:3])})
        fixtures.append(dict(image=path.name,classes=words))
    metadata = dict(model="YOLOE-11s-TP",release="ultralytics/assets v8.4.0",checkpoint_sha=DETECT_SHA,
                    text_sha=TEXT_SHA,head=prefix,strides=head.stride.int().tolist(),reg_max=head.reg_max,
                    names=names,fixtures=fixtures)
    (out/"yoloe11.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print("Exported",out,flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("assets","out","data"):
        p.add_argument("--"+key,type=Path,required=True)
    export(p.parse_args())
