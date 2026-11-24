"""Detect user-supplied text prompts using the native 11s detector and MobileCLIP-B(LT)."""
import argparse
import json
from pathlib import Path

import cv2
import jittor as jt
import numpy as np

from geometry import LetterBox
from yoloe11.detector import MobileCLIPBLT, YOLOE11TP, image_tensor, suppress


def predict(init, checkpoint, image, names, size=320, conf=.25):
    if not names or len(set(names))!=len(names) or any(not s.strip() for s in names):
        raise ValueError("Supply one or more distinct nonempty class names")
    if size<32 or size%32 or not 0<=conf<=1:
        raise ValueError("Choose a positive multiple of 32 and confidence in [0,1]")
    bgr=cv2.imread(str(image))
    if bgr is None:
        raise OSError(f"Cannot open {image}")
    h,w=bgr.shape[:2]
    ratio=size/max(h,w)
    nh,nw=round(h*ratio),round(w*ratio)
    left,top=(size-nw)//2,(size-nh)//2
    resized=LetterBox((size,size))(image=bgr)
    rgb=np.ascontiguousarray(resized[:,:,::-1].transpose(2,0,1)[None])
    encoder=MobileCLIPBLT(init/"mobileclip-blt.npz")
    embeddings=encoder.encode(names)
    del encoder
    model=YOLOE11TP(init,checkpoint=checkpoint)
    with jt.no_grad():
        rec=model.forward(image_tensor(rgb),jt.array(embeddings))
        boxes,scores=model.decoded(rec,size,size)
        rows=suppress(boxes,scores,conf=conf)[0]
    rows[:,[0,2]]=np.clip((rows[:,[0,2]]-left)*w/nw,0,w)
    rows[:,[1,3]]=np.clip((rows[:,[1,3]]-top)*h/nh,0,h)
    return [dict(label=names[int(row[5])],confidence=float(row[4]),xyxy=row[:4].tolist()) for row in rows]


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init",type=Path,required=True)
    p.add_argument("--weights",type=Path)
    p.add_argument("--image",type=Path,required=True)
    p.add_argument("--classes",nargs="+",required=True)
    p.add_argument("--size",type=int,default=320)
    p.add_argument("--conf",type=float,default=.25)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    a=p.parse_args()
    jt.flags.use_cuda=int(a.device=="cuda")
    print(json.dumps(predict(a.init,a.weights,a.image,a.classes,a.size,a.conf),ensure_ascii=False))
