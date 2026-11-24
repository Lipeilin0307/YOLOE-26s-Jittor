"""Open-vocabulary TP detection using a native checkpoint and arbitrary class names."""
import argparse
import json
import math
from pathlib import Path

import cv2
import jittor as jt
import numpy as np

from model import YOLOE26
from objective import detections
from text_encoder import MobileCLIP2


def predict(init, weights, text_model, image_path, names, size=640, confidence=.25):
    if not names or len(names) != len(set(names)) or size <= 0 or size % 32 or not 0 <= confidence <= 1:
        raise ValueError("Supply unique prompts, size divisible by 32, and confidence in [0,1]")
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read {image_path}")
    h, w = image.shape[:2]
    r = size / max(h,w)
    nh, nw = min(size,math.ceil(h*r)), min(size,math.ceil(w*r))
    im = cv2.resize(image,(nw,nh),interpolation=cv2.INTER_LINEAR)
    left,top = (size-nw)//2,(size-nh)//2
    im = cv2.copyMakeBorder(im,top,size-nh-top,left,size-nw-left,cv2.BORDER_CONSTANT,value=(114,114,114))
    rgb = np.ascontiguousarray(im[:,:,::-1].transpose(2,0,1)[None])
    encoder = MobileCLIP2(text_model)
    embeddings = encoder.encode(names)
    model = YOLOE26(init,"TP",checkpoint=weights)
    with jt.no_grad():
        pe = model.text(jt.array(embeddings))
        rec = model.forward(jt.array(rgb).float32()/255., pe=pe, both=False)["one2one"]
        rows = detections(rec,size,size)[0]
    rows = rows[rows[:,4] >= confidence].copy()
    rows[:,[0,2]] = np.clip((rows[:,[0,2]]-left)*w/nw,0,w)
    rows[:,[1,3]] = np.clip((rows[:,[1,3]]-top)*h/nh,0,h)
    return [dict(label=names[int(row[5])], confidence=float(row[4]), xyxy=row[:4].tolist())
            for row in rows]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("init","weights","text-model","image"):
        parser.add_argument("--"+key, required=True, type=Path)
    parser.add_argument("--classes", required=True, nargs="+")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=.25)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    jt.flags.use_cuda = int(args.device == "cuda")
    rows = predict(args.init,args.weights,args.text_model,args.image,args.classes,args.size,args.conf)
    print(json.dumps(dict(image=str(args.image),classes=args.classes,detections=rows),ensure_ascii=False))
