"""Build native COCO embeddings and compare two real-image detections to the release oracle."""
import argparse
import gc
import json
from pathlib import Path

import jittor as jt
import numpy as np

from train import digest, save_json
from yoloe11.detector import MobileCLIPBLT, YOLOE11TP, suppress


def compare(actual, expected, atol, rtol=0):
    np.testing.assert_allclose(actual,expected,atol=atol,rtol=rtol)
    return float(np.max(np.abs(actual-expected)))


def run(init, report):
    meta=json.loads((init/"yoloe11.json").read_bytes())
    encoder=MobileCLIPBLT(init/"mobileclip-blt.npz")
    raw=encoder.encode(meta["names"],batch_size=4)
    text_error=compare(raw,np.load(init/"coco-oracle.npy"),2e-5)
    np.save(init/"raw-text.npy",raw)
    prompts=[encoder.encode(case["classes"],batch_size=4) for case in meta["fixtures"]]
    del encoder
    gc.collect(); jt.gc()
    model=YOLOE11TP(init)
    records=[]
    with jt.no_grad():
        for i,case in enumerate(meta["fixtures"]):
            with np.load(init/f"oracle-{i}.npz") as ref:
                image=jt.array(ref["image"])
                raw=prompts[i]
                pe=model.text(jt.array(raw))
                feats=model.image_features(image)
                rec=model.head_outputs(feats,pe)
                boxes, scores=model.decoded(rec,*image.shape[-2:])
                rows=suppress(boxes,scores)[0]
                checks=dict(text=compare(raw,ref["raw"],2e-5),reprta=compare(pe.numpy(),ref["pe"],2e-5))
                for j,feat in enumerate(feats):
                    checks[f"p{j+3}"]=compare(feat.numpy(),ref[f"p{j+3}"],.003,.001)
                checks["distribution_logits"]=compare(rec["boxes"].numpy(),ref["distributions"],.003,.0002)
                checks["class_logits"]=compare(rec["scores"].numpy(),ref["logits"],.003,.0002)
                checks["decoded_xyxy_pixels"]=compare(boxes,ref["xyxy"],.03)
                np.testing.assert_array_equal(rows[:,5],ref["nms"][:,5])
                checks["post_nms_xyxy_pixels"]=compare(rows[:,:4],ref["nms"][:,:4],.03)
                checks["post_nms_confidence"]=compare(rows[:,4],ref["nms"][:,4],2e-5)
            records.append(dict(**case,post_nms_detections=len(rows),max_abs_error=checks))
            print(records[-1],flush=True)
    save_json(report,dict(model=meta["model"],checkpoint_sha=meta["checkpoint_sha"],text_sha=meta["text_sha"],
                          init_sha=digest(init/"det.npz"),raw_sha=digest(init/"raw-text.npy"),
                          coco_embedding_max_abs=text_error,cases=records))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args=p.parse_args()
    jt.flags.use_cuda=int(args.device=="cuda")
    run(args.init,args.report)
