"""Create the end-to-end comparison only after both standalone evaluations finish."""
import json
import argparse
from pathlib import Path


def compare(pytorch_summary,root):
    pt=json.loads(pytorch_summary.read_text())
    jt=json.loads((root/"independent-validation.json").read_text())
    rows={}
    for stage in ("TP","VP","PF","SEG"):
        cfg=json.loads((root/stage/"config.json").read_text())
        for s in ("train","val"):
            if cfg[s+"_sha"]!=pt["split"]["splits"][s]["manifest_sha256"]: raise ValueError("Mismatch: "+s)
        history=json.loads((root/stage/"epochs.json").read_text())
        if len(history)!=pt["stages"][stage]["epochs"]: raise ValueError("Epoch mismatch: "+stage)
        for name,result in pt["stages"][stage]["standalone_validation"].items():
            if result["images"]!=cfg["nval"]: raise ValueError("Validation size mismatch")
            m=jt[stage][name]
            rows[stage+"/"+name]={k:{"pytorch":v,"jittor":m[k],"difference":m[k]-v}
                                  for k,v in result["metrics"].items()}
    (root/"framework-comparison.json").write_text(json.dumps(rows,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("base",type=Path)
    parser.add_argument("--pytorch-summary",type=Path)
    parser.add_argument("--jittor-dir",type=Path)
    args=parser.parse_args()
    compare(args.pytorch_summary or args.base/"runs/coco20k/summary.json",
            args.jittor_dir or args.base/"jittor/full")
