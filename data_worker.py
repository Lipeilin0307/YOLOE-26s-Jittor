"""Shared CPU augmentation and AP evaluator. No model is loaded in this process."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import wire


def main():
    output=os.fdopen(os.dup(sys.stdout.fileno()),"wb")
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.augment import LoadVisualPrompt
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.models.yolo.segment import SegmentationValidator
    from ultralytics.utils.torch_utils import init_seeds
    import numpy as np

    config=wire.read(sys.stdin.buffer)
    stage=config["stage"]; task="segment" if stage=="SEG" else "detect"
    root=Path(config["data"]); settings=dict(task=task,imgsz=config["size"],batch=config["batch"],
        workers=config["workers"],overlap_mask=False,device="cpu",rect=False,conf=0.001,plots=False,
        nms=False,save_json=False,data=str(root/(task+".yaml")),verbose=False)
    if config["smoke"]: settings.update(mosaic=0.,close_mosaic=0)
    elif stage=="TP": settings["close_mosaic"]=2
    torch.set_num_threads(2); init_seeds(20261121,deterministic=True)
    data=check_det_dataset(settings["data"])
    ids={v:k for k,v in data["names"].items()}
    loaders={}
    mosaic_closed=False

    def send(x):
        wire.write(output,x)

    while True:
        try: mode,epoch=wire.read(sys.stdin.buffer)
        except EOFError: return
        if mode=="close": return
        train=mode=="train"; ref=mode=="ref"; single=stage=="PF" and mode!="vocab"
        args=get_cfg(overrides={**settings,"single_cls":single})
        d=dict(data)
        if single: d.update(names={0:"object"},nc=1)
        if mode not in loaders:
            bs=2*config["batch"] if mode=="epoch_val" else config["batch"]
            dataset=build_yolo_dataset(args,d["train" if train or ref else "val"],bs,d,
                mode="train" if train else "val",rect=mode=="epoch_val",stride=32,multi_modal=train)
            if (train and stage=="VP") or ref: dataset.transforms.append(LoadVisualPrompt())
            loaders[mode]=build_dataloader(dataset,bs,0 if ref else config["workers"],shuffle=train,
                                            rank=-1,device=torch.device("cpu"))
        loader=loaders[mode]
        if train and not mosaic_closed and args.close_mosaic and epoch>=config["epochs"]-args.close_mosaic:
            loader.dataset.close_mosaic(hyp=args)
            if stage=="VP": loader.dataset.transforms.append(LoadVisualPrompt())
            loader.reset()
            mosaic_closed=True
        if not train and not ref:
            klass=SegmentationValidator if stage=="SEG" else DetectionValidator
            validator=klass(dataloader=loader,save_dir=Path(config["out"]),args=vars(args))
            validator.device=torch.device("cpu"); validator.data=d; validator.training=True
            validator.init_metrics(SimpleNamespace(names=d["names"],end2end=True))
        send((len(loader),len(loader.dataset)))
        for batch in loader:
            keys=("img","cls","bboxes","batch_idx","masks","visuals") if train or ref else ("img",)
            msg={k:batch[k].numpy() for k in keys if k in batch}
            if train and stage in ("TP","SEG"):
                msg["text_ids"]=np.array([[ids[t] for t in texts] for texts in batch["texts"]],np.int32)
            send(msg)
            if not train and not ref:
                pred=wire.read(sys.stdin.buffer)
                if isinstance(pred,tuple): pred=tuple(torch.from_numpy(x.copy()) for x in pred)
                else: pred=torch.from_numpy(pred.copy())
                validator.update_metrics(validator.postprocess(pred),validator.preprocess(batch))
        if not train and not ref:
            metrics={k:float(v) for k,v in validator.get_stats().items()}
            send(dict(metrics=metrics,seen=validator.seen))


if __name__=="__main__":
    main()
