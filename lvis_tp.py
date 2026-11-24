"""Evaluate verified YOLOE-26s TP weights on the authors' LVIS minival list."""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import wire


ASSETS = {
    "minival.txt": "4d3489ae1a59dd0e26237e60769180d67ef676a3643aabe983531228c8bd05b0",
    "minival.json": "02301f6ccd89d1ee3d35112cb57d000c3396f34e4073066c90b2c1fbf47b55ce",
    "lvis.yaml": "af2b30726df6bc3019dfef2b5b1d185b352fe2474e10f620d8e851b7e52cc291",
}
PROTOCOL = dict(size=640, batch=1, classes=1203, head="one2one", nms=False,
                confidence=0.001, max_det=1000, detections_per_category=10000,
                metrics="LVIS Fixed AP bbox", paper_e2e_ap=29.9)
JITTOR_CPU_ENV = {"CUDA_VISIBLE_DEVICES": "", "nvcc_path": "", "use_cuda": "0"}


def read_json(path):
    return json.loads(path.read_bytes())


def save_json(path, value):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def checkpoint(base, engine, weights=None):
    if weights is not None:
        return weights, digest(weights)
    if engine == "pytorch":
        directory = base / "runs/coco20k/TP"
        path = directory / "weights/best.pt"
        expected = read_json(directory / "verified.json")["best_sha256"]
    else:
        directory = base / "jittor/full/TP"
        path = directory / "best.npz"
        expected = read_json(directory / "verified.json")["best_sha"]
    if digest(path) != expected:
        raise ValueError(f"Verified {engine} TP best checkpoint has changed")
    return path, expected


def prepare(base, root, manifest_path, image_dir, native_text=False, text_cache=None, encoder=None, device="cuda:0"):
    import yaml
    annotation = root / "minival.json"
    if not annotation.exists():
        tmp = annotation.with_suffix(".tmp")
        with gzip.open(root / "minival.json.gz", "rb") as f:
            tmp.write_bytes(f.read())
        if digest(tmp) != ASSETS[annotation.name]:
            raise ValueError("LVIS annotation checksum mismatch")
        os.replace(tmp, annotation)
    if any(digest(root / name) != sha for name, sha in ASSETS.items()):
        raise ValueError("Benchmark inputs differ from the pinned author/GLIP release")
    gt = read_json(annotation)
    ids = [int(Path(line).stem) for line in (root / "minival.txt").read_text().splitlines()]
    images = {im["id"]: im for im in gt["images"]}
    if len(ids) != 4809 or len(images) != 4809 or set(ids) != set(images):
        raise ValueError("Authors' image list differs from LVIS minival ground truth")
    if sorted(c["id"] for c in gt["categories"]) != list(range(1,1204)):
        raise ValueError("Expected exactly the LVIS v1 1-based category IDs")
    records = [[i,images[i]["height"],images[i]["width"]] for i in ids]
    for i,_,_ in records:
        if not (image_dir / f"{i:012d}.jpg").is_file():
            raise FileNotFoundError(f"Missing minival image {i}")
    def coco_ids(split):
        return {int(Path(s).stem) for s in (base/"data/coco20k"/(split+".txt")).read_text().splitlines()}
    overlaps = {}
    historical_run = not native_text and text_cache is None
    if historical_run or (base/"data/coco20k/train.txt").is_file():
        train, val = coco_ids("train"), coco_ids("val")
        if len(train)!=20000 or len(val)!=5000 or set(ids) & train:
            raise ValueError("COCO split or training/evaluation overlap check failed")
        overlaps = dict(training_overlap=len(set(ids)&train), coco_checkpoint_validation_overlap=len(set(ids)&val))
    names = yaml.safe_load((root/"lvis.yaml").read_text())["names"]
    names = [names[i].split("/",1)[0] for i in range(1203)]
    manifest_path.parent.mkdir(exist_ok=True, parents=True)
    manifest = dict(protocol=PROTOCOL,asset_sha256=ASSETS,images=records,
                    names=names,annotations=len(gt["annotations"]),**overlaps)
    if text_cache is not None:
        load_text(text_cache)
        manifest.update(text_sha=digest(text_cache), text_backend="provided-cache")
    elif not native_text:
        import jittor as jt
        from text_encoder import MobileCLIP2
        weights = encoder or base / "work/yoloe-runtime/assets/mobileclip2_b.ts"
        jt.flags.use_cuda = int(device.startswith("cuda"))
        raw = MobileCLIP2(weights).encode(names)
        check_text(raw)
        text_path = manifest_path.with_name("text.npy")
        with text_path.with_suffix(".tmp").open("wb") as f:
            np.save(f,raw)
        os.replace(text_path.with_suffix(".tmp"),text_path)
        manifest.update(text_sha=digest(text_path), text_encoder_sha=digest(weights), text_backend="jittor-mobileclip2-b")
    save_json(manifest_path, manifest)


def check_text(raw):
    if raw.shape != (1, 1203, 512) or raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise ValueError("Expected finite float32 LVIS text embeddings of shape (1, 1203, 512)")
    return raw


def load_text(path):
    return check_text(np.load(path, allow_pickle=False))


def selected(manifest, smoke):
    records = manifest["images"]
    if smoke:
        return [records[i] for i in np.linspace(0,len(records)-1,12,dtype=int)]
    return records


def record_slice(records, bounds):
    if bounds is None:
        return records
    start, stop = bounds
    if not 0 <= start < stop <= len(records):
        raise ValueError(f"Invalid image range {bounds} for {len(records)} images")
    return records[start:stop]


def images(image_dir, records):
    # Both backends read the identical CPU pixels, using Ultralytics' validation resize and square padding.
    import cv2
    cv2.setNumThreads(1)
    for i,h,w in records:
        im = cv2.imread(str(image_dir/f"{i:012d}.jpg"))
        if im is None or im.shape[:2] != (h,w):
            raise ValueError(f"Unexpected decoded dimensions for {i}")
        size=PROTOCOL["size"]; ratio=size/max(h,w)
        nh,nw = min(size,math.ceil(h*ratio)), min(size,math.ceil(w*ratio))
        if (nh,nw)!=(h,w):
            im=cv2.resize(im,(nw,nh),interpolation=cv2.INTER_LINEAR)
        left,top=(size-nw)//2,(size-nh)//2
        im=cv2.copyMakeBorder(im,top,size-nh-top,left,size-nw-left,cv2.BORDER_CONSTANT,value=(114,114,114))
        rgb=np.ascontiguousarray(im[:,:,::-1].transpose(2,0,1)[None])
        wire.write(sys.stdout.buffer,(i,rgb,(nh/h,nw/w,left,top)))


def predictor(engine, weight, init_dir, manifest, text_path, device, native_text, encoder, fuse_text=False):
    raw = None if native_text else load_text(text_path)
    if engine=="pytorch":
        import torch
        from ultralytics.nn.tasks import load_checkpoint
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        model,_=load_checkpoint(weight,device=device)
        model=model.float().eval()
        head=model.model[-1]
        model.end2end = True
        if not head.end2end or head.is_fused:
            raise ValueError("Expected an unfused text-prompt end-to-end TP checkpoint")
        with torch.inference_mode():
            if native_text:
                from ultralytics.nn.text_model import MobileCLIPTS
                model.clip_model = MobileCLIPTS(torch.device(device), weight=str(encoder))
                pe = model.get_text_pe(manifest["names"], cache_clip_model=True)
            else:
                pe = head.get_tpe(torch.as_tensor(raw, device=device))
            model.set_classes(manifest["names"], pe)
        head.agnostic_nms = False
        head.max_det=PROTOCOL["max_det"]
        @torch.inference_mode()
        def predict(rgb):
            pred=model(torch.as_tensor(rgb.copy(),device=device).float()/255.)
            # Detection returns (pred, raw); segmentation returns ((pred, proto), raw).
            while isinstance(pred, tuple):
                pred = pred[0]
            if pred.ndim != 3 or pred.shape[0] != 1 or pred.shape[2] < 6:
                raise ValueError(f"Unexpected end-to-end predictions: {pred.shape}")
            return pred[0, :, :6].cpu().numpy()
        return predict
    if device == "cpu":
        os.environ.update(JITTOR_CPU_ENV)
    import jittor as jt
    # The A800 abort was reported by a native parallel operator compiler thread.
    # Set this before CUDA initialization or constructing any Jittor variables.
    jt.flags.use_parallel_op_compiler = 0
    from model import YOLOE26, normalized
    from objective import detections
    jt.flags.use_cuda=int(device.startswith("cuda"))
    model=YOLOE26(init_dir,"TP",checkpoint=weight)
    model.training=False
    with jt.no_grad():
        pe=model.text(jt.array(raw))
        if fuse_text:
            # This inference-only instance calls only the one-to-one branch. Folding
            # its fixed LVIS prompts into the final 1x1 convolutions avoids the
            # 512-channel per-image contrastive product in each feature pyramid level.
            frozen_pe = normalized(pe)
            for i in range(3):
                conv_weight, conv_bias = model.projected("one2one_", i, frozen_pe)
                prefix = f"{model.head}.one2one_cv3.{i}.2."
                model.state[prefix+"weight"] = conv_weight.sync()
                model.state[prefix+"bias"] = conv_bias.sync()
            model.fused = True
    def predict(rgb):
        with jt.no_grad():
            rec=model.forward(jt.array(rgb).float32()/255.,pe=pe,both=False)["one2one"]
            return detections(rec,640,640,max_det=PROTOCOL["max_det"])[0]
    return predict


def infer(args):
    base, engine, smoke = args.base, args.engine, args.smoke
    manifest=read_json(args.manifest)
    if manifest["asset_sha256"] != ASSETS or manifest["protocol"] != PROTOCOL:
        raise ValueError("Manifest does not match the pinned LVIS protocol")
    records=record_slice(selected(manifest,smoke), args.record_range)
    out=args.output/("smoke" if smoke else "full"); out.mkdir(exist_ok=True, parents=True)
    weight,sha=checkpoint(base,engine,args.weights)
    text_path = args.text_cache or args.manifest.with_name("text.npy")
    if args.native_text:
        encoder = args.text_encoder or base / "work/yoloe-runtime/assets/mobileclip2_b.ts"
        text_info = dict(text_backend="ultralytics-mobileclip2-b", text_encoder_file=encoder.name,
                         text_encoder_sha=digest(encoder))
    else:
        encoder = None
        text_sha = digest(text_path)
        if args.text_cache is None and text_sha != manifest.get("text_sha"):
            raise ValueError("Frozen text cache changed")
        text_info = dict(text_backend="provided-cache" if args.text_cache else manifest["text_backend"],
                         text_cache_file=text_path.name, text_sha=text_sha)
    if args.fuse_text:
        text_info["fused_text_head"] = True
    metadata = dict(engine=engine, best_sha=sha, weight_file=weight.name,
                    manifest_sha=digest(args.manifest), protocol=PROTOCOL, **text_info)
    metadata["inference_device"] = args.device
    if engine == "jittor":
        metadata["jittor_parallel_op_compiler"] = 0
        if args.cuda_fallback_signal:
            metadata["cuda_fallback_signal"] = args.cuda_fallback_signal
    if args.workers > 1:
        return infer_parallel(args, out, records, metadata)
    predict=predictor(engine,weight,args.init_dir,manifest,text_path,args.device,args.native_text,encoder,args.fuse_text)
    cmd=[sys.executable,str(Path(__file__).resolve()),"images","--manifest",str(args.manifest),"--images-dir",str(args.images_dir)]
    if smoke: cmd.append("--smoke")
    if args.record_range is not None:
        cmd.extend(["--record-range", *map(str, args.record_range)])
    env={**os.environ,"CUDA_VISIBLE_DEVICES":""}
    env.pop("LD_PRELOAD",None)
    rows=[]; counts=[]
    with subprocess.Popen(cmd,stdout=subprocess.PIPE,env=env) as worker:
        try:
            for step,(expected,h,w) in enumerate(records,1):
                i,rgb,(sy,sx,left,top)=wire.read(worker.stdout)
                if i!=expected: raise ValueError("Image stream changed")
                pred=predict(rgb)
                if pred.shape!=(1000,6) or not np.isfinite(pred).all():
                    raise ValueError(f"{engine} produced invalid one-to-one predictions on {i}")
                pred=pred[pred[:,4]>PROTOCOL["confidence"]].copy()
                if np.any((pred[:,5]<0)|(pred[:,5]>=1203)|(pred[:,5]!=np.floor(pred[:,5]))):
                    raise ValueError("Invalid LVIS class index")
                pred[:,[0,2]]=np.clip((pred[:,[0,2]]-left)/sx,0,w)
                pred[:,[1,3]]=np.clip((pred[:,[1,3]]-top)/sy,0,h)
                pred[:,2:4]-=pred[:,:2]
                counts.append(len(pred)); rows.append(pred)
                if step%250==0: print(engine, args.record_range or "", step, "/", len(records), flush=True)
            if worker.stdout.read(1) or worker.wait()!=0:
                raise RuntimeError("CPU image worker did not exit cleanly")
        finally:
            if worker.poll() is None: worker.terminate()
    save_predictions(out, metadata, np.concatenate(rows), np.asarray(counts), np.asarray([x[0] for x in records]))


def supervised_cuda_infer(args, argv):
    """Recover from a native CUDA-process abort without retrying validation errors."""
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), *argv]
    code = subprocess.run([*cmd, "--jittor-cuda-child"]).returncode
    if code == 0:
        return
    if code not in (-signal.SIGABRT, -signal.SIGSEGV):
        raise SystemExit(code if code > 0 else 128 - code)
    failure = signal.Signals(-code).name
    print(f"Jittor CUDA inference exited with {failure}. Retrying LVIS once on CPU.", file=sys.stderr, flush=True)
    # A shard is already a bounded range; only an unsharded request may fan out.
    workers = "3" if args.record_range is None else "1"
    cpu_env = {**os.environ, **JITTOR_CPU_ENV}
    subprocess.run([*cmd, "--device", "cpu", "--workers", workers, "--cuda-fallback-signal", failure],
                   env=cpu_env, check=True)


def save_predictions(out, metadata, rows, counts, ids):
    pred_path=out/(metadata["engine"]+".npz")
    with pred_path.with_suffix(".tmp").open("wb") as f:
        np.savez(f,rows=rows, counts=counts, ids=ids)
    os.replace(pred_path.with_suffix(".tmp"),pred_path)
    save_json(out/(metadata["engine"]+"-inference.json"),dict(**metadata, images=len(ids), detections=len(rows),
                                                             prediction_sha=digest(pred_path)))


def infer_parallel(args, out, records, metadata):
    """Run disjoint contiguous image ranges; the fixed-AP cap is applied after merging."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    workers = min(args.workers, len(records))
    ranges = [(k*len(records)//workers, (k+1)*len(records)//workers) for k in range(workers)]
    env = dict(os.environ, **JITTOR_CPU_ENV, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE")
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "infer", "--engine", "jittor", "--device", "cpu",
           "--base", str(args.base), "--manifest", str(args.manifest), "--images-dir", str(args.images_dir),
           "--init-dir", str(args.init_dir)]
    if args.weights: cmd.extend(["--weights", str(args.weights)])
    if args.text_cache: cmd.extend(["--text-cache", str(args.text_cache)])
    if args.fuse_text: cmd.append("--fuse-text")
    if args.smoke: cmd.append("--smoke")
    if args.cuda_fallback_signal: cmd.extend(["--cuda-fallback-signal", args.cuda_fallback_signal])
    with TemporaryDirectory(dir=out, prefix=".shards-") as tmp:
        dirs = [Path(tmp)/str(start) for start,_ in ranges]
        children = []
        with ThreadPoolExecutor(max_workers=workers) as waiter:
            try:
                for (start, stop), part in zip(ranges, dirs):
                    child = subprocess.Popen([*cmd, "--output", str(part), "--record-range", str(start), str(stop)], env=env)
                    children.append(child)
                waits = {waiter.submit(child.wait): region for child, region in zip(children,ranges)}
                for done in as_completed(waits):
                    if done.result():
                        raise RuntimeError(f"Jittor inference failed for range {waits[done]} (exit {done.result()})")
            finally:
                for child in children:
                    if child.poll() is None:
                        child.terminate()
                for child in children:
                    child.wait()
        row_groups, count_groups = [], []
        expected_ids = np.array([r[0] for r in records])
        if len(np.unique(expected_ids)) != len(expected_ids):
            raise ValueError("Duplicate images in the manifest")
        for part, (start,stop) in zip(dirs, ranges):
            folder = part/("smoke" if args.smoke else "full")
            info = read_json(folder/"jittor-inference.json")
            file = folder/"jittor.npz"
            if (any(info.get(k) != v for k,v in metadata.items()) or
                    bool(info.get("fused_text_head")) != args.fuse_text or digest(file) != info["prediction_sha"]):
                raise ValueError(f"Inference inputs differ in range {start}:{stop}")
            with np.load(file, allow_pickle=False) as data:
                ids, counts, rows = data["ids"], data["counts"], data["rows"]
            if (not np.array_equal(ids, expected_ids[start:stop]) or counts.shape != ids.shape or
                    counts.dtype.kind not in "iu" or np.any(counts<0) or np.any(counts>PROTOCOL["max_det"]) or
                    rows.shape != (counts.sum(), 6) or info["images"] != len(ids) or info["detections"] != len(rows)):
                raise ValueError(f"Incomplete inference in range {start}:{stop}")
            row_groups.append(rows)
            count_groups.append(counts)
        save_predictions(out, dict(metadata, cpu_workers=workers), np.concatenate(row_groups),
                         np.concatenate(count_groups), expected_ids)


def lvis_results(gt, dets):
    from lvis import LVISResults
    if dets:
        return LVISResults(gt, dets, max_dets=-1)
    # lvis 0.5.3 reads results[0] even for an empty prediction list.
    result = LVISResults.__new__(LVISResults)
    result.dataset = dict(gt.dataset, annotations=[])
    result.logger = gt.logger
    result._create_index()
    return result


def reference_ap(gt,dets,ids):
    from lvis import LVISEval
    ev=LVISEval(gt,lvis_results(gt,dets),iou_type="bbox")
    ev.params.img_ids=ids.tolist()
    ev.params.max_dets=-1
    ev.run()
    return {k:float(v)*100 for k,v in ev.results.items() if k.startswith("AP")}


def fixed_ap_in_batches(gt, ids, ranked, split, scores, rows, image_ids, category_batch):
    """Evaluate independently by category, then run LVIS's own summary on the joined precision array."""
    from copy import copy
    from collections import defaultdict
    from lvis import LVISEval

    image_set = set(ids.tolist())
    eligible = {i: set(gt.imgs[i]["neg_category_ids"]) for i in image_set}
    annotations = defaultdict(list)
    for ann in gt.dataset["annotations"]:
        if ann["image_id"] in image_set:
            eligible[ann["image_id"]].add(ann["category_id"])
            annotations[ann["category_id"]].append(ann)
    kept_counts = []
    eligible_count = 0
    precisions, recalls = [], []
    all_cats = sorted(gt.get_cat_ids())
    for start in range(0, len(all_cats), category_batch):
        cats = all_cats[start:start+category_batch]
        anns = [ann for c in cats for ann in annotations[c]]
        batch_ids = {a["image_id"] for a in anns}
        dets = []
        for c in cats:
            a, b = split[c-1:c+1]
            idx = ranked[a:min(b, a+PROTOCOL["detections_per_category"])]
            kept_counts.append(len(idx))
            for k in idx:
                i = int(image_ids[k])
                if c in eligible[i]:
                    batch_ids.add(i)
                    dets.append(dict(image_id=i, category_id=c, score=float(scores[k]),
                                     bbox=[round(float(v), 3) for v in rows[k, :4]]))
        eligible_count += len(dets)
        # Bbox evaluation uses no polygon masks. A category/image pair with neither
        # ground truth nor an eligible detection has no effect on precision or recall.
        part_gt = copy(gt)
        part_gt.dataset = dict(images=[gt.imgs[i] for i in sorted(batch_ids)], categories=gt.load_cats(cats),
                               annotations=[{k:v for k,v in ann.items() if k != "segmentation"} for ann in anns])
        part_gt._create_index()
        dt = lvis_results(part_gt, dets)
        ev = LVISEval(part_gt, dt, iou_type="bbox")
        ev.params.max_dets = -1
        ev.evaluate()
        ev.accumulate()
        precisions.append(ev.eval["precision"])
        recalls.append(ev.eval["recall"])
        del ev

    summary = LVISEval(gt, dt, iou_type="bbox")
    summary.params.max_dets = -1
    summary.freq_groups = summary._prepare_freq_group()
    summary.eval = dict(precision=np.concatenate(precisions, axis=2), recall=np.concatenate(recalls, axis=1))
    summary.summarize()
    metrics = {k:float(v)*100 for k,v in summary.results.items() if k.startswith("AP")}
    return metrics, eligible_count, kept_counts


def score(args):
    from lvis import LVIS
    base, engine, smoke = args.base, args.engine, args.smoke
    out=args.output/("smoke" if smoke else "full")
    manifest=read_json(args.manifest)
    records=selected(manifest,smoke)
    info=read_json(out/(engine+"-inference.json"))
    path=out/(engine+".npz")
    if (info["manifest_sha"]!=digest(args.manifest) or info["best_sha"]!=checkpoint(base,engine,args.weights)[1]
            or info["prediction_sha"]!=digest(path) or info["protocol"]!=PROTOCOL):
        raise ValueError("Inconsistent evaluation inputs")
    annotation = args.assets_dir / "minival.json"
    if digest(annotation) != ASSETS["minival.json"]:
        raise ValueError("LVIS annotation checksum mismatch")
    gt=LVIS(str(annotation))
    with np.load(path) as z:
        rows,counts,ids=z["rows"],z["counts"],z["ids"]
    if ids.tolist()!=[r[0] for r in records] or len(rows)!=sum(counts):
        raise ValueError("Incomplete predictions")
    image_ids=np.repeat(ids,counts)
    class_ids=rows[:,5].astype(np.int32)+1
    # Official predictions.json rounds scores before stable per-category ranking.
    scores=np.fromiter((round(float(s),5) for s in rows[:,4]),dtype=np.float64,count=len(rows))
    ranked=np.lexsort((-scores,class_ids))
    split=np.searchsorted(class_ids[ranked],np.arange(1,1205))
    metrics, eligible_count, kept_counts = fixed_ap_in_batches(
        gt, ids, ranked, split, scores, rows, image_ids, args.category_batch)
    if smoke:
        # Cross-check the compact ranking path against the authors' list/sort recipe.
        from collections import defaultdict
        groups=defaultdict(list)
        for i,(box_x,box_y,bw,bh,conf,c) in zip(image_ids.tolist(),rows.tolist()):
            cat=int(c)+1
            groups[cat].append(dict(image_id=i,category_id=cat,score=round(conf,5),
                                    bbox=[round(v,3) for v in (box_x,box_y,bw,bh)]))
        reference=[d for group in groups.values() for d in
                   sorted(group,key=lambda x:x["score"],reverse=True)[:10000]]
        expected=reference_ap(gt,reference,ids)
        for key,value in expected.items():
            np.testing.assert_allclose(metrics[key],value,rtol=0,atol=1e-10)
    if not all(np.isfinite(v) for v in metrics.values()) or metrics["AP"]<0:
        raise ValueError("Invalid LVIS Fixed AP")
    save_json(out/(engine+"-metrics.json"),dict(**info,metrics=metrics,metric_scale="0-100",
        eligible_detections=eligible_count,categories_below_fixed_cap=sum(k<10000 for k in kept_counts),
        author_sort_cross_checked=smoke))
    print(engine,metrics,flush=True)


def compare(args):
    smoke = args.smoke
    root=args.output/("smoke" if smoke else "full")
    pt,jt=(read_json(root/(name+"-metrics.json")) for name in ("pytorch","jittor"))
    if pt["manifest_sha"]!=jt["manifest_sha"] or not (pt["images"]==jt["images"]==(12 if smoke else 4809)):
        raise ValueError("Frameworks used different images or vocabularies")
    m=read_json(args.manifest)
    if pt["manifest_sha"] != digest(args.manifest):
        raise ValueError("Comparison does not match the selected manifest")
    rows={k:dict(pytorch=v,jittor=jt["metrics"][k],delta=jt["metrics"][k]-v) for k,v in pt["metrics"].items()}
    provenance = {r["engine"]:{k:v for k,v in r.items() if k.startswith("text_") or k in ("weight_file", "fused_text_head")}
                  for r in (pt,jt)}
    save_json(root/"comparison.json",dict(smoke=smoke,images=pt["images"],protocol=PROTOCOL,
        train_image_overlap=m.get("training_overlap"),coco_val_image_overlap_full=m.get("coco_checkpoint_validation_overlap"),
        metrics=rows,best_sha={r["engine"]:r["best_sha"] for r in (pt,jt)},manifest_sha=pt["manifest_sha"],
        **({"inputs_by_engine":provenance} if any(provenance.values()) else {})))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["prepare","images","infer","score","compare"])
    parser.add_argument("--base",type=Path,default=Path.cwd())
    parser.add_argument("--engine",choices=["pytorch","jittor"])
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--assets-dir", type=Path, help="Directory containing the pinned LVIS annotation, image list and YAML")
    parser.add_argument("--manifest", type=Path, help="LVIS manifest input, or destination for prepare")
    parser.add_argument("--images-dir", type=Path, help="Directory holding the original COCO val2017 JPEGs")
    parser.add_argument("--weights", type=Path, help="Explicit model checkpoint; otherwise verify the experiment's TP best")
    parser.add_argument("--output", type=Path, help="Parent directory for smoke/ and full/ artifacts")
    parser.add_argument("--init-dir", type=Path, help="Jittor model specification and initialization directory")
    parser.add_argument("--device", choices=["cpu","cuda","cuda:0"], default="cuda:0")
    text_group = parser.add_mutually_exclusive_group()
    text_group.add_argument("--native-text", action="store_true", help="Use the Ultralytics MobileCLIP2 text encoder for PyTorch")
    text_group.add_argument("--text-cache", type=Path, help="Explicit float32 raw LVIS text cache in manifest class order")
    parser.add_argument("--text-encoder", type=Path, help="MobileCLIP2 TorchScript weights")
    parser.add_argument("--category-batch", type=int, default=16, help="LVIS categories scored in memory at once")
    parser.add_argument("--fuse-text", action="store_true", help="Fold fixed LVIS prompts into Jittor's inference head")
    parser.add_argument("--workers", type=int, default=1, help="Independent single-thread CPU Jittor inference processes")
    parser.add_argument("--record-range", nargs=2, type=int, help=argparse.SUPPRESS)
    parser.add_argument("--jittor-cuda-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cuda-fallback-signal", choices=["SIGABRT","SIGSEGV"], help=argparse.SUPPRESS)
    args=parser.parse_args(); base=args.base.resolve()
    args.base = base
    args.assets_dir = args.assets_dir or base/"lvis/inputs"
    args.manifest = args.manifest or args.assets_dir/"manifest.json"
    args.images_dir = args.images_dir or base/"coco/val2017"
    args.init_dir = args.init_dir or base/"jittor/init"
    args.output = args.output or base/"lvis"
    if args.category_batch < 1:
        parser.error("--category-batch must be positive")
    if args.native_text and args.engine == "jittor":
        parser.error("Jittor inference takes a prepared or explicit raw text cache")
    if args.fuse_text and (args.action != "infer" or args.engine != "jittor"):
        parser.error("--fuse-text applies to Jittor inference")
    if args.workers < 1 or args.workers > 1 and (args.action != "infer" or args.engine != "jittor"
                                                or args.device != "cpu" or args.record_range is not None):
        parser.error("--workers must be positive; parallel inference is available for CPU Jittor")
    if args.action in ("infer","score") and args.engine is None:
        parser.error("--engine is required for inference and scoring")
    if args.action=="prepare": prepare(base,args.assets_dir,args.manifest,args.images_dir,args.native_text,
                                       args.text_cache,args.text_encoder,args.device)
    elif args.action=="images": images(args.images_dir,record_slice(selected(read_json(args.manifest),args.smoke),args.record_range))
    elif args.action=="infer":
        if args.engine == "jittor" and args.device.startswith("cuda") and not args.jittor_cuda_child:
            supervised_cuda_infer(args, sys.argv[1:])
        else:
            infer(args)
    elif args.action=="score": score(args)
    else: compare(args)
