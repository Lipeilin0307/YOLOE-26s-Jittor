"""Materialize detection and instance segmentation views of a fixed COCO split."""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


# Polygon joining adapted from Ultralytics 8.4.216 (AGPL-3.0).
def min_index(arr1: np.ndarray, arr2: np.ndarray):
    """Find a pair of indexes with the shortest distance between two arrays of 2D points.

    Args:
        arr1 (np.ndarray): A NumPy array of shape (N, 2) representing N 2D points.
        arr2 (np.ndarray): A NumPy array of shape (M, 2) representing M 2D points.

    Returns:
        (tuple[int, int]): A tuple (idx1, idx2) where idx1 is the index in arr1 and idx2 is the index in arr2 of the
            pair with the shortest distance.
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)

def merge_multi_segment(segments: list[list]):
    """Merge multiple segments into one by connecting them at their closest points.

    This function connects the coordinates with the minimum distance between each segment with a thin line to merge all
    segments into one.

    Args:
        segments (list[list]): Original segmentations in COCO's JSON file. Each element is a list of coordinates, like
            [segmentation1, segmentation2,...].

    Returns:
        (list[np.ndarray]): A list of connected segments represented as NumPy arrays.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # Record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # Use two round to connect all the segments
    for k in range(2):
        # Forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # Middle segments have two indexes, reverse the segment and map its indexes into the reversed order
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = [len(segments[i]) - 1 - x for x in idx]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # Deal with the first segment and the last one
                if i in {0, len(idx_list) - 1}:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in {0, len(idx_list) - 1}:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


HERE = Path(__file__).resolve().parent


def read_list(path, expected, split):
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if len(lines) != expected or len(set(lines)) != expected:
        raise ValueError(f"{path} must contain {expected} distinct paths")
    if any(Path(p).parent.as_posix() != split or not Path(p).stem.isdecimal() for p in lines):
        raise ValueError(f"Invalid COCO image path in {path}")
    return lines


def write_labels(path, rows):
    path.write_text("".join(" ".join([str(k), *(f"{v:.8f}" for v in xy)]) + "\n"
                            for k, xy in rows), encoding="utf-8")


def materialize(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def build_views(args, work, out):
    report = {"profile": args.profile, "splits": {}}
    for split, count, manifest in [("train2017", 20000, "train2017-20k.txt"), ("val2017", 5000, "val2017-5k.txt")]:
        parent = read_list(args.split_dir / manifest, count, split)
        partition = "train" if split == "train2017" else "val"
        selected = (read_list(args.split_dir / f"{partition}200.txt", 200, split)
                    if args.profile == "smoke" else parent)
        if not set(selected).issubset(parent):
            raise ValueError(f"{split} smoke images outside the agreed manifest")
        coco = json.loads((args.coco_root / "annotations" / f"instances_{split}.json").read_bytes())
        images = {im["id"]: im for im in coco["images"]}
        category_names = [c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])]
        if partition == "train":
            names = category_names
        assert category_names == names and len(names) == 80
        classes = {c["id"]: names.index(c["name"]) for c in coco["categories"]}
        ids = {int(Path(p).stem) for p in selected}
        annotations = defaultdict(list)
        for ann in coco["annotations"]:
            if ann["image_id"] in ids and not ann["iscrowd"] and ann["area"] > 0:
                annotations[ann["image_id"]].append(ann)
        for task in ("detect", "segment"):
            for folder in ("images", "labels"):
                (work / task / folder / partition).mkdir(parents=True)
        boxes = masks = 0
        for relative in selected:
            image_id = int(Path(relative).stem)
            im = images[image_id]
            assert im["file_name"] == Path(relative).name
            source = next((p / relative for p in [*args.image_root, args.coco_root] if (p / relative).is_file()), None)
            if source is None:
                raise FileNotFoundError(relative)
            width, height = im["width"], im["height"]
            det, seg = [], []
            for ann in annotations[image_id]:
                k = classes[ann["category_id"]]
                x, y, w, h = ann["bbox"]
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(width, x + w), min(height, y + h)
                if x2 <= x1 or y2 <= y1:
                    continue
                det.append((k, [(x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height]))
                parts = [p for p in ann["segmentation"] if len(p) >= 6]
                if parts:
                    polygon = (np.concatenate(merge_multi_segment(parts)) if len(parts) > 1
                               else np.asarray(parts[0]).reshape(-1, 2))
                    seg.append((k, np.clip(polygon / [width, height], 0, 1).ravel()))
            for task, labels in (("detect", det), ("segment", seg)):
                target = work / task / "images" / partition / im["file_name"]
                materialize(source if task == "detect" else work / "detect" / "images" / partition / target.name, target)
                write_labels(work / task / "labels" / partition / (target.stem + ".txt"), labels)
            boxes += len(det)
            masks += len(seg)
        listing = "\n".join(selected) + "\n"
        (work / f"{partition}.txt").write_text(listing, encoding="utf-8")
        report["splits"][partition] = {"images": len(selected), "boxes": boxes, "masks": masks,
            "manifest_sha256": hashlib.sha256(listing.encode()).hexdigest(),
            "parent_manifest_sha256": hashlib.sha256(("\n".join(parent)+"\n").encode()).hexdigest()}
        print(partition, report["splits"][partition], flush=True)
        del coco
    for task in ("detect", "segment"):
        cfg = dict(path=str(out / task), train="images/train", val="images/val", names=names)
        (work / f"{task}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        if task == "detect":
            cfg["val"] = cfg["train"]  # visual exemplars come exclusively from training images
            (work / "reference.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (work / "split.json").write_text(json.dumps(report, indent=2) + "\n")


def prepare(args):
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"Use a fresh output directory: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=out.name + "-", dir=out.parent) as tmp:
        work = Path(tmp) / "views"
        build_views(args, work, out)
        work.rename(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "full"], required=True)
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, action="append", default=[])
    parser.add_argument("--split-dir", type=Path, default=HERE / "splits")
    parser.add_argument("--out", type=Path, required=True)
    prepare(parser.parse_args())
