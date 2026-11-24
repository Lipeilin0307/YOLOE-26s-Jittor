"""COCO manifest reader, training transforms and validation batches for the Jittor runner."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import math
from pathlib import Path
import random

import cv2
import numpy as np
from PIL import Image
import yaml

from geometry import (Instances, LetterBox, Mosaic, RandomFlip, RandomHSV, RandomLoadText,
                      RandomPerspective, polygons2masks, resample_segments, segments2boxes, xywh2xyxy)
from native_metrics import Evaluator


def read_targets(path, segmentation, single):
    boxes, classes, polygons = [], [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        values = np.asarray(line.split(), dtype=np.float32)
        classes.append(values[0])
        if segmentation:
            polygons.append(values[1:].reshape(-1, 2))
        else:
            boxes.append(values[1:].reshape(4))
    if segmentation:
        boxes = segments2boxes(polygons) if polygons else []
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    cls = np.asarray(classes, np.float32).reshape(-1, 1)
    if len(cls):
        _, keep = np.unique(np.concatenate((cls, boxes), axis=1), axis=0, return_index=True)
        if len(keep) < len(cls) and polygons:
            keys = np.array([c.tobytes() + p.tobytes() for c,p in zip(cls,polygons)], dtype=object)
            _, keep = np.unique(keys, return_index=True)
        if len(keep) < len(cls):
            cls, boxes = cls[keep], boxes[keep]
            if polygons:
                polygons = [polygons[i] for i in keep]
    if single:
        cls.fill(0)
    return cls, boxes, polygons


def box_grid(boxes, height, width):
    x1, y1, x2, y2 = boxes.T[:, :, None, None]
    x = np.arange(width)[None, None, :]
    y = np.arange(height)[None, :, None]
    return (x >= x1) & (x < x2) & (y >= y1) & (y < y2)


class CocoView:
    cache = None

    def __init__(self, cfg, split, single):
        self.size = cfg["size"]
        self.segment = cfg["stage"] == "SEG"
        task = "segment" if self.segment else "detect"
        root = Path(cfg["data"])
        manifest = root / (split + ".txt")
        if hashlib.sha256(manifest.read_text().encode()).hexdigest() != cfg[split + "_sha"]:
            raise ValueError(f"Changed image manifest {manifest}")
        self.names = yaml.safe_load((root / (task + ".yaml")).read_bytes())["names"]
        if isinstance(self.names, dict):
            self.names = [self.names[i] for i in range(len(self.names))]
        self.names = ["object"] if single else self.names
        self.ids = {name: i for i, name in enumerate(self.names)}
        self.texts = [[name] for name in self.names]
        filenames = [Path(line).name for line in manifest.read_text().splitlines()]
        self.paths = [root / task / "images" / split / name for name in filenames]
        self.labels = [read_targets(root / task / "labels" / split / (p.stem + ".txt"),
                                    self.segment, single) for p in self.paths]
        self.buffer = deque(maxlen=min(1000, len(filenames), cfg["batch"] * 8))
        self.rect_order = None
        self.mosaic = Mosaic(self, self.size)
        self.affine = RandomPerspective(size=(self.size, self.size))
        self.hsv = RandomHSV(.015, .7, .4)
        self.flip = RandomFlip(.5, "horizontal")
        self.letterbox = LetterBox((self.size, self.size), scaleup=False)
        self.text = RandomLoadText(max_samples=min(len(self.names), 80), padding=True,
                                   padding_value=self.names)

    def __len__(self):
        return len(self.paths)

    def rectangles(self, batch_size):
        if self.rect_order is None:
            ratios = []
            for path in self.paths:
                with Image.open(path) as image:
                    ratios.append(image.height / image.width)
            order = np.argsort(ratios)
            self.rect_order = order, np.asarray(ratios)[order]
        order, ratios = self.rect_order
        for start in range(0, len(order), batch_size):
            group = ratios[start:start + batch_size]
            relative = (group.max(), 1) if group.max() < 1 else (1, 1 / max(1, group.min()))
            shape = (np.ceil(np.array(relative) * self.size / 32 + .5).astype(int) * 32).tolist()
            yield order[start:start + batch_size], tuple(shape)

    def get_image_and_label(self, index):
        path = self.paths[index]
        image = cv2.imread(str(path))
        if image is None:
            raise OSError(f"Unable to decode {path}")
        h, w = image.shape[:2]
        r = self.size / max(h, w)
        if r != 1:
            image = cv2.resize(image, (min(self.size, math.ceil(w*r)), min(self.size, math.ceil(h*r))),
                               interpolation=cv2.INTER_LINEAR)
        cls, boxes, polygons = self.labels[index]
        max_len = max(map(len, polygons), default=0)
        n = max_len + 1 if max_len > 1000 else 1000
        segments = np.stack(resample_segments(list(polygons), n)) if polygons else np.zeros((0,n,2),np.float32)
        return dict(img=image, im_file=str(path), ori_shape=(h,w), resized_shape=image.shape[:2],
                    cls=cls.copy(), instances=Instances(boxes.copy(), segments, bbox_format="xywh", normalized=True),
                    texts=list(self.texts))

    def format(self, sample, train, mosaic, shape, stage, visual):
        if train:
            if mosaic:
                sample = self.mosaic(sample)
            for transform in (self.affine, self.hsv, self.flip, self.text):
                sample = transform(sample)
        else:
            sample["rect_shape"] = shape
            sample = self.letterbox(sample)
        im, instances = sample["img"], sample["instances"]
        h, w = im.shape[:2]
        instances.convert_bbox("xywh")
        instances.denormalize(w, h)
        result = dict(img=np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1)),
                      cls=sample["cls"].astype(np.float32).reshape(-1, 1))
        if self.segment:
            result["masks"] = (polygons2masks((h,w), instances.segments, 1, 4) if len(instances)
                               else np.zeros((0,h//4,w//4), np.uint8))
        instances.normalize(w, h)
        result["bboxes"] = instances.bboxes
        if train and stage in ("TP", "SEG"):
            result["text_ids"] = np.array([self.ids[t] for t in sample["texts"]], np.int32)
        if visual:
            boxes = xywh2xyxy(instances.bboxes) * np.array([w,h,w,h],np.float32) / 8
            unique, indices = np.unique(result["cls"].ravel(), return_inverse=True)
            masks = np.zeros((len(unique), h//8, w//8), np.float32)
            np.maximum.at(masks, indices, box_grid(boxes, h//8, w//8))
            result["visuals"] = masks
        return result


def collate(samples):
    batch = {k: np.stack([s[k] for s in samples]) for k in ("img","text_ids") if k in samples[0]}
    for key in ("cls", "bboxes", "masks"):
        if key in samples[0]:
            batch[key] = np.concatenate([s[key] for s in samples])
    batch["batch_idx"] = np.repeat(np.arange(len(samples)), [len(s["cls"]) for s in samples])
    if "visuals" in samples[0]:
        channels = max(1, max(len(s["visuals"]) for s in samples))
        batch["visuals"] = np.stack([np.pad(s["visuals"], ((0,channels-len(s["visuals"])),(0,0),(0,0)))
                                     for s in samples])
    return batch


class NativeWorker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.views = {}
        self.pool = ThreadPoolExecutor(cfg["workers"]) if cfg["workers"] else None
        self.evaluator = self.pending = None
        cv2.setNumThreads(1)

    def batches(self, mode, epoch, expected):
        cfg = self.cfg
        train = mode == "train"
        ref = mode == "ref"
        single = cfg["stage"] == "PF" and mode != "vocab"
        key = ("train" if train or ref else "val", single)
        if key not in self.views:
            self.views[key] = CocoView(cfg, *key)
        data = self.views[key]
        if len(data) != expected:
            raise ValueError(f"{mode} manifest contains {len(data)}, expected {expected}")
        size = cfg["batch"] * (2 if mode == "epoch_val" else 1)
        ids = np.arange(len(data))
        mosaic = train and not cfg["smoke"] and epoch < cfg["epochs"] - (2 if cfg["stage"] == "TP" else 10)
        if train:
            seed = cfg.get("seed", 20261121) + epoch
            random.seed(seed)
            np.random.seed(seed)
            np.random.shuffle(ids)
            data.buffer.clear()
        groups = (data.rectangles(size) if mode == "epoch_val" else
                  ((ids[i:i+size], (cfg["size"],cfg["size"])) for i in range(0,len(ids),size)))
        if not train and not ref:
            self.evaluator = Evaluator(single, data.segment)
        for indices, shape in groups:
            inputs = (self.pool.map(data.get_image_and_label, indices) if self.pool else
                      map(data.get_image_and_label, indices))
            samples = []
            for index, item in zip(indices, inputs):
                if train:
                    data.buffer.append(int(index))
                samples.append(data.format(item, train, mosaic, shape, cfg["stage"],
                                           ref or train and cfg["stage"] == "VP"))
            batch = collate(samples)
            if not train and not ref:
                self.pending = batch
            yield batch

    def send(self, predictions):
        self.evaluator.update(self.pending, predictions)
        self.pending = None

    def read(self):
        return self.evaluator.result()

    def close(self):
        if self.pool:
            self.pool.shutdown()
