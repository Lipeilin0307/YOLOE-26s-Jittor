# Adapted from Ultralytics 8.4.216, AGPL-3.0. See LICENSE.
"""NumPy/OpenCV geometry for the COCO detection and segmentation recipes."""
from __future__ import annotations
from collections import abc
from itertools import repeat
from numbers import Number
from typing import Any
import math
import random
import cv2
import numpy as np


def segment2box(segment: np.ndarray, width: int = 640, height: int = 640) -> np.ndarray:
    'Convert segment coordinates to bounding box coordinates.'
    if not len(segment):
        return np.zeros(4, dtype=segment.dtype)
    x, y = segment[:, 0], segment[:, 1]
    xmin, ymin, xmax, ymax = x.min(), y.min(), x.max(), y.max()
    if xmin >= 0 and ymin >= 0 and xmax <= width and ymax <= height:  # fully inside image
        return np.array([xmin, ymin, xmax, ymax], dtype=segment.dtype)
    if xmax < 0 or ymax < 0 or xmin > width or ymin > height:  # fully outside image
        return np.zeros(4, dtype=segment.dtype)
    axes = np.array((0, 0, 1, 1))
    bounds = np.array((0, width, 0, height), dtype=segment.dtype)
    lims = np.array((height, height, width, width), dtype=segment.dtype)  # (height, width)[axis] per boundary
    start, delta = segment, np.roll(segment, -1, axis=0) - segment
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (bounds - start[:, axes]) / delta[:, axes]
        inter = start[:, None, :] + t[:, :, None] * delta[:, None, :]
    other = inter[:, np.arange(4), 1 - axes]
    corners = np.array(((0, 0), (width, 0), (0, height), (width, height)), dtype=segment.dtype)
    contour = segment.astype(np.float32)
    points = np.concatenate(
        (
            segment[(x >= 0) & (y >= 0) & (x <= width) & (y <= height)],
            inter[(t >= 0) & (t <= 1) & (other >= 0) & (other <= lims)],
            corners[[cv2.pointPolygonTest(contour, tuple(map(float, p)), False) >= 0 for p in corners]],
        )
    )
    return (
        np.array([*points.min(0), *points.max(0)], dtype=segment.dtype)
        if len(points)
        else np.zeros(4, dtype=segment.dtype)
    )


def xyxy2xywh(x):
    'Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format.'
    if isinstance(x, (list, tuple)):
        x = np.asarray(x, dtype=np.float32)  # float so odd integer boxes keep fractional centers
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = np.empty_like(x, dtype=np.float32)  # faster than clone/copy
    x1, y1, x2, y2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    y[..., 0] = (x1 + x2) / 2  # x center
    y[..., 1] = (y1 + y2) / 2  # y center
    y[..., 2] = x2 - x1  # width
    y[..., 3] = y2 - y1  # height
    return y


def xywh2xyxy(x):
    'Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format.'
    if isinstance(x, (list, tuple)):
        x = np.asarray(x, dtype=np.float32)  # float so odd integer boxes keep fractional centers
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = np.empty_like(x, dtype=np.float32)  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y


def xywh2ltwh(x):
    'Convert bounding box format from [x, y, w, h] to [x1, y1, w, h] where x1, y1 are top-left coordinates.'
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    return y


def xyxy2ltwh(x):
    'Convert bounding boxes from [x1, y1, x2, y2] to [x1, y1, w, h] format.'
    y = np.copy(x)
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def ltwh2xywh(x):
    'Convert bounding boxes from [x1, y1, w, h] to [x, y, w, h] where xy1=top-left, xy=center.'
    y = np.copy(x)
    y[..., 0] = x[..., 0] + x[..., 2] / 2  # center x
    y[..., 1] = x[..., 1] + x[..., 3] / 2  # center y
    return y


def ltwh2xyxy(x):
    'Convert bounding box from [x1, y1, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right.'
    y = np.copy(x)
    y[..., 2] = x[..., 2] + x[..., 0]  # x2
    y[..., 3] = x[..., 3] + x[..., 1]  # y2
    return y


def segments2boxes(segments):
    'Convert segment coordinates to bounding box labels in xywh format.'
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # xyxy
    return xyxy2xywh(np.array(boxes).reshape(-1, 4))  # xywh


def resample_segments(segments, n: int = 1000):
    'Resample closed segments to n points each using linear interpolation.'
    for i, s in enumerate(segments):
        if len(s) == n:
            continue
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n - len(s) if len(s) < n else n)
        xp = np.arange(len(s))
        x = np.insert(x, np.searchsorted(x, xp), xp) if len(s) < n else x
        segments[i] = (
            np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)], dtype=np.float32).reshape(2, -1).T
        )  # segment xy
    return segments


def _ntuple(n):
    'Create a function that converts input to n-tuple by repeating singleton values.'
    def parse(x):
        'Parse input to return n-tuple by repeating singleton values n times.'
        return x if isinstance(x, abc.Iterable) else tuple(repeat(x, n))
    return parse
to_4tuple = _ntuple(4)
_formats = ["xyxy","xywh","ltwh"]


class Bboxes:
    'A class for handling bounding boxes in multiple formats.'

    def __init__(self, bboxes: np.ndarray, format: str = "xyxy") -> None:
        'Initialize the Bboxes class with bounding box data in a specified format.'
        assert format in _formats, f"Invalid bounding box format: {format}, format must be one of {_formats}"
        bboxes = bboxes[None, :] if bboxes.ndim == 1 else bboxes
        assert bboxes.ndim == 2
        assert bboxes.shape[1] == 4
        self.bboxes = bboxes
        self.format = format

    def convert(self, format: str) -> None:
        'Convert bounding box format from one type to another.'
        assert format in _formats, f"Invalid bounding box format: {format}, format must be one of {_formats}"
        if self.format == format:
            return
        elif self.format == "xyxy":
            func = xyxy2xywh if format == "xywh" else xyxy2ltwh
        elif self.format == "xywh":
            func = xywh2xyxy if format == "xyxy" else xywh2ltwh
        else:
            func = ltwh2xyxy if format == "xyxy" else ltwh2xywh
        self.bboxes = func(self.bboxes)
        self.format = format

    def areas(self) -> np.ndarray:
        'Calculate the area of bounding boxes.'
        return (
            (self.bboxes[:, 2] - self.bboxes[:, 0]) * (self.bboxes[:, 3] - self.bboxes[:, 1])  # format xyxy
            if self.format == "xyxy"
            else self.bboxes[:, 3] * self.bboxes[:, 2]  # format xywh or ltwh
        )

    def mul(self, scale: float | tuple | list) -> None:
        'Multiply bounding box coordinates by scale factor(s).'
        if isinstance(scale, Number):
            scale = to_4tuple(scale)
        assert isinstance(scale, (tuple, list))
        assert len(scale) == 4
        self.bboxes[:, 0] *= scale[0]
        self.bboxes[:, 1] *= scale[1]
        self.bboxes[:, 2] *= scale[2]
        self.bboxes[:, 3] *= scale[3]

    def add(self, offset: float | tuple | list) -> None:
        'Add offset to bounding box coordinates.'
        if isinstance(offset, Number):
            offset = to_4tuple(offset)
        assert isinstance(offset, (tuple, list))
        assert len(offset) == 4
        self.bboxes[:, 0] += offset[0]
        self.bboxes[:, 1] += offset[1]
        self.bboxes[:, 2] += offset[2]
        self.bboxes[:, 3] += offset[3]

    def __len__(self) -> int:
        'Return the number of bounding boxes.'
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, boxes_list: list[Bboxes], axis: int = 0) -> Bboxes:
        'Concatenate a list of Bboxes objects into a single Bboxes object.'
        assert isinstance(boxes_list, (list, tuple))
        if not boxes_list:
            return cls(np.empty((0, 4)))
        assert all(isinstance(box, Bboxes) for box in boxes_list)
        if len(boxes_list) == 1:
            return boxes_list[0]
        return cls(np.concatenate([b.bboxes for b in boxes_list], axis=axis), format=boxes_list[0].format)

    def __getitem__(self, index: int | np.ndarray | slice) -> Bboxes:
        'Retrieve a specific bounding box or a set of bounding boxes using indexing.'
        b = self.bboxes[index]
        if b.ndim == 1:
            b = b.reshape(1, -1)
        assert b.ndim == 2, f"Indexing on Bboxes with {index} failed to return a matrix!"
        return Bboxes(b, format=self.format)


class Instances:
    'Container for bounding boxes, segments, and keypoints of detected objects in an image.'

    def __init__(
        self,
        bboxes: np.ndarray,
        segments: np.ndarray | None = None,
        keypoints: np.ndarray | None = None,
        bbox_format: str = "xywh",
        normalized: bool = True,
    ) -> None:
        'Initialize the Instances object with bounding boxes, segments, and keypoints.'
        self._bboxes = Bboxes(bboxes=bboxes, format=bbox_format)
        self.keypoints = keypoints
        self.normalized = normalized
        self.segments = segments if segments is not None else np.zeros((0, 0, 2), dtype=np.float32)

    def convert_bbox(self, format: str) -> None:
        'Convert bounding box format.'
        self._bboxes.convert(format=format)

    @property
    def bbox_areas(self) -> np.ndarray:
        'Return the areas of the bounding boxes.'
        return self._bboxes.areas()

    def scale(self, scale_w: float, scale_h: float, bbox_only: bool = False):
        'Scale coordinates by given factors.'
        self._bboxes.mul(scale=(scale_w, scale_h, scale_w, scale_h))
        if bbox_only:
            return
        self.segments[..., 0] *= scale_w
        self.segments[..., 1] *= scale_h
        if self.keypoints is not None:
            self.keypoints[..., 0] *= scale_w
            self.keypoints[..., 1] *= scale_h

    def denormalize(self, w: int, h: int) -> None:
        'Convert normalized coordinates to absolute coordinates.'
        if not self.normalized:
            return
        self._bboxes.mul(scale=(w, h, w, h))
        self.segments[..., 0] *= w
        self.segments[..., 1] *= h
        if self.keypoints is not None:
            self.keypoints[..., 0] *= w
            self.keypoints[..., 1] *= h
        self.normalized = False

    def normalize(self, w: int, h: int) -> None:
        'Convert absolute coordinates to normalized coordinates.'
        if self.normalized:
            return
        self._bboxes.mul(scale=(1 / w, 1 / h, 1 / w, 1 / h))
        self.segments[..., 0] /= w
        self.segments[..., 1] /= h
        if self.keypoints is not None:
            self.keypoints[..., 0] /= w
            self.keypoints[..., 1] /= h
        self.normalized = True

    def add_padding(self, padw: int, padh: int) -> None:
        'Add padding to coordinates.'
        assert not self.normalized, "you should add padding with absolute coordinates."
        self._bboxes.add(offset=(padw, padh, padw, padh))
        self.segments[..., 0] += padw
        self.segments[..., 1] += padh
        if self.keypoints is not None:
            self.keypoints[..., 0] += padw
            self.keypoints[..., 1] += padh

    def __getitem__(self, index: int | np.ndarray | slice) -> Instances:
        'Retrieve a specific instance or a set of instances using indexing.'
        index = [index] if isinstance(index, (int, np.integer)) else index
        segments = self.segments[index] if len(self.segments) else self.segments
        keypoints = self.keypoints[index] if self.keypoints is not None else None
        bboxes = self.bboxes[index]
        bbox_format = self._bboxes.format
        return Instances(
            bboxes=bboxes,
            segments=segments,
            keypoints=keypoints,
            bbox_format=bbox_format,
            normalized=self.normalized,
        )

    def flipud(self, h: int) -> None:
        'Flip coordinates vertically.'
        if self._bboxes.format == "xyxy":
            y1 = self.bboxes[:, 1].copy()
            y2 = self.bboxes[:, 3].copy()
            self.bboxes[:, 1] = h - y2
            self.bboxes[:, 3] = h - y1
        elif self._bboxes.format == "ltwh":
            self.bboxes[:, 1] = h - self.bboxes[:, 1] - self.bboxes[:, 3]
        else:
            self.bboxes[:, 1] = h - self.bboxes[:, 1]
        self.segments[..., 1] = h - self.segments[..., 1]
        if self.keypoints is not None:
            self.keypoints[..., 1] = h - self.keypoints[..., 1]

    def fliplr(self, w: int) -> None:
        'Flip coordinates horizontally.'
        if self._bboxes.format == "xyxy":
            x1 = self.bboxes[:, 0].copy()
            x2 = self.bboxes[:, 2].copy()
            self.bboxes[:, 0] = w - x2
            self.bboxes[:, 2] = w - x1
        elif self._bboxes.format == "ltwh":
            self.bboxes[:, 0] = w - self.bboxes[:, 0] - self.bboxes[:, 2]
        else:
            self.bboxes[:, 0] = w - self.bboxes[:, 0]
        self.segments[..., 0] = w - self.segments[..., 0]
        if self.keypoints is not None:
            self.keypoints[..., 0] = w - self.keypoints[..., 0]

    def clip(self, w: int, h: int, preserve_obb: bool = False) -> None:
        'Clip coordinates to stay within image boundaries.'
        ori_format = self._bboxes.format
        self.convert_bbox(format="xyxy")
        if preserve_obb and len(self.segments):
            segment_mins, segment_maxs = self.segments.min(1), self.segments.max(1)
            inside = ((segment_mins >= 0) & (segment_maxs <= (w, h))).all(1)
            if not inside.all():
                canvas = np.array(((0, 0), (w, 0), (w, h), (0, h)), dtype=np.float32)
                bboxes, segments = self.bboxes.copy(), self.segments.copy()
                for i in np.flatnonzero(~inside):
                    segment = self.segments[i]
                    visible_area, visible = cv2.intersectConvexConvex(
                        cv2.convexHull(segment.astype(np.float32)), canvas
                    )
                    if visible is None or visible_area <= 0:
                        bboxes[i], segments[i] = 0, 0
                        continue
                    visible = visible.reshape(-1, 2)
                    bboxes[i] = np.array((*visible.min(0), *visible.max(0)), dtype=segment.dtype)
                    # Fit the visible polygon in the full box's coordinates so edge crops cannot change its angle.
                    (_, _), (box_w, box_h), angle = cv2.minAreaRect(segment.astype(np.float32))
                    if box_w < box_h:
                        angle += 90
                    angle = np.deg2rad(angle)
                    cos, sin = np.cos(angle), np.sin(angle)
                    basis = np.array(((cos, -sin), (sin, cos)), dtype=np.float32)
                    aligned = visible @ basis
                    (u1, v1), (u2, v2) = aligned.min(0), aligned.max(0)
                    corners = np.array(((u2, v2), (u2, v1), (u1, v1), (u1, v2)), dtype=np.float32)
                    segments[i] = resample_segments([corners @ basis.T], n=len(segment))[0]
                self.update(bboxes, segments)
        else:
            self.bboxes[:, [0, 2]] = self.bboxes[:, [0, 2]].clip(0, w)
            self.bboxes[:, [1, 3]] = self.bboxes[:, [1, 3]].clip(0, h)
            self.segments[..., 0] = self.segments[..., 0].clip(0, w)
            self.segments[..., 1] = self.segments[..., 1].clip(0, h)
        if ori_format != "xyxy":
            self.convert_bbox(format=ori_format)
        if self.keypoints is not None:
            # Set out of bounds visibility to zero
            self.keypoints[..., 2][
                (self.keypoints[..., 0] < 0)
                | (self.keypoints[..., 0] > w)
                | (self.keypoints[..., 1] < 0)
                | (self.keypoints[..., 1] > h)
            ] = 0.0
            self.keypoints[..., 0] = self.keypoints[..., 0].clip(0, w)
            self.keypoints[..., 1] = self.keypoints[..., 1].clip(0, h)

    def crop(self, window: tuple[int, int, int, int], preserve_obb: bool = False) -> np.ndarray:
        'Crop pixel-space instances and return the mask retaining at least 10% of each original box area.'
        x1, y1, x2, y2 = window
        w, h = x2 - x1, y2 - y1
        areas = self.bbox_areas.copy()
        self.convert_bbox("xyxy")
        self.add_padding(-x1, -y1)
        if len(self.segments) and not preserve_obb:
            outside = ((self.segments < 0) | (self.segments > (w, h))).any((1, 2))
            for i in np.flatnonzero(outside):
                self.bboxes[i] = box = segment2box(self.segments[i], w, h)
                self.segments[i] = self.segments[i].clip(box[:2], box[2:])
        self.clip(w, h, preserve_obb=preserve_obb)
        return (self.bbox_areas > 0) & (self.bbox_areas >= areas * 0.1)

    def remove_zero_area_boxes(self) -> np.ndarray:
        'Remove zero-area boxes, i.e. after clipping some boxes may have zero width or height.'
        good = self.bbox_areas > 0
        if not all(good):
            self._bboxes = self._bboxes[good]
            if self.segments is not None and len(self.segments):
                self.segments = self.segments[good]
            if self.keypoints is not None:
                self.keypoints = self.keypoints[good]
        return good

    def update(self, bboxes: np.ndarray, segments: np.ndarray | None = None, keypoints: np.ndarray | None = None):
        'Update instance variables.'
        self._bboxes = Bboxes(bboxes, format=self._bboxes.format)
        if segments is not None:
            self.segments = segments
        if keypoints is not None:
            self.keypoints = keypoints

    def __len__(self) -> int:
        'Return the number of instances.'
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, instances_list: list[Instances], axis: int = 0) -> Instances:
        'Concatenate a list of Instances objects into a single Instances object.'
        assert isinstance(instances_list, (list, tuple))
        if not instances_list:
            return cls(np.empty((0, 4)))
        assert all(isinstance(instance, Instances) for instance in instances_list)
        if len(instances_list) == 1:
            return instances_list[0]
        use_keypoint = instances_list[0].keypoints is not None
        bbox_format = instances_list[0]._bboxes.format
        normalized = instances_list[0].normalized
        cat_boxes = np.concatenate([ins.bboxes for ins in instances_list], axis=axis)
        seg_len = [b.segments.shape[1] for b in instances_list]
        if len(frozenset(seg_len)) > 1:  # resample segments if there's different length
            max_len = max(seg_len)
            cat_segments = np.concatenate(
                [
                    resample_segments(list(b.segments), max_len)
                    if len(b.segments)
                    else np.zeros((0, max_len, 2), dtype=np.float32)  # re-generating empty segments
                    for b in instances_list
                ],
                axis=axis,
            )
        else:
            cat_segments = np.concatenate([b.segments for b in instances_list], axis=axis)
        cat_keypoints = np.concatenate([b.keypoints for b in instances_list], axis=axis) if use_keypoint else None
        return cls(cat_boxes, cat_segments, cat_keypoints, bbox_format, normalized)

    @property
    def bboxes(self) -> np.ndarray:
        'Return bounding boxes.'
        return self._bboxes.bboxes

    def __repr__(self) -> str:
        'Return a string representation of the Instances object.'
        # Map private to public names and include direct attributes
        attr_map = {"_bboxes": "bboxes"}
        parts = []
        for key, value in self.__dict__.items():
            name = attr_map.get(key, key)
            if name == "bboxes":
                value = self.bboxes  # Use the property
            if value is not None:
                parts.append(f"{name}={value!r}")
        return "Instances({})".format("\n".join(parts))


class BaseTransform:
    'Base class for image transformations in the Ultralytics library.'

    def __call__(self, labels):
        'Apply transformation to labels dict.'
        params = self.get_params(labels)
        labels = self.apply_image(labels, params)
        labels = self.apply_instances(labels, params)
        labels = self.apply_semantic(labels, params)
        labels = self.apply_depth(labels, params)
        return labels

    def get_params(self, labels):
        'Compute and return transformation parameters.'
        return {}

    def apply_image(self, labels, params=None):
        'Apply transformation to image.'
        return labels

    def apply_instances(self, labels, params=None):
        'Apply transformation to object instances.'
        return labels

    def apply_semantic(self, labels, params=None):
        'Apply transformation to semantic segmentation mask.'
        return labels

    def apply_depth(self, labels, params=None):
        'Apply transformation to depth map.'
        return labels


class BaseMixTransform(BaseTransform):
    'Base class for mix transformations like Cutmix, MixUp and Mosaic.'

    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        'Initialize the BaseMixTransform object for mix transformations like CutMix, MixUp and Mosaic.'
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.p = p
        self.preserve_obb = getattr(dataset, "use_obb", False)

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        'Apply pre-processing transforms and cutmix/mixup/mosaic transforms to labels data.'
        if random.uniform(0, 1) > self.p:
            return labels
        params = self.get_params(labels)
        labels = self.apply_image(labels, params)
        labels = self.apply_instances(labels, params)
        labels = self.apply_semantic(labels, params)
        labels = self.apply_depth(labels, params)
        labels.pop("mix_labels", None)
        return labels

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        'Prepare mixed labels and update text labels.'
        # Get index of one or three other images
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]
        # Get images information will be used for Mosaic, CutMix or MixUp
        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]
        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)
        labels["mix_labels"] = mix_labels
        # Update cls and texts
        self._update_label_text(labels)
        return {"mix_labels": mix_labels}

    def get_indexes(self):
        'Get a random index for mix augmentation.'
        return random.randint(0, len(self.dataset) - 1)

    @staticmethod
    def _update_label_text(labels: dict[str, Any]) -> dict[str, Any]:
        'Update label text and class IDs for mixed labels in image augmentation.'
        if "texts" not in labels:
            return labels
        mix_texts = [*labels["texts"], *(item for x in labels["mix_labels"] for item in x["texts"])]
        mix_texts = [list(x) for x in dict.fromkeys(tuple(x) for x in mix_texts)]
        text2id = {tuple(text): i for i, text in enumerate(mix_texts)}
        for label in [labels] + labels["mix_labels"]:
            for i, cls in enumerate(label["cls"].squeeze(-1).tolist()):
                text = label["texts"][int(cls)]
                label["cls"][i] = text2id[tuple(text)]
            label["texts"] = mix_texts
        return labels


class Mosaic(BaseMixTransform):
    'Mosaic augmentation for image datasets.'

    def __init__(self, dataset, imgsz: int = 640, p: float = 1.0, n: int = 4):
        'Initialize the Mosaic augmentation object.'
        assert 0 <= p <= 1.0, f"The probability should be in range [0, 1], but got {p}."
        assert n in {4, 9}, "grid must be equal to 4 or 9."
        super().__init__(dataset=dataset, p=p)
        self.imgsz = imgsz
        self.border = (-imgsz // 2, -imgsz // 2)  # width, height
        self.n = n
        self.buffer_enabled = self.dataset.cache != "ram"

    def get_indexes(self):
        'Return a list of random indexes from the dataset for mosaic augmentation.'
        if self.buffer_enabled:  # select images from buffer
            return random.choices(list(self.dataset.buffer), k=self.n - 1)
        else:  # select any images
            return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        'Compute mosaic layout parameters.'
        params = super().get_params(labels)
        assert labels.get("rect_shape") is None, "rect and mosaic are mutually exclusive."
        assert len(labels.get("mix_labels", [])), "There are no other images for mosaic augment."
        s = self.imgsz
        layout = []
        if self.n == 4:
            yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)
            for i in range(4):
                labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
                img = labels_patch["img"]
                h, w = labels_patch.get("resized_shape", img.shape[:2])
                if i == 0:  # top left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
                elif i == 1:  # top right
                    x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                    x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                elif i == 2:  # bottom left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
                elif i == 3:  # bottom right
                    x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                    x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
                padw = x1a - x1b
                padh = y1a - y1b
                layout.append(
                    {
                        "labels_patch": labels_patch,
                        "x1a": x1a,
                        "y1a": y1a,
                        "x2a": x2a,
                        "y2a": y2a,
                        "x1b": x1b,
                        "y1b": y1b,
                        "x2b": x2b,
                        "y2b": y2b,
                        "padw": padw,
                        "padh": padh,
                        "img_shape": (h, w),
                    }
                )
        elif self.n == 9:
            hp, wp = -1, -1
            h0, w0 = None, None
            for i in range(9):
                labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
                img = labels_patch["img"]
                h, w = labels_patch.get("resized_shape", img.shape[:2])
                if i == 0:  # center
                    c = s, s, s + w, s + h
                    h0, w0 = h, w
                elif i == 1:  # top
                    c = s, s - h, s + w, s
                elif i == 2:  # top right
                    c = s + wp, s - h, s + wp + w, s
                elif i == 3:  # right
                    c = s + w0, s, s + w0 + w, s + h
                elif i == 4:  # bottom right
                    c = s + w0, s + hp, s + w0 + w, s + hp + h
                elif i == 5:  # bottom
                    c = s + w0 - w, s + h0, s + w0, s + h0 + h
                elif i == 6:  # bottom left
                    c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
                elif i == 7:  # left
                    c = s - w, s + h0 - h, s, s + h0
                elif i == 8:  # top left
                    c = s - w, s + h0 - hp - h, s, s + h0 - hp
                padw, padh = c[:2]
                x1, y1, x2, y2 = (max(x, 0) for x in c)
                layout.append(
                    {
                        "labels_patch": labels_patch,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "padw": padw,
                        "padh": padh,
                        "img_shape": (h, w),
                    }
                )
                hp, wp = h, w
        params["layout"] = layout
        return params

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        'Apply mosaic augmentation to the image.'
        layout = params["layout"]
        if self.n == 4:
            img4 = np.full((self.imgsz * 2, self.imgsz * 2, labels["img"].shape[2]), 114, dtype=np.uint8)
            for item in layout:
                labels_patch = item["labels_patch"]
                img = labels_patch["img"]
                x1a, y1a, x2a, y2a = item["x1a"], item["y1a"], item["x2a"], item["y2a"]
                x1b, y1b, x2b, y2b = item["x1b"], item["y1b"], item["x2b"], item["y2b"]
                img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            labels["img"] = img4
        elif self.n == 9:
            img9 = np.full((self.imgsz * 3, self.imgsz * 3, labels["img"].shape[2]), 114, dtype=np.uint8)
            for item in layout:
                labels_patch = item["labels_patch"]
                img = labels_patch["img"]
                x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
                padw, padh = item["padw"], item["padh"]
                x1b, y1b = x1 - padw, y1 - padh
                x2b, y2b = x1b + (x2 - x1), y1b + (y2 - y1)
                img9[y1:y2, x1:x2] = img[y1b:y2b, x1b:x2b]
            labels["img"] = img9[-self.border[0] : self.border[0], -self.border[1] : self.border[1]]
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        'Apply mosaic augmentation to instances.'
        layout = params["layout"]
        mosaic_labels = []
        for item in layout:
            if self.n == 4:
                padw = item["padw"]
                padh = item["padh"]
            else:  # n == 9
                padw = item["padw"] + self.border[0]
                padh = item["padh"] + self.border[1]
            labels_patch = self._update_labels(item["labels_patch"], padw, padh, item.get("img_shape"))
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)
        labels.update(final_labels)
        return labels

    def apply_semantic(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        'Apply mosaic augmentation to semantic mask.'
        if labels.get("semantic_mask") is None and all(
            m.get("semantic_mask") is None for m in labels.get("mix_labels", [])
        ):
            return labels
        layout = params["layout"]
        if self.n == 4:
            mask4 = np.full((self.imgsz * 2, self.imgsz * 2), 255, dtype=np.uint8)
            for item in layout:
                labels_patch = item["labels_patch"]
                mask = labels_patch.get("semantic_mask")
                if mask is None:
                    continue
                x1a, y1a, x2a, y2a = item["x1a"], item["y1a"], item["x2a"], item["y2a"]
                x1b, y1b, x2b, y2b = item["x1b"], item["y1b"], item["x2b"], item["y2b"]
                mask4[y1a:y2a, x1a:x2a] = mask[y1b:y2b, x1b:x2b]
            labels["semantic_mask"] = mask4
        elif self.n == 9:
            mask9 = np.full((self.imgsz * 3, self.imgsz * 3), 255, dtype=np.uint8)
            for item in layout:
                labels_patch = item["labels_patch"]
                mask = labels_patch.get("semantic_mask")
                if mask is None:
                    continue
                x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
                padw, padh = item["padw"], item["padh"]
                x1b, y1b = x1 - padw, y1 - padh
                x2b, y2b = x1b + (x2 - x1), y1b + (y2 - y1)
                mask9[y1:y2, x1:x2] = mask[y1b:y2b, x1b:x2b]
            labels["semantic_mask"] = mask9[-self.border[0] : self.border[0], -self.border[1] : self.border[1]]
        return labels

    @staticmethod
    def _update_labels(labels, padw: int, padh: int, img_shape: tuple[int, int] | None = None) -> dict[str, Any]:
        'Update label coordinates with padding values.'
        nh, nw = img_shape if img_shape is not None else labels["img"].shape[:2]
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(nw, nh)
        labels["instances"].add_padding(padw, padh)
        return labels

    def _cat_labels(self, mosaic_labels: list[dict[str, Any]]) -> dict[str, Any]:
        'Concatenate and process labels for mosaic augmentation.'
        if not mosaic_labels:
            return {}
        cls = []
        instances = []
        imgsz = self.imgsz * 2  # mosaic imgsz
        for labels in mosaic_labels:
            cls.append(labels["cls"])
            instances.append(labels["instances"])
        # Final labels
        final_labels = {
            "im_file": mosaic_labels[0]["im_file"],
            "ori_shape": mosaic_labels[0]["ori_shape"],
            "resized_shape": (imgsz, imgsz),
            "cls": np.concatenate(cls, 0),
            "instances": Instances.concatenate(instances, axis=0),
        }
        final_labels["instances"].clip(imgsz, imgsz, preserve_obb=self.preserve_obb)
        good = final_labels["instances"].remove_zero_area_boxes()
        final_labels["cls"] = final_labels["cls"][good]
        if "texts" in mosaic_labels[0]:
            final_labels["texts"] = mosaic_labels[0]["texts"]
        return final_labels


class RandomPerspective(BaseTransform):
    'Implement random perspective and affine transformations on images and corresponding annotations.'

    def __init__(
        self,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float | tuple[float, float] = 0.5,
        shear: float = 0.0,
        perspective: float = 0.0,
        size: tuple[int, int] | None = None,
        preserve_obb: bool = False,
    ):
        'Initialize RandomPerspective object with transformation parameters.'
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.size = size
        self.preserve_obb = preserve_obb

    def _compute_affine_matrix(self, img: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float]:
        'Compute the affine transformation matrix without applying it.'
        # Center
        C = np.eye(3, dtype=np.float32)
        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)
        # Perspective
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)
        # Rotation and Scale
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        if isinstance(self.scale, (tuple, list)):
            s = random.uniform(self.scale[0], self.scale[1])
        else:
            s = random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
        # Shear
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)
        # Translation
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * size[0]  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * size[1]  # y translation (pixels)
        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        return M, s

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        'Compute affine transformation parameters shared across image and instances.'
        img = labels["img"]
        if (rect_shape := labels.get("rect_shape")) is not None:  # rect has higher priority
            size = (int(rect_shape[1]), int(rect_shape[0]))  # rect mode batch shape (h, w) to (w, h)
        else:
            size = (img.shape[1], img.shape[0]) if self.size is None else self.size  # w, h
        orig_shape = img.shape[:2]
        M, scale = self._compute_affine_matrix(img, size)
        return {"M": M, "scale": scale, "orig_shape": orig_shape, "size": size}

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        'Apply affine warp to the image.'
        img = labels["img"]
        M = params["M"]
        size = params["size"]
        # 4 values: cv2 tiles borderValue in blocks of 4, so a 3-tuple zeroes every 4th multispectral channel
        if self.perspective:
            img = cv2.warpPerspective(img, M, dsize=size, borderValue=(114, 114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=size, borderValue=(114, 114, 114, 114))
        if img.ndim == 2:
            img = img[..., None]
        labels["img"] = img
        labels["resized_shape"] = img.shape[:2]
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        'Apply the affine transformation to object instances and filter out overly distorted ones.'
        cls = labels["cls"]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xyxy")
        instances.denormalize(*params["orig_shape"][::-1])
        M = params["M"]
        scale = params["scale"]
        bboxes = self.apply_bboxes(instances.bboxes, M)
        segments = instances.segments
        keypoints = instances.keypoints
        # Drop objects crossing the projection plane (w = 0), e.g. at Mosaic canvas corners, as they flip into huge boxes
        pts = segments if len(segments) else instances.bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(-1, 4, 2)
        w = pts @ M[2, :2] + M[2, 2]  # projective divisor
        # Update bboxes if there are segments.
        if len(segments):
            bboxes, segments = self.apply_segments(segments, M, params["size"])
        if keypoints is not None:
            keypoints = self.apply_keypoints(keypoints, M, params["size"])
        new_instances = Instances(bboxes, segments, keypoints, bbox_format="xyxy", normalized=False)
        # Clip
        new_instances.clip(*params["size"], preserve_obb=self.preserve_obb)
        # Filter instances
        instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
        # Make the bboxes have the same scale with new_bboxes
        i = (w.min(1) * w.max(1) > 0) & self.box_candidates(
            box1=instances.bboxes.T, box2=new_instances.bboxes.T, area_thr=0.01 if len(segments) else 0.10
        )
        labels["instances"] = new_instances[i]
        labels["cls"] = cls[i]
        return labels

    def apply_bboxes(self, bboxes: np.ndarray, M: np.ndarray) -> np.ndarray:
        'Apply affine transformation to bounding boxes.'
        n = len(bboxes)
        if n == 0:
            return bboxes
        xy = np.ones((n * 4, 3), dtype=bboxes.dtype)
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine
        # Create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        return np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=bboxes.dtype).reshape(4, n).T

    def apply_segments(
        self, segments: np.ndarray, M: np.ndarray, size: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        'Transform segments and derive their bounding boxes.'
        n, num = segments.shape[:2]
        if n == 0:
            return [], segments
        xy = np.ones((n * num, 3), dtype=segments.dtype)
        segments = segments.reshape(-1, 2)
        xy[:, :2] = segments
        xy = xy @ M.T  # transform
        xy = xy[:, :2] / xy[:, 2:3]
        segments = xy.reshape(n, -1, 2)
        bboxes = np.stack([segment2box(xy, size[0], size[1]) for xy in segments], 0)
        if not self.preserve_obb:
            segments[..., 0] = segments[..., 0].clip(bboxes[:, 0:1], bboxes[:, 2:3])
            segments[..., 1] = segments[..., 1].clip(bboxes[:, 1:2], bboxes[:, 3:4])
        return bboxes, segments

    def apply_keypoints(self, keypoints: np.ndarray, M: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        'Apply affine transformation to keypoints.'
        n, nkpt = keypoints.shape[:2]
        if n == 0:
            return keypoints
        xy = np.ones((n * nkpt, 3), dtype=keypoints.dtype)
        visible = keypoints[..., 2].reshape(n * nkpt, 1)
        xy[:, :2] = keypoints[..., :2].reshape(n * nkpt, 2)
        xy = xy @ M.T  # transform
        xy = xy[:, :2] / xy[:, 2:3]  # perspective rescale or affine
        out_mask = (xy[:, 0] < 0) | (xy[:, 1] < 0) | (xy[:, 0] > size[0]) | (xy[:, 1] > size[1])
        visible[out_mask] = 0
        return np.concatenate([xy, visible], axis=-1).reshape(n, nkpt, 3)

    def apply_semantic(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        'Apply affine transformation to semantic segmentation mask.'
        if "semantic_mask" not in labels or labels["semantic_mask"] is None:
            return labels
        mask = labels["semantic_mask"]
        M = params["M"]
        size = params["size"]
        if (size[0] != mask.shape[1] or size[1] != mask.shape[0]) or (M != np.eye(3)).any():
            if self.perspective:
                mask = cv2.warpPerspective(mask, M, dsize=size, flags=cv2.INTER_NEAREST, borderValue=255)
            else:
                mask = cv2.warpAffine(mask, M[:2], dsize=size, flags=cv2.INTER_NEAREST, borderValue=255)
        labels["semantic_mask"] = mask
        return labels

    def apply_depth(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        'Apply the same projective warp to metric depth maps.'
        depth = labels.get("depth")
        if depth is None:
            return labels
        M = params["M"]
        size = params["size"]
        if (size[0] != depth.shape[1] or size[1] != depth.shape[0]) or (M != np.eye(3)).any():
            if self.perspective:
                depth = cv2.warpPerspective(depth, M, dsize=size, flags=cv2.INTER_NEAREST, borderValue=0)
            else:
                depth = cv2.warpAffine(depth, M[:2], dsize=size, flags=cv2.INTER_NEAREST, borderValue=0)
        labels["depth"] = depth
        return labels

    @staticmethod
    def box_candidates(
        box1: np.ndarray,
        box2: np.ndarray,
        wh_thr: int = 2,
        ar_thr: int = 100,
        area_thr: float = 0.1,
        eps: float = 1e-16,
    ) -> np.ndarray:
        'Compute candidate boxes for further processing based on size and aspect ratio criteria.'
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


class RandomHSV(BaseTransform):
    'Randomly adjust the Hue, Saturation, and Value (HSV) channels of an image.'

    def __init__(self, hgain: float = 0.5, sgain: float = 0.5, vgain: float = 0.5) -> None:
        'Initialize the RandomHSV object for random HSV (Hue, Saturation, Value) augmentation.'
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def apply_image(self, labels, params: dict[str, Any] | None = None):
        'Apply random HSV augmentation to an image within predefined limits.'
        img = labels["img"]
        if img.shape[-1] != 3:  # only apply to 3-channel (BGR) images
            return labels
        if self.hgain or self.sgain or self.vgain:
            dtype = img.dtype  # uint8
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain]  # random gains
            x = np.arange(0, 256, dtype=r.dtype)
            # lut_hue = ((x * (r[0] + 1)) % 180).astype(dtype)   # original hue implementation from ultralytics<=8.3.78
            lut_hue = ((x + r[0] * 180) % 180).astype(dtype)
            lut_sat = np.clip(x * (r[1] + 1), 0, 255).astype(dtype)
            lut_val = np.clip(x * (r[2] + 1), 0, 255).astype(dtype)
            lut_sat[0] = 0  # prevent pure white changing color, introduced in 8.3.79
            im_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lut = np.stack((lut_hue, lut_sat, lut_val), -1).reshape(1, 256, 3)  # single 3-channel LUT
            cv2.cvtColor(cv2.LUT(im_hsv, lut), cv2.COLOR_HSV2BGR, dst=img)  # no return needed
        return labels


class RandomFlip(BaseTransform):
    'Apply a random horizontal or vertical flip to an image with a given probability.'

    def __init__(self, p: float = 0.5, direction: str = "horizontal", flip_idx: list[int] | None = None) -> None:
        'Initialize the RandomFlip class with probability and direction.'
        assert direction in {"horizontal", "vertical"}, f"Support direction `horizontal` or `vertical`, got {direction}"
        assert 0 <= p <= 1.0, f"The probability should be in range [0, 1], but got {p}."
        self.p = p
        self.direction = direction
        self.flip_idx = flip_idx

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        'Compute random flip parameters.'
        img = labels["img"]
        instances = labels["instances"]
        h, w = img.shape[:2]
        h = 1 if instances.normalized else h
        w = 1 if instances.normalized else w
        return {
            "flip": random.random() < self.p,
            "h": h,
            "w": w,
            "direction": self.direction,
            "flip_idx": self.flip_idx,
        }

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        'Apply flip to the image.'
        img = labels["img"]
        if params["flip"]:
            if params["direction"] == "vertical":
                img = np.flipud(img)
            elif params["direction"] == "horizontal":
                img = np.fliplr(img)
        labels["img"] = img
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        'Apply flip to object instances.'
        instances = labels.pop("instances")
        instances.convert_bbox(format="xywh")
        if params["flip"]:
            if params["direction"] == "vertical":
                instances.flipud(params["h"])
            elif params["direction"] == "horizontal":
                instances.fliplr(params["w"])
            if params["flip_idx"] is not None and instances.keypoints is not None:
                instances.keypoints = np.ascontiguousarray(instances.keypoints[:, params["flip_idx"], :])
        labels["instances"] = instances
        return labels

    def apply_semantic(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        'Apply flip to semantic segmentation mask.'
        if "semantic_mask" not in labels or labels["semantic_mask"] is None:
            return labels
        if params["flip"]:
            if params["direction"] == "vertical":
                labels["semantic_mask"] = np.flipud(labels["semantic_mask"])
            elif params["direction"] == "horizontal":
                labels["semantic_mask"] = np.fliplr(labels["semantic_mask"])
        return labels

    def apply_depth(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        'Apply flip to the paired metric depth map.'
        if labels.get("depth") is None:
            return labels
        if params["flip"]:
            if params["direction"] == "vertical":
                labels["depth"] = np.flipud(labels["depth"])
            elif params["direction"] == "horizontal":
                labels["depth"] = np.fliplr(labels["depth"])
        return labels


class LetterBox(BaseTransform):
    'Resize image and padding for detection, instance segmentation, pose.'

    def __init__(
        self,
        new_shape: int | tuple[int, int] = (640, 640),
        auto: bool = False,
        scale_fill: bool = False,
        scaleup: bool = True,
        center: bool = True,
        stride: int = 32,
        padding_value: int = 114,
        interpolation: int = cv2.INTER_LINEAR,
    ):
        'Initialize LetterBox object for resizing and padding images.'
        self.new_shape = new_shape
        self.auto = auto
        self.scale_fill = scale_fill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left
        self.padding_value = padding_value
        self.interpolation = interpolation

    def __call__(
        self, labels: dict[str, Any] | None = None, image: np.ndarray | None = None
    ) -> dict[str, Any] | np.ndarray:
        'Resize and pad an image for object detection, instance segmentation, or pose estimation tasks.'
        if labels is None:
            labels = {}
        return_image_only = len(labels) == 0
        if image is not None:
            labels["img"] = image
        params = self.get_params(labels)
        labels = self.apply_image(labels, params)
        if not return_image_only:
            labels = self.apply_instances(labels, params)
        labels = self.apply_semantic(labels, params)
        if return_image_only:
            return labels["img"]
        return labels

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        'Compute letterboxing parameters.'
        img = labels["img"]
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)
        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = max(round(shape[1] * r), 1), max(round(shape[0] * r), 1)  # keep thin images at least 1 pixel
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scale_fill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2
        top, bottom = round(dh - 0.1) if self.center else 0, round(dh + 0.1)
        left, right = round(dw - 0.1) if self.center else 0, round(dw + 0.1)
        return {
            "orig_shape": shape,
            "new_shape": new_shape,
            "ratio": ratio,
            "new_unpad": new_unpad,
            "top": top,
            "bottom": bottom,
            "left": left,
            "right": right,
        }

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        'Resize and pad the image.'
        img = labels["img"]
        shape = img.shape[:2]
        new_unpad = params["new_unpad"]
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=self.interpolation)
        if img.ndim == 2:  # single-channel input (e.g. SAM mask prompts), needs a channel dim for unpack and pad below
            img = img[..., None]
        h, w, c = img.shape
        top, bottom = params["top"], params["bottom"]
        left, right = params["left"], params["right"]
        if c == 3:
            img = cv2.copyMakeBorder(
                img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(self.padding_value,) * 3
            )
        else:  # multispectral
            pad_img = np.full((h + top + bottom, w + left + right, c), fill_value=self.padding_value, dtype=img.dtype)
            pad_img[top : top + h, left : left + w] = img
            img = pad_img
        labels["img"] = img
        labels["resized_shape"] = params["new_shape"]
        return labels

    def apply_semantic(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        'Apply letterboxing to semantic segmentation mask.'
        if "semantic_mask" not in labels or labels["semantic_mask"] is None:
            return labels
        mask = labels["semantic_mask"]
        shape = params["orig_shape"]
        new_unpad = params["new_unpad"]
        if shape[::-1] != new_unpad:
            mask = cv2.resize(mask, new_unpad, interpolation=cv2.INTER_NEAREST)
        top, bottom = params["top"], params["bottom"]
        left, right = params["left"], params["right"]
        mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)
        labels["semantic_mask"] = mask
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        'Update instance coordinates after letterboxing.'
        if "instances" in labels:
            labels = self._update_labels(labels, params["ratio"], params["left"], params["top"], params["orig_shape"])
        if labels.get("ratio_pad"):
            gain_h, gain_w = labels["ratio_pad"]
            ratio_w, ratio_h = params["ratio"]
            labels["ratio_pad"] = (
                (gain_h * ratio_h, gain_w * ratio_w),
                (params["left"], params["top"]),
            )  # for evaluation
        return labels

    @staticmethod
    def _update_labels(
        labels: dict[str, Any], ratio: tuple[float, float], padw: float, padh: float, orig_shape: tuple[int, int]
    ) -> dict[str, Any]:
        'Update labels after applying letterboxing to an image.'
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*orig_shape[::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels


class RandomLoadText(BaseTransform):
    'Randomly sample positive and negative texts and update class indices accordingly.'

    def __init__(
        self,
        prompt_format: str = "{}",
        neg_samples: tuple[int, int] = (80, 80),
        max_samples: int = 80,
        padding: bool = False,
        padding_value: list[str] | None = None,
    ) -> None:
        'Initialize the RandomLoadText class for randomly sampling positive and negative texts.'
        self.prompt_format = prompt_format
        self.neg_samples = neg_samples
        self.max_samples = max_samples
        self.padding = padding
        self.padding_value = padding_value if padding_value is not None else [""]

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        'Compute text sampling parameters.'
        assert "texts" in labels, "No texts found in labels."
        class_texts = labels["texts"]
        num_classes = len(class_texts)
        cls = np.asarray(labels.pop("cls"), dtype=int)
        pos_labels = np.unique(cls).tolist()
        if len(pos_labels) > self.max_samples:
            pos_labels = random.sample(pos_labels, k=self.max_samples)
        neg_samples = min(min(num_classes, self.max_samples) - len(pos_labels), random.randint(*self.neg_samples))
        neg_labels = [i for i in range(num_classes) if i not in pos_labels]
        neg_labels = random.sample(neg_labels, k=neg_samples)
        sampled_labels = pos_labels + neg_labels
        # Randomness
        # random.shuffle(sampled_labels)
        label2ids = {label: i for i, label in enumerate(sampled_labels)}
        valid_idx = np.zeros(len(labels["instances"]), dtype=bool)
        new_cls = []
        for i, label in enumerate(cls.squeeze(-1).tolist()):
            if label not in label2ids:
                continue
            valid_idx[i] = True
            new_cls.append([label2ids[label]])
        # Randomly select one prompt when there's more than one prompts
        texts = []
        for label in sampled_labels:
            prompts = class_texts[label]
            assert len(prompts) > 0
            prompt = self.prompt_format.format(prompts[random.randrange(len(prompts))])
            texts.append(prompt)
        if self.padding:
            valid_labels = len(pos_labels) + len(neg_labels)
            num_padding = self.max_samples - valid_labels
            if num_padding > 0:
                texts += random.choices(self.padding_value, k=num_padding)
        return {"valid_idx": valid_idx, "new_cls": np.array(new_cls), "texts": texts}

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        'Filter instances and update class labels based on sampled texts.'
        labels["instances"] = labels["instances"][params["valid_idx"]]
        labels["cls"] = params["new_cls"]
        labels["texts"] = params["texts"]
        return labels


def polygon2mask(
    imgsz: tuple[int, int], polygons: list[np.ndarray], color: int = 1, downsample_ratio: int = 1
) -> np.ndarray:
    'Convert a list of polygons to a binary mask of the specified image size.'
    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons, dtype=np.int32)
    polygons = polygons.reshape((polygons.shape[0], -1, 2))
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio)
    # Note: fillPoly first then resize is trying to keep the same loss calculation method when mask-ratio=1
    return cv2.resize(mask, (nw, nh))


def polygons2masks(
    imgsz: tuple[int, int], polygons: list[np.ndarray], color: int, downsample_ratio: int = 1
) -> np.ndarray:
    'Convert a list of polygons to a set of binary masks of the specified image size.'
    return np.array([polygon2mask(imgsz, [x.reshape(-1)], color, downsample_ratio) for x in polygons])
