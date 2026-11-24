"""NumPy port of the Ultralytics 8.4.216 end-to-end validation protocol (AGPL-3.0)."""
import numpy as np

from geometry import xywh2xyxy


def match(iou, targets, classes):
    thresholds = np.linspace(.5, .95, 10, dtype=np.float32)
    iou = iou * (targets[:, None] == classes)
    correct = np.zeros((len(classes), len(thresholds)), bool)
    claimed = np.zeros((len(targets), len(thresholds)), bool)
    columns = np.arange(len(thresholds))
    for det in np.flatnonzero((iou >= thresholds.min()).any(0)):
        available = np.where(claimed, 0, iou[:, det, None])
        best = available.argmax(0)
        correct[det] = available[best, columns] >= thresholds
        claimed[best, columns] |= correct[det]
    return correct


def bbox_iou(target, pred):
    lower = np.maximum(target[:, None, :2], pred[:, :2])
    upper = np.minimum(target[:, None, 2:], pred[:, 2:])
    intersection = np.maximum(upper - lower, 0).prod(2)
    a = (target[:, 2:] - target[:, :2]).prod(1)
    b = (pred[:, 2:] - pred[:, :2]).prod(1)
    return intersection / (a[:, None] + b - intersection + 1e-7)


def mask_iou(gt, rows, proto):
    c, h, w = proto.shape
    gh, gw = gt.shape[1:]
    if not len(gt) or not len(rows):
        return np.zeros((len(gt), len(rows)), np.float32)
    flat = gt.reshape(len(gt), -1).astype(np.float32)
    overlaps = []
    for start in range(0, len(rows), 32):
        chunk = rows[start:start + 32]
        masks = (chunk[:, 6:] @ proto.reshape(c, -1)).reshape(-1,h,w) > 0
        x1,y1,x2,y2 = (chunk[:, :4] / 4).T[:, :, None, None]
        x, y = np.arange(w)[None,None,:], np.arange(h)[None,:,None]
        masks &= (x >= x1) & (x < x2) & (y >= y1) & (y < y2)
        sy, sx = h // gh, w // gw
        masks = masks[:,sy//2::sy,sx//2::sx][:,:gh,:gw].reshape(len(chunk), -1).astype(np.float32)
        shared = flat @ masks.T
        overlaps.append(shared / (flat.sum(1)[:,None] + masks.sum(1) - shared + 1e-7))
    return np.concatenate(overlaps, 1)


def summary(tp, confidence, pred, truth):
    order = np.argsort(-confidence)
    tp, confidence, pred = tp[order], confidence[order], pred[order]
    classes, counts = np.unique(truth, return_counts=True)
    grid = np.linspace(0, 1, 1000)
    ap_grid = np.linspace(0, 1, 101)
    p = np.zeros((len(classes), len(grid)))
    r = np.zeros_like(p)
    ap = np.zeros((len(classes), 10))
    for i, category in enumerate(classes):
        keep = pred == category
        if not keep.any():
            continue
        hits = tp[keep].cumsum(0)
        precision = hits / np.arange(1, len(hits) + 1)[:, None]
        recall = hits / (counts[i] + 1e-16)
        p[i] = np.interp(-grid, -confidence[keep], precision[:,0], left=1)
        r[i] = np.interp(-grid, -confidence[keep], recall[:,0], left=0)
        for j in range(10):
            xs = np.r_[0., recall[:,j], recall[-1,j], 1.]
            ys = np.r_[1., precision[:,j], 0., 0.]
            envelope = np.maximum.accumulate(ys[::-1])[::-1]
            ap[i,j] = np.trapz(np.interp(ap_grid, xs, envelope), ap_grid)
    if not len(classes):
        return [0.] * 4
    f1 = (2 * p * r / (p + r + 1e-16)).mean(0)
    index = np.convolve(np.pad(f1, (50,50), mode="edge"), np.ones(101)/101, "valid").argmax()
    return [float(p[:,index].mean()), float(r[:,index].mean()), float(ap[:,0].mean()), float(ap.mean())]


class Evaluator:
    def __init__(self, single, segment):
        self.single, self.segment = single, segment
        self.seen = 0
        self.rows = []

    def update(self, batch, predictions):
        rows, protos = predictions if self.segment else (predictions, None)
        h,w = batch["img"].shape[-2:]
        for i, preds in enumerate(rows):
            preds = preds[preds[:,4] > .001][:300]
            labels = batch["batch_idx"] == i
            target = batch["cls"][labels].ravel()
            boxes = xywh2xyxy(batch["bboxes"][labels]) * np.array([w,h,w,h], np.float32)
            classes = np.zeros_like(preds[:,5]) if self.single else preds[:,5]
            matched = match(bbox_iou(boxes, preds[:,:4]), target, classes)
            parts = [target, classes, preds[:,4], matched]
            if self.segment:
                parts.append(match(mask_iou(batch["masks"][labels], preds, protos[i]), target, classes))
            self.rows.append(parts)
            self.seen += 1

    def result(self):
        truth, pred, conf, *matches = [np.concatenate(col) for col in zip(*self.rows)]
        metrics = {}
        fitness = 0.
        for kind, tp in zip(("B","M"), matches):
            values = summary(tp, conf, pred, truth)
            for name, value in zip(("precision","recall","mAP50","mAP50-95"), values):
                metrics[f"metrics/{name}({kind})"] = value
            fitness += values[-1]
        return dict(seen=self.seen, metrics={**metrics, "fitness":fitness})
