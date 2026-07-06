"""Day 42 reference solution — detection fundamentals: boxes, IoU, NMS.

Run ``LP_IMPL=solution pytest -q`` to confirm the tests are correct. Self-contained (numpy only).
Boxes are ``[x1, y1, x2, y2]`` (top-left, bottom-right) unless noted.
"""
from __future__ import annotations

import numpy as np


def box_area(box) -> float:
    """Area of a box [x1, y1, x2, y2]. A degenerate/negative box has area 0."""
    x1, y1, x2, y2 = box
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def xywh_to_xyxy(box):
    """Convert a top-left [x, y, w, h] box to corner form [x1, y1, x2, y2]."""
    x, y, w, h = box
    return [x, y, x + w, y + h]


def iou(box_a, box_b) -> float:
    """Intersection-over-Union of two [x1,y1,x2,y2] boxes: overlap area / combined area."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(box_a) + box_area(box_b) - inter
    return float(inter / union) if union > 0 else 0.0


def nms(boxes, scores, iou_threshold: float = 0.5):
    """Greedy non-max suppression. Return the kept box indices, highest score first.
    Repeatedly keep the top-scoring box and drop every remaining box that overlaps it by
    more than ``iou_threshold``."""
    boxes = [list(b) for b in boxes]
    order = list(np.argsort(scores)[::-1])   # indices, high score -> low
    keep = []
    while order:
        i = int(order.pop(0))
        keep.append(i)
        order = [j for j in order if iou(boxes[i], boxes[j]) <= iou_threshold]
    return keep
