"""Day 42 homework — detection fundamentals: boxes, IoU, NMS.

Fill in every ``TODO``. Run ``pytest -q`` in this folder until it's green.
Try for ~20 minutes before peeking at ``solution.py``; ask Claude Code for a *hint*, not the answer.

Every object detector (YOLO on Day 43, and the KITTI labels on Day 47) speaks in boxes. Two operations
are universal: **IoU** measures how much two boxes overlap, and **NMS** uses IoU to remove duplicate
detections of the same object. Build them and you understand the output of any detector.

Boxes are ``[x1, y1, x2, y2]`` — top-left corner, bottom-right corner.
"""
from __future__ import annotations

import numpy as np


def box_area(box) -> float:
    """Area of a box [x1, y1, x2, y2]. Return 0.0 for a degenerate box (x2<x1 or y2<y1)."""
    # TODO: unpack the four numbers; multiply width by height, clamped at 0
    raise NotImplementedError


def xywh_to_xyxy(box):
    """Convert a top-left [x, y, w, h] box to corner form [x1, y1, x2, y2].
    (Detectors and datasets disagree on box format — converting is a daily chore.)"""
    # TODO
    raise NotImplementedError


def iou(box_a, box_b) -> float:
    """Intersection-over-Union of two boxes = overlap area / combined (union) area.

    intersection corners: ix1=max(ax1,bx1), iy1=max(ay1,by1), ix2=min(ax2,bx2), iy2=min(ay2,by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = area(a) + area(b) - inter
    Return inter/union (or 0.0 if union is 0).
    """
    # TODO
    raise NotImplementedError


def nms(boxes, scores, iou_threshold: float = 0.5):
    """Greedy non-max suppression. Return the kept box indices, highest score first.

    Repeat: take the highest-scoring box that's left, keep it, and drop every remaining box
    whose IoU with it exceeds ``iou_threshold`` (they're duplicates of the same object).
    """
    # TODO: sort indices by score (descending); loop, keeping the top and filtering the rest by iou.
    # New numpy tool: np.argsort(scores) returns the POSITIONS that would sort the values low->high
    # (a cousin of Day 31's argmin). Add [::-1] to reverse it to high->low:
    #     order = list(np.argsort(scores)[::-1])
    raise NotImplementedError
