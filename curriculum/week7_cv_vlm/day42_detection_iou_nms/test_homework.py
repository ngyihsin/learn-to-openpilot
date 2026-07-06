"""Auto-grader for Day 42 — detection fundamentals: boxes, IoU, NMS.

Run ``pytest -q`` here to grade ``homework.py``.
Grade the reference solution with:  ``LP_IMPL=solution pytest -q``
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest


def _load(name: str):
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    path = os.path.join(here, name + ".py")
    modname = f"lp_{os.path.basename(here)}_{name}"
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_impl = _load(os.environ.get("LP_IMPL", "homework"))
box_area = _impl.box_area
xywh_to_xyxy = _impl.xywh_to_xyxy
iou = _impl.iou
nms = _impl.nms


def test_box_area():
    assert box_area([0, 0, 2, 3]) == pytest.approx(6.0)
    assert box_area([5, 5, 2, 2]) == 0.0            # degenerate -> 0, not negative


def test_xywh_to_xyxy():
    assert list(xywh_to_xyxy([10, 20, 30, 40])) == [10, 20, 40, 60]


def test_iou_extremes():
    assert iou([0, 0, 2, 2], [0, 0, 2, 2]) == pytest.approx(1.0)   # identical
    assert iou([0, 0, 1, 1], [5, 5, 6, 6]) == pytest.approx(0.0)   # disjoint


def test_iou_partial_overlap():
    # a=[0,0,2,2] area 4, b=[1,1,3,3] area 4, overlap [1,1,2,2] area 1 -> 1/(4+4-1)=1/7
    assert iou([0, 0, 2, 2], [1, 1, 3, 3]) == pytest.approx(1 / 7)


def test_nms_suppresses_duplicates():
    boxes = [[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]]
    scores = [0.9, 0.8, 0.7]
    # boxes 0 and 1 overlap heavily -> keep the higher-scoring 0; box 2 is separate -> kept
    assert nms(boxes, scores, iou_threshold=0.5) == [0, 2]


def test_nms_keeps_all_when_disjoint():
    boxes = [[0, 0, 1, 1], [5, 5, 6, 6], [10, 10, 11, 11]]
    scores = [0.5, 0.9, 0.7]
    # nothing overlaps -> all kept, ordered by score (desc)
    assert nms(boxes, scores, iou_threshold=0.5) == [1, 2, 0]


def test_nms_threshold_matters():
    boxes = [[0, 0, 2, 2], [1, 1, 3, 3]]   # IoU = 1/7 ~= 0.143
    scores = [0.9, 0.8]
    # a LOWER threshold suppresses MORE aggressively:
    assert nms(boxes, scores, iou_threshold=0.5) == [0, 1]   # 0.143 <= 0.5 -> kept (not a duplicate)
    assert nms(boxes, scores, iou_threshold=0.1) == [0]      # 0.143 >  0.1 -> box 1 suppressed
