# Day 42 — Detection Fundamentals: Boxes, IoU & NMS

> **Week 7 · CV & VLMs** — before you run a detector (Day 43), build the two operations every detector
> depends on. Small functions, huge payoff — you'll recognize them inside YOLO, SAM, and KITTI labels.
>
> *Gradable: assumes basic Python + Day 31 numpy.*

## Why today matters

A detector outputs **boxes** — but boxes need two universal tools:

- **IoU (Intersection-over-Union)** — a single number, 0 to 1, for *how much two boxes overlap*. It's how
  you compare a prediction to the truth, and how you tell two boxes apart.
- **NMS (Non-Max Suppression)** — a detector often fires several overlapping boxes on one object; NMS
  keeps the best and removes the duplicates, using IoU to decide what's a duplicate.

Every detector runs NMS internally, and every evaluation uses IoU. Build them once and the rest of the
week's models stop being black boxes.

## Key words (plain language)

| Term | In one sentence |
|------|-----------------|
| **box `[x1,y1,x2,y2]`** | Top-left and bottom-right corners of a rectangle in the image. |
| **IoU** | overlap area ÷ combined area. 1 = identical, 0 = no overlap. |
| **NMS** | Keep the top-scoring box; drop others that overlap it too much; repeat. |
| **IoU threshold** | The overlap above which NMS calls two boxes "the same object." *Lower = more aggressive.* |

## See it with real numbers

Two boxes `a=[0,0,2,2]` and `b=[1,1,3,3]`: each has area 4; they overlap in the square `[1,1,2,2]` of
area 1. So `IoU = 1 / (4 + 4 − 1) = 1/7 ≈ 0.143`. NMS with threshold 0.5 keeps both (0.143 < 0.5, not
duplicates); with threshold 0.1 it suppresses the lower-scoring one (0.143 > 0.1).

## Do this — four steps

Work top-to-bottom in `homework.py`; run each check in this folder.

**Step 1 · `box_area(box)`** — width × height, clamped at 0 for a degenerate box.
```bash
python3 -c "from homework import box_area as f; print(f([0,0,2,3]), f([5,5,2,2]))"   # expect: 6.0 0.0
```

**Step 2 · `xywh_to_xyxy(box)`** — `[x, y, w, h]` → `[x, y, x+w, y+h]`.
```bash
python3 -c "from homework import xywh_to_xyxy as f; print(f([10,20,30,40]))"         # expect: [10, 20, 40, 60]
```

**Step 3 · `iou(a, b)`** — intersection corners with max/min, then `inter / union`.
```bash
python3 -c "from homework import iou as f; print(round(f([0,0,2,2],[1,1,3,3]),4))"   # expect: 0.1429
```

**Step 4 · `nms(boxes, scores, iou_threshold)`** — greedy: keep top score, drop overlaps, repeat.
One new numpy tool: `np.argsort(scores)` returns the *positions* that would sort the scores low→high
(argmin's big sibling); add `[::-1]` to flip it to high→low. Everything else is a loop and your `iou`.
```bash
python3 -c "from homework import nms as f; print(f([[0,0,10,10],[1,1,10,10],[20,20,30,30]],[0.9,0.8,0.7],0.5))"   # expect: [0, 2]
```

**Grade the whole day:** `pytest -q`  ·  or from the repo root `python tools/grade.py day 42`.

## Check yourself

- Why divide by the *union* instead of just the intersection? (What would "intersection ÷ area of a"
  fail to penalize?)
- NMS with `iou_threshold=0.1` vs `0.7`: which keeps *more* boxes? Why is a low threshold "aggressive"?
- A detector reports 3 boxes on one car. Which function removes the extras, and what does it use to
  decide they're the same car?

## Where this shows up later

Day 43 runs YOLO, whose output is exactly these boxes+scores (and which calls NMS for you). Day 47's
KITTI labels store boxes in this format. Week 8's evaluation uses IoU to decide whether a prediction
"matches" the ground truth. You just built the vocabulary of detection.

**Next:** Day 43 — YOLO: running a real detector.
