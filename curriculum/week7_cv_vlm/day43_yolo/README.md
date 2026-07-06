# Day 43 — YOLO: Running a Real Detector

> **Week 7 · CV & VLMs** — YOLO ("You Only Look Once") is the workhorse object detector: fast, accurate,
> and the first model your advisor named. Today you run it and read its output.
>
> *Guided, hands-on. Apply the Day-40 reproduction checklist.*

## Why today matters

Object detection = "what objects are in this image, and *where*?" — output as labeled boxes with
confidence scores. It's the backbone of driving perception (cars, pedestrians, signs). Modern YOLO
(v8/v11 and beyond) ships as a clean Python package, so it's the perfect first real model to run: you'll
see the whole pipeline — image in, boxes out — and the exact data format Day 42's IoU/NMS operate on.

## What YOLO gives you

For each detected object: a **box** `[x1, y1, x2, y2]`, a **class label** ("car", "person", ...), and a
**confidence** in `[0, 1]`. Internally it runs Day 42's **NMS** to remove duplicate boxes before
returning them. You already understand the output — today you generate it.

## Do this — run YOLO

**1 · Install (in a fresh venv — Day 37 habit).**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install ultralytics        # the maintained YOLO package; pulls in torch
```

**2 · Run inference on a sample image.**
```bash
# Downloads a small pretrained model (yolo11n) and a demo image the first time:
yolo predict model=yolo11n.pt source='https://ultralytics.com/images/bus.jpg'
# results (annotated image) are saved under runs/detect/predict/
```

**3 · Do it from Python and inspect the raw output** — this is the part that matters:
```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
results = model("https://ultralytics.com/images/bus.jpg")
r = results[0]
for box in r.boxes:
    xyxy = box.xyxy[0].tolist()          # [x1, y1, x2, y2]
    cls  = model.names[int(box.cls[0])]  # e.g. "bus", "person"
    conf = float(box.conf[0])            # confidence 0..1
    print(f"{cls:10s} {conf:.2f}  box={ [round(v) for v in xyxy] }")
```
Two bits of syntax worth decoding: `box.xyxy[0]` is just *reaching into an object's attributes* —
`results` holds boxes, each box holds its coordinates, `[0]` takes the first row, `.tolist()` converts
to a plain list. And inside the f-string, `{conf:.2f}` means "2 decimal places", `{cls:10s}` means
"pad the name to 10 characters" — formatting hints after a `:`, nothing more.

**4 · Connect it back to Day 42.** Those `xyxy` boxes are exactly what your `iou` and `nms` take. Try
lowering the confidence threshold (`model(..., conf=0.1)`) and watch duplicate/low-quality boxes appear —
the mess NMS exists to clean up.

## ✅ You did it right if…

- Step 2 saved an annotated image under `runs/detect/predict/` showing a labeled **bus** box and
  several **person** boxes.
- Step 3 prints one line per object with a class name, a confidence around 0.3–0.95, and four
  plausible pixel coordinates (inside the image's width/height).
- With `conf=0.1`, the box count goes **up** (more low-confidence detections) — exactly the Day 42
  threshold trade-off.

## If the install fails

- `pip install ultralytics` pulls in **torch** (~hundreds of MB). If the download stalls or 403s,
  install torch first from default PyPI (`pip install torch`) and retry — and remember this
  environment's rule from `docs/HANDOFF.md`: use the *default* index, not the pytorch CDN.
- No GPU is fine: `yolo11n` (nano) runs on CPU in seconds per image. Start there, not with the large models.
- Any other error: search the **ultralytics GitHub Issues** for the exact message (Day 40 habit).

## Check yourself

- YOLO returns a box, a class, and a confidence. Which two did you build tools for on Day 42?
- You lower the confidence threshold and suddenly get 3 overlapping boxes on one car. What algorithm
  removes the duplicates, and what parameter controls how aggressively?
- The model file is `yolo11n.pt` (`n` = nano). Why might you start with the smallest model on a CPU
  before reaching for the large one?

## Where this shows up later

Day 44 (SAM) *segments* what YOLO *detects*; Day 45 (Grounding DINO) detects from a text prompt instead
of fixed classes. On KITTI/NuScenes (Day 47) you'll run detection on real driving frames — the same task
openpilot's vision stack performs on the road.

**Next:** Day 44 — Segmentation with SAM 2 / SAM 3.
