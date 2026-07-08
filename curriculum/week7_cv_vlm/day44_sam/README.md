# Day 44 — Segmentation with SAM 2 / SAM 3

> **Week 7 · CV & VLMs** — detection gives you a *box*; segmentation gives you the exact *pixels*.
> SAM (Segment Anything Model) is the foundation model for this, and second on your advisor's list.
>
> *Guided, hands-on. Apply the Day-40 reproduction checklist.*

## Why today matters

A box around a pedestrian is coarse; a **mask** — the precise set of pixels that *are* the pedestrian —
is what you need for accurate perception, occlusion reasoning, and measuring free space on the road.
**SAM** is *promptable*: you give it a point or a box, and it returns the mask of the object there. SAM 2
extends this to video (tracking a mask across frames). It's a Transformer under the hood — the
self-attention you built on Day 36, scaled up and pretrained on millions of masks.

## Key idea: promptable segmentation

- **Input:** an image + a prompt (a click point, a box, or "everything").
- **Output:** one or more **masks** (a boolean array the size of the image) + a quality score.
- **Why "foundation model":** it was trained on a huge, diverse mask dataset, so it segments objects it
  was never explicitly taught — including driving scenes.

## Do this — run SAM

**1 · Install (fresh venv).**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "sam2 @ git+https://github.com/facebookresearch/sam2.git"   # follow the repo's README for the current install
# then download a checkpoint the README points to, e.g. sam2 hiera-tiny
```

**2 · Load an image as a numpy array** (the step every CV demo assumes — here it is explicitly):
```python
from PIL import Image           # pip install pillow  (usually already present)
import numpy as np
image = np.array(Image.open("photo.jpg").convert("RGB"))
print(image.shape)              # (H, W, 3) — height x width x 3 color channels. Day 31's arrays!
```

**3 · Segment from a point prompt (Python sketch — mirror the repo's demo):**
```python
from sam2.sam2_image_predictor import SAM2ImagePredictor
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")
predictor.set_image(image)                       # a HxWx3 numpy array
masks, scores, _ = predictor.predict(
    point_coords=[[500, 375]],                   # click one point on an object
    point_labels=[1],                            # 1 = foreground
)
print("mask shape:", masks[0].shape, " score:", float(scores[0]))
```

**4 · Understand the output.** `masks[0]` is a boolean `HxW` array — `True` where the object is. Overlay
it on the image to see the segmentation. Try different prompt points and watch the mask change.

**5 · The YOLO → SAM combo** (a real pipeline): run YOLO to get a *box*, feed that box to SAM as the
prompt to get a precise *mask*. Detection finds *what/where*; segmentation refines it to *exact pixels*.

## ✅ You did it right if…

- `image.shape` prints `(H, W, 3)` for your photo.
- `masks[0].shape` equals `(H, W)` — same height/width as the image — and `masks[0].sum()` is a big
  number (the count of `True` pixels; 0 would mean an empty mask).
- Clicking a point on a *different* object gives a visibly different mask.

## If the install fails (it's the fragile part)

The `git+https` install and checkpoint download are the two usual failure points. In order:
1. Re-read the **sam2 repo README** — install steps change between releases; follow its current words,
   not this page's sketch.
2. Fallback that avoids the git install entirely: SAM is also on Hugging Face —
   `pip install transformers` then use `facebook/sam2-hiera-tiny` via `transformers`' Sam classes.
3. Still stuck? Check the repo's **GitHub Issues** for your exact error (Day 40 habit), and do the
   concept work today: run steps 2 and 4 mentally on the README's figures — the promptable-mask idea
   is the lesson; the install is logistics.

## Check yourself

- What's the difference between a detection **box** and a segmentation **mask**? When do you need the mask?
- SAM is "promptable." What are the three prompt types, and what does `point_labels=1` vs `0` mean?
- SAM segments objects it was never explicitly trained on. What property of its training makes that
  possible? (Same reason it's called a *foundation* model.)

## Where this shows up later

Day 45 (Grounding DINO) supplies *text-driven* boxes you can feed to SAM ("segment the traffic lights").
Segmentation masks on driving data feed the free-space and obstacle reasoning that perception stacks —
openpilot included — depend on.

**Next:** Day 45 — Open-vocabulary detection: Grounding DINO.
