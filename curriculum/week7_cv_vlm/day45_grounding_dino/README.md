# Day 45 — Open-Vocabulary Detection: Grounding DINO

> **Week 7 · CV & VLMs** — YOLO detects a *fixed* list of classes. Grounding DINO detects **anything you
> can describe in words** — the third model on your advisor's list, and the bridge to vision-language.
>
> *Guided, hands-on. Apply the Day-40 reproduction checklist.*

## Why today matters

Classic detectors are trained on a fixed set (COCO's 80 classes). Ask for "a traffic cone" or "the
person on a bicycle" and they can't — it's not in the list. **Grounding DINO** takes a **text prompt**
and finds matching objects, even categories it never saw a label for. This "open-vocabulary" ability is
what makes modern perception flexible, and it's your first true **vision-language** model: it fuses a
text encoder and an image encoder with — again — self-attention (Day 36).

## Key idea: text prompt → boxes

- **Input:** an image + a text prompt like `"traffic light . car . pedestrian ."` (phrases separated by
  periods).
- **Output:** boxes + the phrase each one matched + a confidence.
- **Why it matters:** you change *what* it detects by changing *text*, with no retraining. That's the
  leap from "closed set" to "open vocabulary."

## Do this — run Grounding DINO

**1 · Install (fresh venv).** The easiest route is via Hugging Face Transformers:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install transformers torch pillow
```

**2 · Detect from a text prompt (Python sketch):**
```python
from transformers import pipeline
detector = pipeline(model="IDEA-Research/grounding-dino-tiny", task="zero-shot-object-detection")
image = "https://ultralytics.com/images/bus.jpg"
results = detector(image, candidate_labels=["a bus", "a person", "a traffic light"])
for r in results:
    print(f"{r['label']:16s} {r['score']:.2f}  box={r['box']}")
```

**3 · Change the words, change the detections.** Re-run with `candidate_labels=["the front wheel"]` or
`["anyone wearing a backpack"]`. No retraining — the *language* is the interface. That's the whole point.

**4 · Compose the perception stack.** Grounding DINO (text → box) + SAM (box → mask) = "segment whatever
I can name." This prompt-driven pipeline is a common modern baseline — and exactly the kind of system a
vision-language lab builds on.

## Check yourself

- What can Grounding DINO detect that YOLO (trained on COCO) fundamentally cannot, and why?
- The interface is *text*. What do you change to make it detect a new kind of object — the model weights,
  or the prompt?
- Why is Grounding DINO called a *vision-language* model, while plain YOLO is not?

## Where this shows up later

Day 46 (CLIP) opens up *how* images and text get compared in a shared space — the mechanism underneath
open-vocabulary detection. VLMs like this are the research frontier your lab works in; being able to run
one end-to-end is a Stage-2 milestone.

**Next:** Day 46 — CLIP & vision-language models.
