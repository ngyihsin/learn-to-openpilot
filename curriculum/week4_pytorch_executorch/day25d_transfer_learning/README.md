# Day 25d — Transfer Learning & Fine-Tuning

> **Week 4 · PyTorch / ExecuTorch** — reuse a pretrained backbone, swap the head, and fine-tune just what you need.

## Why today matters

This is *the* most common workflow in real research — you will do it far more often than you
train a model from scratch. Someone already spent thousands of GPU-hours teaching a network to
see: a **YOLO** backbone that has learned edges, textures, and object parts; a **CLIP** encoder
that has learned to line up images with language. You don't throw that away. You **freeze** most
of it, **replace the final head** with one shaped for *your* task (your number of classes, your
output), and **fine-tune** — usually just the new head, sometimes the top few layers too.

Why it works: the early layers learned general features (a car looks like a car whether it's in
ImageNet or in openpilot's road-facing camera), so you only need to re-teach the last, task-specific
part. It's faster, needs far less data, and reaches higher accuracy than training from zero.

In real code you'd grab a backbone like this:

```python
import torchvision
model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
```

That downloads pretrained weights. To keep this lesson **offline and instant** we use a tiny
local model instead — but the freeze / swap-head / fine-tune moves you practice here are exactly
the ones you'll run on ResNet, YOLO, or CLIP.

## Learning goals

By the end you can:

- Explain what `requires_grad` does to a parameter, and how freezing a module stops it from
  training.
- Freeze and unfreeze a whole module by walking its `.parameters()`.
- Replace a model's `.head` with a new layer sized for your task, and confirm only the new head
  is trainable.
- Run a fine-tuning step and *prove* the frozen backbone doesn't move while the head does.

## Do this

1. **Homework (~60 min).** Implement `freeze`, `unfreeze`, `replace_head`,
   `trainable_param_names`, and `finetune_step` in `homework.py`. Each operates on any model that
   exposes a `.head` attribute.

2. **Grade it.**
   ```bash
   pytest -q
   # or:  python tools/grade.py day 25d
   ```
   The grader builds a tiny two-layer net, freezes the backbone, swaps the head, and checks that
   after one step the backbone weights are **unchanged** while the head weights **moved**.

## Setup note

Week 4 needs PyTorch. CPU-only is fine for everything here:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

(If torch isn't installed, this day's grader **skips** rather than fails.)

## Hints

- A "frozen" parameter is just a parameter with `requires_grad = False`. Autograd then computes
  **no gradient** for it, so the optimizer has nothing to update — it stays put.
- Freezing is per-*parameter*, but you apply it to a whole *module* by looping over
  `module.parameters()`.
- `replace_head` is a one-liner: assign the new layer to `model.head`. A freshly built `nn.Linear`
  has `requires_grad=True` params by default, so the new head trains automatically.
- The #1 beginner bug: building the optimizer over `model.parameters()` *before* you freeze. That's
  actually fine here — a frozen param gets no gradient, so the optimizer skips it — but if you
  *unfreeze* later, make sure those params are in the optimizer, or they still won't move.
- `trainable_param_names` should read from `model.named_parameters()` and return a **sorted** list
  of the names where `requires_grad` is True.

## Check yourself

- Why does an optimizer built over *all* parameters still leave the frozen ones untouched?
- If you freeze the backbone but forget to give the new head to the optimizer, what happens when
  you train?
- When would you unfreeze the top few backbone layers too, instead of only training the head?

## Where this shows up later

Week 7 (Days 42–47) is computer vision and vision-language models — you'll take a pretrained
vision backbone and a CLIP-style encoder and adapt them exactly this way. Openpilot's driving
model is itself a shared backbone with multiple task **heads** (path, lead car, lane lines); when
the team adds a new prediction, they attach a new head rather than retrain the whole network.
Transfer learning is the bridge from "someone's pretrained model" to "a model that does my task."

**Next:** Day 25e — Building a Transformer Block.
