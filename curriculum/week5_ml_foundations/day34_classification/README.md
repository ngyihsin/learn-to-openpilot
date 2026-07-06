# Day 34 — Classification: Softmax & Cross-Entropy

> **Week 5 · ML & DL Foundations** — regression predicts a *number*; classification predicts a
> *category*. Today you build the output layer and loss that nearly every classifier on earth uses.
>
> *Assumes basic Python + Day 31 numpy. All the ML ideas are explained here.*

## Why today matters

"Is this a car, a pedestrian, or a cyclist?" is a **classification** question, and openpilot's
perception — like every model you'll run in Week 7 (YOLO, SAM, Grounding DINO) — answers thousands of
them per second. A classifier doesn't output a clean label directly. It outputs raw scores called
**logits**, one per class. Two tools turn those into something usable and trainable:

- **softmax** squashes the logits into **probabilities** (all positive, summing to 1),
- **cross-entropy** turns "how much probability did you put on the *true* class?" into a loss to
  minimize (with the gradient descent you learned on Day 33).

Get these two right and you understand the business end of almost any neural classifier.

### Key words (plain language)

| Term | In one sentence |
|------|-----------------|
| **Logits** | Raw, unbounded scores the model emits — one per class. Can be negative. |
| **Softmax** | Converts logits → probabilities: exponentiate, then divide by the total. Bigger logit ⇒ bigger share. |
| **One-hot** | The true label as a vector: class 2 of 4 → `[0, 0, 1, 0]`. |
| **Cross-entropy** | The loss: `-log(probability the model gave the true class)`. Confident & right → ~0; confident & wrong → huge. |
| **argmax** | The index of the largest value — how you read the final predicted class off the probabilities. |

### See it with real numbers first

Logits `[1, 1, 3]` → exponentiate `[2.72, 2.72, 20.1]` → divide by their sum `25.5` →
probabilities ≈ `[0.11, 0.11, 0.78]`. The class with logit 3 gets most of the probability.
If the true class was #2 (the last one), the loss is `-log(0.78) ≈ 0.25` — small, because we were
mostly right. If the true class had been #0, the loss would be `-log(0.11) ≈ 2.2` — much bigger.

**Stability note:** `exp(1000)` overflows to infinity. Subtracting the max logit first
(`exp(z - max(z))`) gives identical probabilities without the overflow — always do it.

## Learning goals

By the end you can:

- Turn logits into a probability distribution with a **numerically stable softmax**.
- Compute **cross-entropy** loss and explain why wrong-and-confident is punished hardest.
- Read a prediction with **argmax** and score a batch with **accuracy**.

## Do this — five small steps

Work top-to-bottom in `homework.py`; run each check in this folder.

**Step 1 · `softmax(z)`** — `e = np.exp(z - z.max()); return e / e.sum()`.
```bash
python3 -c "from homework import softmax as f; print(f([0,0]))"                 # expect: [0.5 0.5]
python3 -c "from homework import softmax as f; print(f([1000,1000]))"           # expect: [0.5 0.5]  (no nan!)
```

**Step 2 · `one_hot(label, num_classes)`** — zeros with a single 1.0 at `label`.
```bash
python3 -c "from homework import one_hot as f; print(f(2,4))"                   # expect: [0. 0. 1. 0.]
```

**Step 3 · `cross_entropy(probs, label)`** — `-log(clip(probs[label], 1e-12, 1.0))`.
```bash
python3 -c "from homework import cross_entropy as f; print(round(f([0.5,0.5],0),4))"   # expect: 0.6931
```

**Step 4 · `predict(probs)`** — `int(np.argmax(probs))`.
```bash
python3 -c "from homework import predict as f; print(f([0.1,0.7,0.2]))"         # expect: 1
```

**Step 5 · `accuracy(pred_labels, true_labels)`** — mean of elementwise equality.
```bash
python3 -c "from homework import accuracy as f; print(f([0,1,1,0],[0,1,0,0]))"  # expect: 0.75
```

**Grade the whole day:** `pytest -q`  ·  or from the repo root `python tools/grade.py day 34`.

## Check yourself

- Why does softmax exponentiate instead of just dividing each logit by the sum? (What would happen
  with negative logits?)
- Cross-entropy punishes a confident *wrong* answer far more than an unsure one. Why is that the
  behavior you *want* when training?
- `softmax` gives probabilities but `predict` just takes the argmax. When would you care about the
  actual probabilities, not only the winning class? (Hint: a detector's confidence threshold.)

## Where this shows up later

Day 35 puts a softmax + cross-entropy head on top of a small neural network and trains the whole thing
with backprop. In Week 7, every detector you run reports exactly these: class probabilities and a
confidence you threshold on.

**Next:** Day 35 — Neural networks & backpropagation.
