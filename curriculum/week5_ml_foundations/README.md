# Week 5 — Machine Learning & Deep Learning Foundations

> *Goal: go from "I can write PyTorch" (Week 4) to "I understand what the model is actually
> doing and why it generalizes." This is the runway the advanced 2022 lectures assume.*

Week 4 taught you the *mechanics* of PyTorch — tensors, autograd, `nn.Module`, training loops.
This week teaches the *ideas underneath*: what "learning" even means, why a model trained on one
set of data works on data it has never seen, and the two mechanisms (deep networks and
self-attention) that everything modern is built from.

It exists because the course we're building toward — **Hung-yi Lee's 【機器學習 2022】** (covered
day-by-day in Week 6) — opens at an *advanced* level. It assumes you already know regression,
gradient descent, classification losses, backprop, and basic self-attention. Week 5 is that
assumed background, taught from zero.

| Day | Topic | Format | You'll build | Status |
|-----|-------|--------|--------------|--------|
| 31 | **numpy & array thinking** | notebook + pytest | Vectorized math, masks, axes, argmin, seeded rng | ✅ |
| 32 | **The ML framing: model, loss, generalization** | notebook + pytest | Train/val split, polynomial fit, model selection | ✅ |
| 33 | **Regression & gradient descent** | notebook + pytest | A linear model trained by hand-written gradient descent | ✅ |
| 34 | **Classification: softmax & cross-entropy** | notebook + pytest | Softmax, cross-entropy, one-hot, argmax, accuracy | ✅ |
| 35 | **Neural networks & backpropagation** | notebook + pytest | A 2-layer net with hand-derived backprop (gradient-checked) | ✅ |
| 36 | **Self-attention & Transformers** | notebook + pytest | Scaled dot-product self-attention from scratch | ✅ |

**Why this order:** Day 31 gives you the *tool* (numpy) everything else is written in. Day 32 gives
you the vocabulary (*model / loss / generalization*) every later day reuses. Days 33–35 build the
learning machine bottom-up. Day 36 is the one mechanism the 2022 course leans on hardest — you build
it here so its "variants" lecture (in Week 7) makes sense.

**Next:** Week 6 — Hung-yi Lee's 2022 course, one lecture per day.
