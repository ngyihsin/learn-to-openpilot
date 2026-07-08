# Day 36 — Self-Attention & Transformers

> **Week 5 · ML & DL Foundations** — the final foundation, and the most important one for modern AI.
> Attention is the layer inside every Transformer, LLM, and the vision models you'll run in Week 7.
>
> *Assumes basic Python + Days 31–35.*

## Why today matters

A ReLU network (Day 35) treats each input position independently. But meaning is *relational*: a word
depends on other words, a pixel on other pixels, a car on the lane lines around it. **Attention** lets
every position look at every other position and pull in whatever is relevant.

The mechanism is surprisingly simple — three arrays and two matrix multiplies:

- **Q**uery: "what am I looking for?"
- **K**ey: "what do I contain?" — each position advertises itself.
- **V**alue: "what I'll actually hand over if you attend to me."

You match queries to keys (dot product), turn the matches into weights (softmax), and blend the values
by those weights. That's it. **SAM, Grounding DINO, and every VLM in your advisor's Stage 2 are built
from stacks of this layer** — build it once and their papers become readable.

### Key words (plain language)

| Term | In one sentence |
|------|-----------------|
| **Query / Key / Value** | Three views of the data: what I want, what I offer, what I give. |
| **Attention score** | `Q · K` — how well a query matches a key. High = "pay attention here." |
| **Scaling by √dₖ** | Divide scores by `sqrt(d_k)` so large dimensions don't make softmax razor-sharp. |
| **Attention weights** | Softmax of the scores — each query's relevance distribution over all keys (sums to 1). |
| **Self-attention** | Q, K, V all come from the *same* input (each position attending to the whole sequence). |

### See it with intuition first

If one key matches a query far better than the rest, softmax gives it almost all the weight, and the
output is basically that key's value — the query "picked" the most relevant position. If all keys look
identical, the weights are uniform and the output is just the *average* of the values. Everything in
between is a soft, learnable blend. (Both extremes are checked by the grader.)

## Learning goals

By the end you can:

- Implement a **row-wise stable softmax** that turns scores into per-query distributions.
- Build **scaled dot-product attention** and read its weight matrix.
- Build **self-attention** by projecting one input into Q, K, V.

## Do this — three steps

Work top-to-bottom in `homework.py`; run each check in this folder.

**Step 1 · `softmax(z, axis=-1)`** — stable, along an axis: subtract `z.max(axis=axis, keepdims=True)`,
exponentiate, divide by the sum along that axis.
```bash
python3 -c "from homework import softmax as f; print(f([[0,0],[1,3]]))"
# expect rows summing to 1: [[0.5 0.5] [0.11920292 0.88079708]]
```

**Step 2 · `scaled_dot_product_attention(Q, K, V)`** — scores = `Q @ K.T / sqrt(d_k)`, softmax, then
`weights @ V`. Return `(output, weights)`.
```bash
python3 -c "from homework import scaled_dot_product_attention as f; import numpy as np; o,w=f([[10,0]],[[10,0],[0,10]],[[5,6],[7,8]]); print(np.round(w,3), np.round(o,3))"
# expect the query to lock onto key 0: [[1. 0.]] [[5. 6.]]
```

**Step 3 · `self_attention(X, Wq, Wk, Wv)`** — project `Q=X@Wq, K=X@Wk, V=X@Wv`, then attend.
```bash
python3 -c "from homework import self_attention as f; import numpy as np; o,w=f(np.eye(2), np.eye(2), np.eye(2), np.eye(2)); print(o.shape)"
# expect: (2, 2)
```

**Grade the whole day:** `pytest -q`  ·  or from the repo root `python tools/grade.py day 36`.

## Check yourself

- Why divide the scores by `sqrt(d_k)`? What happens to softmax if the scores are huge? (Look back at
  Day 34's stability point.)
- In self-attention, Q, K, and V all come from the same `X`. So what are `Wq`, `Wk`, `Wv` *for* — why
  not just use `X` three times?
- Attention weights form an (n_queries × n_keys) matrix. For a 1000-token sequence attending to itself,
  how big is that matrix — and why is *that* the number the "efficient attention" papers (Week 7) fight to shrink?

## Where this shows up later

This is the last foundation. **Week 6** puts these skills to work on the research toolchain (Git,
Docker, CUDA, reproducing a paper), and **Week 7** runs real Transformer-based vision models — YOLO,
SAM, Grounding DINO — whose cores are stacks of exactly this attention layer.

**Next:** Week 6 — the research toolchain (Day 37 onward).
