# Day 25e — Building a Transformer Block

> **Week 4 · PyTorch / ExecuTorch** — the attention → residual → norm → feed-forward sandwich behind every VLM.

## Why today matters

On **Day 36** you derived scaled-dot-product self-attention on paper: scores, a softmax, a
weighted sum of values. Today you turn that math into a real, reusable PyTorch module — the
**transformer encoder block** — and discover it's the exact building block inside CLIP, ViT,
and every vision-language model the lab works on. A ViT is just this block stacked a dozen
times; a VLM is two stacks glued together. Learn to build one cleanly and the architectures in
next week's papers stop looking like magic and start looking like Lego.

You'll do it in two passes. First you'll implement the attention kernel *from scratch* so you
own the math end to end. Then you'll wrap PyTorch's battle-tested `nn.MultiheadAttention` into
the full block — attention, a residual connection, a layer-norm, a feed-forward network,
another residual, another norm — the sandwich that repeats everywhere.

## Learning goals

By the end you can:

- Implement `softmax(QKᵀ/√d)V` yourself and check it against PyTorch's reference kernel.
- Explain why the `√d` scaling is there (keep the softmax out of its saturated, near-zero-gradient regime).
- Assemble a **post-norm** encoder block and say what each of the four pieces —
  attention, residual, layer-norm, feed-forward — contributes.

## Do this

1. **Homework (~60 min).** Open `homework.py` and implement:
   - `scaled_dot_product_attention(q, k, v)` — the kernel, by hand, returning
     `(output, attn_weights)`.
   - `TransformerEncoderBlock(nn.Module)` — the full block wrapping
     `nn.MultiheadAttention`.

2. **Grade it.**
   ```bash
   pytest -q
   # or:  python tools/grade.py day 25e
   ```
   The grader confirms your attention matches PyTorch's own
   `torch.nn.functional.scaled_dot_product_attention`, that your attention weights form a
   proper probability distribution, and that a full block produces the right shape and is
   trainable (gradients flow to its parameters).

## Setup note

Week 4 needs PyTorch. CPU-only is fine for everything here:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

(If torch isn't installed, this day's grader **skips** rather than fails.)

## Hints

- `q @ k.transpose(-2, -1)` gives the `(B, T, T)` score matrix: entry `(i, j)` is how much
  token `i` should attend to token `j`. Softmax over `dim=-1` so **each query's** weights sum to 1.
- Divide by `math.sqrt(d)` *before* the softmax, not after. Skip it and, for large `d`, the
  scores blow up, the softmax saturates to nearly one-hot, and gradients vanish — the classic bug.
- `nn.MultiheadAttention(..., batch_first=True)` expects `(B, T, d)` — the same layout as your
  data. It returns `(attn_output, attn_weights)`; you only need the first, so
  `attn_out, _ = self.attn(x, x, x)`. Passing `x` three times is what makes it **self**-attention.
- **Post-norm** means normalize *after* the residual: `x = LayerNorm(x + sublayer(x))`. The
  `x +` residual is the highway that lets gradients reach the bottom of a deep stack — the #1
  reason transformers train at all.

## Check yourself

- Why must the softmax be over the last dimension and not the second-to-last? (Whose weights
  are supposed to sum to 1 — each query's, or each key's?)
- What would break if you dropped the residual connections and just wrote
  `x = LayerNorm(sublayer(x))`? (Think about a 24-layer stack and where the gradient goes.)
- The feed-forward network runs on each token *independently*. So where does one token ever get
  to look at another? (Answer: only in the attention sublayer — that's the whole point.)

## Where this shows up later

This block *is* the encoder in Week 7's Vision Transformer and the text/vision towers of CLIP.
When you read a VLM paper next week and it says "a stack of N transformer layers," this is the
layer. Day 26 exports a trained model to a portable format, and Day 27 lowers it to
**ExecuTorch** for on-device inference — the same journey a transformer takes from a research
notebook to running on a comma device in openpilot.

**Next:** Day 26 — Exporting Models.
