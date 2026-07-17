# Gap Probe — Section C answers (Student S3, re-sit)

## C1 — How QAT works mechanically, and why it recovers accuracy

QAT works by "lying to the model during training." Mechanically, during every forward pass
of training, the weights are **fake-quantized**: you snap them onto the int grid first
(scale from the tensor's max magnitude, then round, clamp, dequantize — the recipe is
`qmax = 2**(bits-1) - 1`, `scale = |x|.max() / qmax`, then
`(x/scale).round().clamp(-qmax, qmax) * scale`). So the loss — and therefore every gradient
step — sees exactly the numerics the deployed model will have. The stored weights stay
full-precision float ("shadow weights") and get quantized on the fly each forward.

There is one obstacle: `round()` is a staircase, its gradient is zero almost everywhere, so
backprop through it would stop learning dead. The fix is the **straight-through estimator
(STE)**: use the staircase in the forward pass, but in the backward pass pretend it was the
identity and pass the gradient through untouched. It's crude and biased, but it works. You
build it as a custom `torch.autograd.Function` with two static methods — `forward` returns
the fake-quantized tensor, `backward` returns the gradient unchanged for `x` and `None` for
`bits`.

Why does this *recover* the accuracy PTQ lost? With PTQ you trained the weights to
fine-grained float values and then snapped them to a coarse grid they never knew existed —
the rounding error is a surprise the model never got to react to. With QAT the training
itself sees the rounding, so the weights "learn to be quantized": they settle where the grid
can represent them well. The lesson says at 3 bits PTQ made val MAE roughly 7× worse on the
Day 28c steering regressor, and QAT trained at the same 3 bits recovers most of that damage.

來源: Day 28d README "Why today matters" (fake-quantized forward, "the weights learn to be
quantized: they settle where the grid can represent them well", "snapped them to a coarse
grid they never knew existed", the STE paragraph) and the Hints (the quantize recipe, the
two-static-methods autograd.Function, shadow weights).

## C2 — What is static quantization's calibration set for?

Calibration is for the **activations**, not the weights. The Day 28d hint says static
quantization's calibration set "runs representative inputs through the model to estimate
**activation** ranges" — weights need no data because you already have them in hand. So the
numbers it estimates are the ranges (and from those, I believe the scales) of the
activations, so their quantization can be pre-computed instead of done on the fly like
dynamic does. The hint compares it to picking an ADC's full-scale range, which as an EE I
find comforting: you look at the signal you expect and set the range so it fits.

來源: Day 28d Hints ("it runs representative inputs through the model to estimate
activation ranges (weights need no data — you already have them)... same idea as picking an
ADC's full-scale range") plus Day 28 README ("static (also pre-computes activation scales
from a small calibration set)"). The word "scales" for activations comes from Day 28; the
rest is quoted.

## C3 — Where QAT sits in the pipeline, and the concrete change to the Day 28c loop

QAT sits **during training** — it happens before export and lowering, at the very start of
the train → export → lower → quantize pipeline, not as a step you bolt on after. The Day 28d
learning goals state it plainly: QAT "happens during training, before export/lower."

The concrete change to the Day 28c training loop: almost nothing changes in the loop itself.
The homework file says the provided `train_regressor` "works unchanged for float models AND
QAT models — that's the point: QAT is the same training loop with quantization-aware layers
inside." The one concrete change is in the **model**, which the loop trains: you replace the
plain `nn.Linear` layers with `QATLinear` layers, whose `forward` uses
`STEQuant.apply(self.weight, self.bits)` instead of the raw weights (via
`nn.functional.linear(x, w_q, self.bias)`), while the stored weight stays float. Same MSE
loss, same Adam, same keep-the-best-val-MAE checkpointing — only every forward now goes
through the fake quantizer. (One practical note from the Hints: the QAT training loss looks
jumpy because weights hop between grid points, so you still judge by val MAE of the best
checkpoint, the Day 28c way.)

來源: Day 28d Learning goals ("it happens during training, before export/lower"), Hints
("QATLinear is nn.Linear with one change..."), and the homework.py docstring for
`train_regressor` ("works unchanged for float models AND QAT models").
