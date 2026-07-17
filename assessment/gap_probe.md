# Gap Probe — Advanced Topics (diagnostic, 30 pts)

**Purpose (not shown to students as such):** this probe is *expected* to be hard. It measures
how far the current curriculum carries a student into three candidate topics — temporal
models, real data augmentation, and quantization-aware training — to decide which deserves
the next lesson. Students: answer with your best reasoning from what the course taught; do
not leave blanks.

Rules: closed book except the course materials in `curriculum/`. Prose and/or pseudocode.

---

## Section A — Temporal context (10 pts)

**A1 (4 pts).** Your Day 28c steering regressor sees one frame at a time. Now the task is to
also estimate the **lead car's relative speed**. Explain why a single-frame model
fundamentally cannot do this, then propose **two different concrete ways** to give a model
temporal context, describing each in enough detail that a classmate could build it.

**A2 (3 pts).** One of your proposals keeps internal **state** between inference calls at
20 Hz. Name the new *deployment* complications this creates versus the stateless Day 28c
model (think: what the runtime must now carry between calls, what export must capture, and
what must happen when a new drive starts).

**A3 (3 pts).** You build a dataset of logged drives for this temporal model. Day 25 taught
you to shuffle the training set every epoch. What goes wrong if you keep shuffling
*individual frames*, and what should the unit of shuffling become?

## Section B — Data augmentation on real frames (10 pts)

**B1 (4 pts).** To fight overfitting you augment the Day 28c steering dataset by randomly
**flipping frames horizontally**. What *must* happen to the steering label of a flipped
frame, and why? Give one augmentation that is safe *without* touching the label, and one
more (besides flipping) that *would* require changing it.

**B2 (3 pts).** Do you apply augmentation to the training set, the validation set, or both?
Justify from the course's rules about honest evaluation.

**B3 (3 pts).** Where in the Dataset/DataLoader pipeline does augmentation belong so that a
sample is augmented **differently each epoch**, and why would augmenting the dataset once,
ahead of time, throw away most of the benefit?

## Section C — Quantization-aware training (10 pts)

**C1 (4 pts).** You quantize the Day 28c regressor with dynamic int8 quantization and its
val MAE doubles — unacceptable. Day 28 named two stronger flavors. For **QAT** specifically:
describe mechanically how it works during training and why that lets the model *recover* the
accuracy that post-training quantization lost.

**C2 (3 pts).** **Static** quantization needs a small "calibration set" that dynamic
quantization does not. What is calibration *for* — what numbers does it estimate, and about
what (weights or activations)?

**C3 (3 pts).** In the Week 4 pipeline (train → export → lower → quantize), where does QAT
sit — and name the concrete change it makes to the training loop you wrote on Day 28c.
