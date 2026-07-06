# Day 48 — How to Read a Paper (CVPR / ICCV / ECCV)

> **Week 8 · Research On-ramp** — your advisor's Stage 3 starts with reading top-venue papers. A paper
> is not a textbook chapter to read start-to-finish; it's a structured argument you decode in passes.
>
> *Guided. The exercise is reading one real paper with the method below.*

## Why today matters

Research runs on the literature. You'll read hundreds of papers, and reading them linearly is slow and
demoralizing. The pros use a **three-pass method**: skim for the gist, read for the idea, then (only if
it matters) study for the details. Just as important is reading *critically* — every paper is written to
persuade, so you learn to find the real contribution *and* the limitation the authors soft-pedal.

## The three-pass method (Keshav)

**Pass 1 — the 5-minute skim.** Title, abstract, intro, section headings, figures, conclusion. Answer:
*What problem? What's the claimed contribution? Does this matter to me?* Most papers stop here.

**Pass 2 — the ~1-hour read.** Read the whole thing, but **skip heavy math**. Study the figures and
tables (that's where results live). Answer: *What's the key idea, and what's the evidence?* You should
be able to explain the paper to a labmate now.

**Pass 3 — the deep study** (only for papers you'll build on). Re-derive the method, question every
assumption, note what you'd need to reproduce it. This is where you'd start a reimplementation.

## Read every paper against these questions

- **Contribution:** what's genuinely new here (vs prior work)?
- **Evidence:** what experiments support the claim? On what data/metrics? (Day 50.)
- **Baselines:** what do they compare against — and is the comparison fair?
- **The catch:** what's the limitation? (Check the "Limitations" section, and what they *didn't* test.)
- **Reproducibility:** is there code/weights? (Then Day 40 applies.)

> Read the **figures and tables first**. A paper's real claims live in its result tables; the prose
> explains and defends them. If a table doesn't support the abstract's claim, be skeptical.

## Do this

1. Pick a paper near your interest — e.g. the **YOLO**, **SAM**, **CLIP**, or **Grounding DINO** paper
   (models you already ran in Week 7, so the ideas are concrete). Find it on **arXiv** or the venue site.
2. Do **Pass 1** (5 min) and write 3 sentences: problem, contribution, why it matters.
3. Do **Pass 2** (~1 hr) and fill in the five questions above.
4. Say out loud (or write) the one-sentence idea and the one limitation. If you can't, do Pass 2 again.

## Check yourself

- What do you look at *first* in Pass 1 — and why figures/tables before the prose?
- A paper claims state-of-the-art. What would make that comparison *unfair*, and where would you check?
- When is Pass 3 (deep study) worth it, and when should you stop at Pass 2?

## Where this shows up later

Day 49 turns reading into a *map* of a subfield; Day 50 lets you judge whether a paper's numbers are
meaningful; Day 52 has you reproduce and extend one. Reading fluently is the single highest-leverage
research skill — it's how every project begins.

**Next:** Day 49 — Finding a topic & the literature map.
