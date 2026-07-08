# Day 52 — Capstone: Your Mini Research Project

> **Week 8 · Research On-ramp** — the finish line, and the real start. You've gone from "what is a
> variable?" to running vision-language models. Now do a small slice of *actual research*:
> **reproduce a baseline, then change one thing and measure it.**
>
> *Guided. The deliverable is a tiny project you could show your advisor.*

## Why today matters

Everything in this course was building to this: the ability to take a published method, get it running,
and push it one honest step further. That "reproduce-then-tweak" loop is the atom of research — a
master's thesis is just many of these, stacked. Doing one end-to-end, however small, is what turns a
student into a researcher. It's also the perfect thing to show a prospective advisor.

## The project shape (keep it small)

**Reproduce → tweak → evaluate → write up.** Deliberately modest scope; depth over breadth.

1. **Pick a baseline you can run.** A Week-7 model is ideal: YOLO on a few KITTI/NuScenes frames,
   CLIP zero-shot classification, or Grounding DINO on a handful of images. Something that runs on your
   hardware in minutes, not days.
2. **Reproduce it** (Day 40 checklist): clone, env, weights, run, confirm sane output on sample inputs.
3. **Form one small hypothesis.** "Prompt phrasing changes Grounding DINO's recall." "Confidence
   threshold trades YOLO's precision for recall." "CLIP's caption wording changes its accuracy."
4. **Run the controlled experiment** (Day 51): baseline vs your one change, fixed seed, a few inputs.
5. **Evaluate honestly** (Day 50): precision / recall / F1, not a vibe. A tiny results table.
6. **Write it up** (one page): question, method, result table, one figure, what you'd do next.

## A reproducibility checklist for your write-up

- [ ] A `README` with exact steps to rerun (env, commands).
- [ ] Seeds fixed; config captured (Day 41).
- [ ] The git commit hash for each result (Day 38).
- [ ] `.gitignore` for weights/data; a link to download them instead (Day 38).
- [ ] A results table with an honest metric (Day 50), ideally across a couple of seeds (Day 51).

## Do this

Scope one project from the list above (or your own), and run all six steps. Aim for something that fits
in a **single afternoon** and a **one-page report**. The point isn't a breakthrough — it's the complete
loop, done cleanly enough that someone else could reproduce it.

## Check yourself

- Can someone else rerun your project from your README alone? If not, what's missing?
- Is your conclusion supported by the results table, and does it survive a second seed?
- What's the single next experiment you'd run — and why that one?

## Where this goes next

This is the on-ramp, not the destination. From here: read more papers (Day 48), map a subfield (Day 49),
and bring a reproduced baseline + a question to your advisor. You started at "what is a program?" and
finished able to reproduce and extend modern vision-language research — the exact footing a master's
project starts from. The road is yours.

**The end of the built curriculum — and the beginning of your research.**
