# Day 49 — Finding a Topic & the Literature Map

> **Week 8 · Research On-ramp** — a research topic isn't handed to you fully formed; you *carve* it out
> of a broad interest by reading and mapping what exists. Today you learn how.
>
> *Guided. The exercise is building a small literature map around one question.*

## Why today matters

The hardest part of starting research isn't the coding — it's finding a question that is *interesting*,
*doable*, and *not already solved*. Your advisor mentioned this happens after you've built background and
the lab's projects (NSTC / ITRI) come into focus. But you can practice the skill now: take a broad area,
read into it, and find the gap. A good topic lives at the intersection of "I find this interesting,"
"the lab/hardware can support it," and "the field hasn't nailed it yet."

## From interest to question

Turn a vague interest into a sharp, answerable question by adding constraints:

- ❌ "I want to work on self-driving perception." (too broad — a whole field)
- 🔸 "Object detection on driving data." (better, still broad)
- ✅ "Does adding radar to a camera-only detector improve small-object detection at night on NuScenes?"
  (specific: a method, a dataset, a metric, a condition — you could actually test this.)

A researchable question names **what you'd change**, **what you'd measure**, and **on what data**.

## Building a literature map (the "related work" web)

1. **Start from an anchor paper** (a survey, or a strong recent paper in the area — Day 48).
2. **Go backward:** its citations → the foundational work it builds on.
3. **Go forward:** who cites *it*? (Google Scholar's "Cited by", Semantic Scholar, Connected Papers.)
4. **Cluster** what you find into a few groups ("approaches that do X" vs "that do Y") and note, for each,
   its **result, dataset, and limitation**.
5. **Find the gap:** what has *no one* combined, tested on your data, or measured? That gap is a topic.

> Tools that make this fast: **arXiv**, **Google Scholar** ("Cited by"), **Semantic Scholar**,
> **Papers with Code** (results + code per benchmark), **Connected Papers** (visual citation graph).

## Do this

1. Pick a broad interest (say, "vision-language models for driving").
2. Find **one anchor paper** and read it Pass-2 (Day 48).
3. Collect **5–8 related papers** (backward + forward citations). For each, one line:
   *idea · dataset · metric · limitation.*
4. Cluster them into 2–3 groups and write one sentence naming a **gap** — something untested or
   uncombined. That's a candidate topic.

## Check yourself

- What three properties make a research question good? Rewrite a too-broad interest into one.
- "Backward" vs "forward" citation search — what does each reveal about a paper's place in the field?
- Why note each paper's **limitation and dataset**, not just its idea? (How does that surface a gap?)

## Where this shows up later

Day 51 designs experiments to answer a question like the one you framed; Day 52 scopes a mini project
around a real gap. In your master's, this is the loop you'll run for every project: read → map → find the
gap → test it.

**Next:** Day 50 — Evaluating models: metrics done right.
