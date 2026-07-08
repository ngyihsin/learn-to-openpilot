# Day 51 — Experiments, Tracking & Ablations

> **Week 8 · Research On-ramp** — a result you can't attribute to a cause isn't a result. This is the
> discipline that turns "I changed five things and it got better" into "*this* change caused *that*
> improvement."
>
> *Guided. The exercise is designing a fair comparison and logging it.*

## Why today matters

Research is a sequence of controlled experiments. The two failure modes that ruin them: **changing more
than one thing at a time** (so you can't say what helped) and **not recording what you did** (so you
can't reproduce or trust it). The fix is old and boring and works: change one variable, seed everything
(Day 41), measure with an honest metric (Day 50), and log every run. This is exactly what a reviewer —
and your advisor — will ask you to defend.

## The core ideas

**Controlled comparison.** To claim "X helps," run two configs identical *except* X. Same data, same
seed, same everything else. If two things differ, the result is uninterpretable.

**Ablation study.** Start from your full method and *remove* one component at a time; the drop measures
that component's contribution. It answers "which parts actually matter?" — the table reviewers look for.

| Config | Change from full | Val F1 |
|--------|------------------|--------|
| Full method | — | 0.82 |
| − augmentation | remove data aug | 0.78 |
| − pretraining | train from scratch | 0.71 |
| − attention | swap for plain conv | 0.75 |

Each row changes **one** thing, so each number *means* something.

**Track every run.** For each experiment record: the **config** (Day 41's `merge_config`), the **seed**,
the **metric** (Day 50), and the **commit hash** (Day 38). Then any result is reproducible. Tools:
a CSV/JSON log for a start; **TensorBoard** or **Weights & Biases** when you scale up.

**Read results honestly.** Is a 0.82-vs-0.81 gap real, or noise? Re-run with a few seeds and look at the
*spread*. One lucky seed is not a result.

## Do this

1. Take any model you can run (Day 43's YOLO, or a small net from Week 5).
2. Define a **baseline config** and one **single change** (e.g. with vs without an augmentation, or two
   learning rates). Write both as configs.
3. Run each with a **fixed seed**, record **Day-50 metrics** in a small table, and note the **git commit**.
4. Change the seed and re-run. Is your improvement bigger than the seed-to-seed wobble? Write the
   one-sentence conclusion you could defend.

## Check yourself

- Why change exactly one variable per experiment? What breaks if you change two?
- What's the difference between a *comparison* and an *ablation*, and what question does each answer?
- Your method beats the baseline by 0.5% on one seed. What must you check before claiming it's better?

## Where this shows up later

Day 52 wraps this into a small end-to-end project. In your master's, this loop — hypothesis → controlled
run → honest metric → logged result → conclusion — *is* the research. Everything else is setup.

**Next:** Day 52 — Capstone: your mini research project.
