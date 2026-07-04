# Day 09 — Context Switching & the Process Lifecycle

> **Week 2 · Operating Systems** — a process is a little state machine the OS drives.

## Why today matters

On Day 08 you scheduled processes by their CPU bursts. But a real process isn't simply
"waiting then running" — it moves through a defined set of **states**: it's admitted (`new →
ready`), dispatched onto the CPU (`ready → running`), preempted when its time slice expires
(`running → ready`), blocked when it asks for I/O (`running → waiting`), woken when the I/O
finishes (`waiting → ready`), and finally exits (`running → terminated`). Every time the OS
takes the CPU from one process and gives it to another, that's a **context switch** — cheap but
not free, and a big part of why scheduling policy matters.

Modeling this as a state machine is exactly how kernels think about it, and it makes illegal
moves (dispatching a blocked process, running a dead one) impossible by construction.

## Learning goals

By the end you can:

- Name the process states and the events that move between them.
- Implement the lifecycle as a state machine that rejects illegal transitions.
- Explain what a context switch is and count how many times a process was dispatched.

## Do this

1. **Concept + visualization (~20 min).** `jupyter lab lesson.ipynb` — see the state diagram and
   watch a process walk through a realistic lifecycle (run, get preempted, block on I/O, resume,
   exit), tallying context switches.
2. **Homework (~45 min).** Implement `can`, `on`, and `run` in `homework.py`. The transition
   table is given — your job is the validation, history, and counting logic.
3. **Grade it.** `pytest -q`  ·  or `python tools/grade.py day 9`.

## Hints

- A transition is legal iff `(state, event)` is a key in `TRANSITIONS`. Reject everything else
  with `InvalidTransition` and **don't change state** on a rejected event.
- Count a context switch on every `dispatch` (each time this process is handed the CPU).

## Check yourself

- Why can't a `waiting` process go straight to `running`? What must happen first, and why does
  that model reality?
- A context switch saves/restores registers, the program counter, and memory mappings. Why does
  a scheduler with a tiny time quantum (lots of switches) waste CPU?
- Where does the scheduler from Day 08 fit into this diagram? (Hint: it chooses who gets the
  `dispatch`.)

## Where this shows up later

This state model underlies Day 11 (threads share a process but have their own run states),
Day 12 (a process in `waiting` is often blocked on a lock/semaphore), and Day 13 (deadlock = a
set of processes all stuck in `waiting` forever).

**Next:** Day 10 — Virtual Memory & Paging (already built), then Day 11 — Threads & the GIL.
