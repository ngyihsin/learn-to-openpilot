# Day 10 — Virtual Memory & Paging

> **Week 2 · Operating Systems** — the illusion that every process owns all of memory.

## Why today matters

When your program reads address `0x7ffe...`, that's a *virtual* address — a fiction the OS and
CPU maintain. Real memory is carved into fixed-size **pages**, and a **page table** maps each
virtual page to a physical frame (or to "not currently in RAM"). Touch a page that isn't
resident and you get a **page fault**: the OS loads it, possibly **evicting** another page.

This machinery is why processes can't stomp on each other's memory, why a program can use more
memory than you physically have, and why "why is this suddenly slow?" is often "it started
thrashing the page cache." Today you build the translation step and two eviction policies —
and meet **Belady's anomaly**, a result that surprises almost everyone.

## Learning goals

By the end you can:

- Translate a virtual address to a physical one using page size + page table (vpn / offset).
- Implement **FIFO** and **LRU** page replacement and count page faults.
- Explain why LRU can never be hurt by adding frames, but **FIFO can** (Belady's anomaly).

## Do this

1. **Concept + visualization (~25 min).** `jupyter lab lesson.ipynb` — you'll watch frames fill
   and evict on a reference string, and plot **faults vs. number of frames** to *see* FIFO's
   anomaly (the curve goes the wrong way) while LRU stays monotonic.
2. **Homework (~70 min).** Implement `translate`, `fifo`, and `lru` in `homework.py`.
3. **Grade it.** `pytest -q`  ·  or `python tools/grade.py day 10`.

## The reference string you'll reason about

```
1 2 3 4 1 2 5 1 2 3 4 5
```

- **FIFO, 3 frames → 9 faults.  FIFO, 4 frames → 10 faults.** More memory, more faults. That's
  Belady's anomaly, and it happens because FIFO's eviction order ignores *usage*.
- **LRU, 3 frames → 10 faults**, and LRU never gets worse as you add frames.

## Hints

- `translate`: `vpn, offset = divmod(vaddr, page_size)`. The offset passes through unchanged;
  only the page number is remapped.
- FIFO cares about **load order** — a `deque` of pages, evict from the left. A hit does *not*
  change the order.
- LRU cares about **use order** — an `OrderedDict` is perfect: `move_to_end` on a hit,
  `popitem(last=False)` to evict the least-recently-used.
- Snapshot a *copy* of the frames each step (`list(...)`), or every entry in
  `frames_over_time` will point at the same mutating object.

## Check yourself

- Why can't FIFO's anomaly happen to LRU? (Look up "stack algorithm": with LRU, the pages held
  by *k* frames are always a subset of those held by *k+1* frames.)
- The perfect policy (evict the page used furthest in the future) is called OPT/Belady's
  algorithm. Why can't a real OS use it?
- Real CPUs cache translations in a **TLB**. Given today's model, what would a TLB hit vs. miss
  correspond to?

## Where this shows up later

Day 11 (threads & the GIL) and Day 12 (synchronization) build on this process/memory model;
Day 15 dives into the stack/heap layout *inside* a process's address space. The LRU cache you
built today is the same structure behind countless real caches — including asset/tile caches in
robotics UIs.

**Next:** Day 11 — Threads & the GIL.
