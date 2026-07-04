# Day 04 — Hash Maps

> **Week 1 · Data Structures** — O(1) lookup, and the collisions you have to tame to get it.

## Why today matters

`dict` is probably the data structure you use most, and it feels free. It isn't — under the
hood it's an array plus a **hash function** that turns a key into an index, plus a plan for when
two keys want the same slot. Today you build that plan yourself with **open addressing + linear
probing**, and you'll hit the two things every hash table must get right: **deletion without
breaking probe chains** (tombstones) and **resizing** before it slows to a crawl.

## Learning goals

By the end you can:

- Explain hashing → index, and why collisions are unavoidable.
- Implement linear probing for insert/lookup, tombstone deletion, and load-factor resizing.
- Reason about why lookup is *amortized* O(1) and what wrecks it (high load factor, bad hashes).

## Do this

1. **Concept + visualization (~20 min).** `jupyter lab lesson.ipynb` — watch keys land in buckets,
   collisions probe forward, and see a plot of **average probe length vs. load factor** (it
   explodes as the table fills — that's why we resize).
2. **Homework (~70 min).** Implement `_find_slot`, `put`, `remove`, `_resize`. `get` is given
   as a worked probing example.
3. **Grade it.** `pytest -q`  ·  or `python tools/grade.py day 4`.

## Hints

- **Two sentinels:** `_EMPTY` (never used) and `_DELETED` (tombstone). `get`/`remove` stop at
  `_EMPTY` but must *skip over* `_DELETED`. This is the whole trick — blanking a removed slot
  to `_EMPTY` would make later keys in the chain unreachable.
- **`_find_slot` for `put`:** scan the whole probe chain to see if the key already exists (so you
  overwrite), but remember the first tombstone you passed and insert there if the key is new.
- **Resize** by rehashing: make a bigger array and `put` every active entry again. Do it *before*
  the load factor crosses ~0.7; the grader checks `capacity` actually grew.

## Check yourself

- Why does removing a key by setting its slot back to `_EMPTY` break lookups for *other* keys?
- What happens to average probe length as load factor → 1.0? (The notebook plot answers this.)
- Chaining (a linked list per bucket) is the other collision strategy. What does open addressing
  trade away to get better cache behavior?

## Where this shows up later

Day 10's LRU cache pairs a hash map with a linked list. Symbol tables, dedup, memoization,
routing tables — hash maps are everywhere in systems and in openpilot's message registries.

**Next:** Day 05 — Trees & BSTs.
