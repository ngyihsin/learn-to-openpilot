# Day 18 — Syscalls & the Kernel Boundary

> **Week 3 · Systems Programming** — the doorway between your program and the hardware.

## Why today matters

Your program can't touch the disk, network, or another process directly — the **kernel** owns
all of that. To ask for it, you make a **system call**: `open`, `read`, `write`, `lseek`,
`close`, and hundreds more. Every high-level convenience (`open("f").read()`, `print`, sockets)
bottoms out in syscalls. Understanding this boundary is how you reason about performance ("this
is slow because it makes a million tiny `write` syscalls"), permissions, and what tools like
`strace` show you. Today you bypass Python's file objects and call the syscall layer directly.

## Learning goals

By the end you can:

- Explain the user-space/kernel boundary and what a syscall is.
- Use raw `os.open/read/write/lseek/close` with file descriptors and open flags.
- Handle two real behaviors: partial `write`s (loop until done) and `read` returning `b""` at EOF.

## Do this

1. **Concept + visualization (~20 min).** `jupyter lab lesson.ipynb` — see that Python's `open`
   ends in the same syscalls, and (on Linux) how `strace` reveals every call a program makes.
2. **Homework (~50 min).** Implement `low_level_write`, `low_level_read`, `low_level_copy`, and
   `file_size` using only `os.*`.
3. **Grade it.** `pytest -q`  ·  or `python tools/grade.py day 18`.

## Hints

- A **file descriptor** is just a small integer the kernel hands back from `os.open`; you pass it
  to every subsequent call and must `os.close` it (use `try/finally`).
- `os.write` can write *fewer* bytes than you asked — loop with `data[written:]` until it's all out.
- `os.read(fd, n)` returns up to `n` bytes and `b""` at end of file. `os.lseek(fd, 0, SEEK_END)`
  gives you the size without reading a single byte.

## Check yourself

- Why is making one `write` of 1 MB faster than 1,000,000 writes of 1 byte? (Each syscall crosses
  the boundary — that transition has a cost.)
- What does buffering (Python's `open`, C's `stdio`) do to reduce the number of syscalls?
- A file descriptor is an `int`. What does that suggest about how the kernel tracks your open files?

## Where this shows up later

Day 19's sockets are syscalls (`socket`, `connect`, `send`, `recv`) over a network fd. Day 20
profiles syscall-heavy code. openpilot's logging/messaging is careful about syscall counts for
exactly the throughput reasons above.

**Next:** Day 19 — Sockets & Networking.
