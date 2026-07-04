# Day 14 — IPC, Pipes & the File System

> **Week 2 · Operating Systems** — how separate processes actually talk. Week 2 finale.

## Why today matters

Processes have separate memory, so they can't just share a variable — they pass **bytes** through
a channel: a pipe, a socket, or a file. But those channels are raw *byte streams* with no concept
of "a message." One `read()` might return half of what you sent, or two messages glued together.
The universal fix is **framing**: prefix each message with its length so the receiver knows
exactly where it ends. Get this one idea and you understand pipes, sockets, and every wire
protocol — including the message bus that carries sensor data between openpilot's processes.

## Learning goals

By the end you can:

- Explain why a byte stream needs message framing, and implement length-prefixed frames.
- Write a stream parser that reassembles messages split across reads (holding partial-frame leftovers).
- Build a request/response protocol (a tiny KV store) and run it over a real OS pipe.

## Do this

1. **Concept + visualization (~25 min).** `jupyter lab lesson.ipynb` — send framed messages
   through an `os.pipe()` between two threads, reading in tiny chunks so you *see* the parser
   stitch split frames back together.
2. **Homework (~60 min).** Implement `frame`, `parse_frames`, `KVStore.execute`, and `serve`.
3. **Grade it.** `pytest -q`  ·  or `python tools/grade.py day 14`.

## Hints

- A frame is `struct.pack(">I", len(payload)) + payload` — 4 bytes of big-endian length, then the
  bytes. `parse_frames` reads the length, and if the full payload hasn't arrived, stops and
  returns the unparsed tail as *leftover* to prepend to the next chunk.
- `KVStore.execute`: `command.split(" ", 2)` keeps spaces inside a SET value (`SET k hello world`).
- The grader reads the pipe in 4-byte chunks on purpose — if your parser only works when a whole
  message arrives at once, that test will catch it.

## Check yourself

- Why can't the receiver just call `read()` once and assume it got a whole message?
- Length-prefixing is one framing strategy; delimiter-based (e.g. newline-terminated) is another.
  What breaks with delimiters if the payload can contain the delimiter?
- Pipes, sockets, and files are all "a stream of bytes with a file descriptor." What does that
  uniformity buy you (think `cat`, redirection, `|`)?

## Where this shows up later

This closes Week 2. In Week 3 you'll go lower: sockets (Day 19) are pipes across machines, and
syscalls (Day 18) are how your process asks the kernel to move these bytes. openpilot's `msgq`/
`cereal` layer is exactly this — framed messages between processes — at production scale.

**Next:** Week 3 — Systems Programming (Day 15, the memory model).
