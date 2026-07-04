# Day 19 — Sockets & Networking

> **Week 3 · Systems Programming** — a pipe between programs, across the network.

## Why today matters

A socket is how two programs talk over a network (or the same machine): one **listens**, another
**connects**, and then they exchange bytes. TCP makes it a reliable, ordered **byte stream** —
which means, exactly like the pipe on Day 14, you don't get message boundaries for free. Build a
framed echo server/client and you understand the shape of *every* network service: HTTP, Redis,
databases, and the loopback links robotics stacks use between processes.

## Learning goals

By the end you can:

- Explain the listen/accept/connect model and what a socket file descriptor is.
- Read *exactly* N bytes off a stream (looping over `recv`) and length-frame messages.
- Run a server in one thread and a client in another over `127.0.0.1`.

## Do this

1. **Concept + visualization (~20 min).** `jupyter lab lesson.ipynb` — see the client/server
   handshake and a live echo exchange over a loopback socket.
2. **Homework (~60 min).** Implement `recv_exactly`, `send_message`, `recv_message`, `serve_echo`,
   and `echo_client`. `make_listen_socket` is provided.
3. **Grade it.** `pytest -q`  ·  or `python tools/grade.py day 19`.

## Hints

- `recv(n)` returns **up to** n bytes, and `b""` when the peer closes. Loop in `recv_exactly`
  until you've collected all n (or the peer closed → return None).
- Use `sendall` to send a whole frame (it loops internally). Framing is the same 4-byte
  length prefix from Day 14 — sockets and pipes are both just byte streams.
- The grader binds with `port=0` so the OS picks a free port; read it back with
  `getsockname()[1]`. That avoids "address already in use" flakiness.

## Check yourself

- Why must you loop over `recv` instead of assuming one call returns the whole message?
- What's the difference between TCP (stream, reliable, ordered) and UDP (datagrams, best-effort)?
  When would a robot prefer UDP?
- The server handles one client here. What would you change to handle many at once? (Threads, a
  select/epoll loop, or async.)

## Where this shows up later

Day 20 profiles I/O-bound code (sockets are a classic case where threads *do* help — Day 11).
This client/server framing is the backbone of every networked system you'll touch.

**Next:** Day 20 — Profiling & Performance.
