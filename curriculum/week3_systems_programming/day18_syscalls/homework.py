"""Day 18 homework — syscalls & the kernel boundary.

When you write `open("f").read()`, Python is really asking the **kernel** to do the work via
**system calls**: `open`, `read`, `write`, `lseek`, `close`. A syscall is the doorway between
your program (user space) and the kernel (which owns the hardware). Today you skip Python's file
objects and call that layer directly through `os.*`, so the boundary stops being abstract.

Two real-world gotchas you'll handle: `os.write` may write *fewer* bytes than you gave it (you
loop until done), and `os.read` returns `b""` at end-of-file.

Fill in every ``TODO`` and run ``pytest -q``.
"""
from __future__ import annotations

import os


def low_level_write(path: str, data: bytes) -> int:
    """Write `data` to `path` using raw syscalls (truncating/creating the file).
    Return the number of bytes written."""
    # TODO:
    #   fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    #   loop os.write(fd, data[written:]) until all bytes are written (it can write partially!)
    #   always os.close(fd); return the total written
    raise NotImplementedError


def low_level_read(path: str) -> bytes:
    """Read the entire file at `path` using raw syscalls, returning its bytes."""
    # TODO:
    #   fd = os.open(path, os.O_RDONLY)
    #   loop os.read(fd, 4096) collecting chunks until it returns b"" (EOF); close; join
    raise NotImplementedError


def low_level_copy(src: str, dst: str) -> int:
    """Copy `src` to `dst` with raw read/write syscalls in a loop. Return bytes copied."""
    # TODO: open src for reading and dst for writing; stream chunks across; close both fds
    raise NotImplementedError


def file_size(path: str) -> int:
    """Return the file's size WITHOUT reading it, by seeking to the end (os.lseek + SEEK_END)."""
    # TODO: os.open read-only, os.lseek(fd, 0, os.SEEK_END) gives the size; close; return it
    raise NotImplementedError
