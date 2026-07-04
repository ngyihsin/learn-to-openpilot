"""Day 18 reference solution — raw file syscalls via os.*."""
from __future__ import annotations

import os


def low_level_write(path: str, data: bytes) -> int:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        written = 0
        while written < len(data):
            written += os.write(fd, data[written:])   # os.write may write partially
        return written
    finally:
        os.close(fd)


def low_level_read(path: str) -> bytes:
    fd = os.open(path, os.O_RDONLY)
    try:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 4096)
            if not chunk:                              # b"" -> EOF
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def low_level_copy(src: str, dst: str) -> int:
    rfd = os.open(src, os.O_RDONLY)
    wfd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        total = 0
        while True:
            chunk = os.read(rfd, 65536)
            if not chunk:
                break
            n = 0
            while n < len(chunk):
                n += os.write(wfd, chunk[n:])
            total += len(chunk)
        return total
    finally:
        os.close(rfd)
        os.close(wfd)


def file_size(path: str) -> int:
    fd = os.open(path, os.O_RDONLY)
    try:
        return os.lseek(fd, 0, os.SEEK_END)
    finally:
        os.close(fd)
