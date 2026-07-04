"""Day 14 reference solution — length-prefixed framing + KV store over a pipe."""
from __future__ import annotations

import struct


def frame(payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + payload


def parse_frames(buffer: bytes) -> tuple[list[bytes], bytes]:
    messages: list[bytes] = []
    i = 0
    n = len(buffer)
    while n - i >= 4:
        (length,) = struct.unpack(">I", buffer[i:i + 4])
        if n - i - 4 < length:
            break                       # payload not fully arrived yet
        start = i + 4
        messages.append(buffer[start:start + length])
        i = start + length
    return messages, buffer[i:]


class KVStore:
    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def execute(self, command: str) -> str:
        parts = command.split(" ", 2)
        verb = parts[0].upper()
        if verb == "SET" and len(parts) == 3:
            self._data[parts[1]] = parts[2]
            return "OK"
        if verb == "GET" and len(parts) == 2:
            return self._data.get(parts[1], "NULL")
        if verb == "DEL" and len(parts) == 2:
            return "DELETED" if self._data.pop(parts[1], None) is not None else "NOTFOUND"
        if verb == "KEYS" and len(parts) == 1:
            return ",".join(sorted(self._data))
        return "ERR"


def serve(request_bytes: bytes, store: KVStore) -> bytes:
    messages, _leftover = parse_frames(request_bytes)
    out = b""
    for msg in messages:
        out += frame(store.execute(msg.decode()).encode())
    return out
