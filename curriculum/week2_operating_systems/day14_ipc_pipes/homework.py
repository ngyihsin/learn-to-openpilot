"""Day 14 homework — IPC: message framing & a key-value store over a pipe.

Processes don't share memory (Day 11), so they talk by passing **bytes** through channels: pipes,
sockets, files. But a pipe is a raw *byte stream* with no message boundaries — read() might hand
you half a message, or two at once. The fix is **framing**: prefix each message with its length
so the receiver knows exactly where it ends. This one idea underlies pipes, sockets, and every
network protocol (and openpilot's inter-process messaging).

You'll build length-prefixed framing, a stream parser that reassembles messages, and a tiny
key-value store that speaks a request/response protocol over it.

Fill in every ``TODO`` and run ``pytest -q``.
"""
from __future__ import annotations

import struct


def frame(payload: bytes) -> bytes:
    """Wrap `payload` as a frame: a 4-byte big-endian length prefix, then the payload bytes."""
    # TODO: return struct.pack(">I", len(payload)) + payload
    raise NotImplementedError


def parse_frames(buffer: bytes) -> tuple[list[bytes], bytes]:
    """Pull every COMPLETE frame out of `buffer`. Return (messages, leftover), where leftover is
    the trailing bytes of an incomplete frame (fewer than a full length+payload).

    This is the heart of stream reassembly: you might be handed a partial frame and have to hold
    the leftover until more bytes arrive.
    """
    # TODO:
    #   walk an index i; while at least 4 bytes remain, read the length; if the full payload
    #   isn't present yet, stop and return what's parsed + buffer[i:] as leftover.
    raise NotImplementedError


class KVStore:
    """A tiny in-memory store driven by text commands. Supported commands and their responses:
        SET <key> <value>   -> "OK"
        GET <key>           -> the value, or "NULL" if absent
        DEL <key>           -> "DELETED" if it existed, else "NOTFOUND"
        KEYS                -> comma-separated sorted keys (empty string if none)
    Anything else          -> "ERR"
    """

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def execute(self, command: str) -> str:
        # TODO: split into at most 3 parts ("SET k v v v" keeps spaces in the value), match the
        #       verb (case-insensitive), mutate/read self._data, and return the response string.
        raise NotImplementedError


def serve(request_bytes: bytes, store: KVStore) -> bytes:
    """Parse framed command messages out of `request_bytes`, execute each against `store`, and
    return the framed responses concatenated in order."""
    # TODO: messages, _leftover = parse_frames(request_bytes); for each, frame(store.execute(...))
    raise NotImplementedError
