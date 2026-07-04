"""Auto-grader for Day 14 — IPC framing & KV store.

Run ``pytest -q`` here to grade ``homework.py``.
Grade the reference solution with:  ``LP_IMPL=solution pytest -q``
"""
from __future__ import annotations

import importlib.util
import os
import sys


def _load(name: str):
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    path = os.path.join(here, name + ".py")
    modname = f"lp_{os.path.basename(here)}_{name}"
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_impl = _load(os.environ.get("LP_IMPL", "homework"))
frame, parse_frames, serve = _impl.frame, _impl.parse_frames, _impl.serve
KVStore = _impl.KVStore


def test_frame_has_length_prefix():
    f = frame(b"hello")
    assert f == b"\x00\x00\x00\x05hello"
    assert len(f) == 4 + 5


def test_roundtrip_multiple_frames():
    payloads = [b"a", b"bb", b"ccc"]
    buf = b"".join(frame(p) for p in payloads)
    msgs, leftover = parse_frames(buf)
    assert msgs == payloads
    assert leftover == b""


def test_partial_frame_is_held_as_leftover():
    buf = frame(b"hello") + frame(b"world")
    # Chop the last byte: the second frame is incomplete.
    msgs, leftover = parse_frames(buf[:-1])
    assert msgs == [b"hello"]
    assert leftover == buf[len(frame(b"hello")):-1]
    # Feeding leftover + the missing byte completes it.
    msgs2, leftover2 = parse_frames(leftover + buf[-1:])
    assert msgs2 == [b"world"]
    assert leftover2 == b""


def test_kvstore_commands():
    s = KVStore()
    assert s.execute("SET name vincent") == "OK"
    assert s.execute("GET name") == "vincent"
    assert s.execute("GET missing") == "NULL"
    assert s.execute("SET greeting hello world") == "OK"     # value keeps its spaces
    assert s.execute("GET greeting") == "hello world"
    assert s.execute("DEL name") == "DELETED"
    assert s.execute("DEL name") == "NOTFOUND"
    assert s.execute("KEYS") == "greeting"
    assert s.execute("BOGUS x") == "ERR"


def test_serve_pipeline():
    store = KVStore()
    cmds = ["SET a 1", "SET b 2", "GET a", "GET z", "DEL b", "KEYS"]
    request = b"".join(frame(c.encode()) for c in cmds)
    responses, _ = parse_frames(serve(request, store))
    assert [r.decode() for r in responses] == ["OK", "OK", "1", "NULL", "DELETED", "a"]


def test_kv_over_a_real_os_pipe():
    """End-to-end through an actual OS pipe, read in tiny 4-byte chunks to prove the parser
    reassembles messages that arrive split across reads."""
    store = KVStore()
    cmds = ["SET x 10", "SET y 20", "GET x", "KEYS", "DEL x", "GET x"]
    request = b"".join(frame(c.encode()) for c in cmds)

    r, w = os.pipe()
    os.write(w, request)
    os.close(w)
    received = b""
    while True:
        chunk = os.read(r, 4)          # deliberately small: frames span multiple reads
        if not chunk:
            break
        received += chunk
    os.close(r)

    responses, leftover = parse_frames(serve(received, store))
    assert leftover == b""
    assert [r.decode() for r in responses] == ["OK", "OK", "10", "x,y", "DELETED", "NULL"]
