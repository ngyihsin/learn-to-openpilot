"""Auto-grader for Day 19 — TCP sockets.

Run ``pytest -q`` here to grade ``homework.py``.
Grade the reference solution with:  ``LP_IMPL=solution pytest -q``

Uses a real loopback (127.0.0.1) server on an OS-assigned port, in a background thread.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import threading

import pytest


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
make_listen_socket = _impl.make_listen_socket
serve_echo = _impl.serve_echo
echo_client = _impl.echo_client


def _run_echo_test(messages):
    server = make_listen_socket()
    port = server.getsockname()[1]
    t = threading.Thread(target=serve_echo, args=(server,), daemon=True)
    t.start()
    try:
        replies = echo_client("127.0.0.1", port, messages)
    finally:
        t.join(timeout=5)
        server.close()
    return replies


def test_single_message():
    assert _run_echo_test([b"hello"]) == [b"hello"]


def test_multiple_messages_one_connection():
    msgs = [b"one", b"two", b"three"]
    assert _run_echo_test(msgs) == msgs


def test_large_message_spans_many_packets():
    big = bytes(i % 256 for i in range(100_000))     # forces recv to loop
    assert _run_echo_test([big]) == [big]


def test_mixed_sizes():
    msgs = [b"", b"x", b"y" * 5000, b"z"]
    assert _run_echo_test(msgs) == msgs


def test_port_is_assigned():
    server = make_listen_socket()
    try:
        assert server.getsockname()[1] > 0
    finally:
        server.close()
