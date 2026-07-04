"""Auto-grader for Day 18 — syscalls.

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
low_level_write = _impl.low_level_write
low_level_read = _impl.low_level_read
low_level_copy = _impl.low_level_copy
file_size = _impl.file_size


def test_write_read_roundtrip(tmp_path):
    p = str(tmp_path / "a.bin")
    data = b"hello, kernel\x00\xff binary bytes"
    assert low_level_write(p, data) == len(data)
    assert low_level_read(p) == data


def test_write_truncates_existing(tmp_path):
    p = str(tmp_path / "b.txt")
    low_level_write(p, b"a much longer original content")
    low_level_write(p, b"short")
    assert low_level_read(p) == b"short"      # O_TRUNC wiped the old contents


def test_large_payload_exercises_chunk_loops(tmp_path):
    p = str(tmp_path / "big.bin")
    data = bytes(i % 256 for i in range(200_000))
    low_level_write(p, data)
    assert low_level_read(p) == data
    assert file_size(p) == len(data)


def test_empty_file(tmp_path):
    p = str(tmp_path / "empty")
    assert low_level_write(p, b"") == 0
    assert low_level_read(p) == b""
    assert file_size(p) == 0


def test_copy(tmp_path):
    src = str(tmp_path / "src")
    dst = str(tmp_path / "dst")
    data = b"copy me across file descriptors" * 1000
    low_level_write(src, data)
    assert low_level_copy(src, dst) == len(data)
    assert low_level_read(dst) == data


def test_file_size_matches_content(tmp_path):
    p = str(tmp_path / "sz")
    low_level_write(p, b"1234567890")
    assert file_size(p) == 10
