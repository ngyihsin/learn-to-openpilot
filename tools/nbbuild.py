"""Tiny helper for authoring Jupyter notebooks from plain Python.

We keep lesson notebooks reviewable by generating them from source instead of
hand-editing JSON. Use ``md(...)`` and ``code(...)`` to build cells, then
``write_notebook(path, cells)``.

Example
-------
>>> from tools.nbbuild import md, code, write_notebook
>>> write_notebook("lesson.ipynb", [
...     md("# Hello", "", "A markdown cell."),
...     code("print('a runnable code cell')"),
... ])
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _source(lines: tuple[str, ...]) -> list[str]:
    """Join lines into nbformat's list-of-strings-with-trailing-newlines shape."""
    text = "\n".join(lines)
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def md(*lines: str) -> dict[str, Any]:
    """A markdown cell."""
    return {"cell_type": "markdown", "metadata": {}, "source": _source(lines)}


def code(*lines: str) -> dict[str, Any]:
    """A code cell (outputs are left empty; run the notebook to populate them)."""
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _source(lines),
    }


def notebook(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(path: str | Path, cells: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook(cells), indent=1) + "\n", encoding="utf-8")
    return path
