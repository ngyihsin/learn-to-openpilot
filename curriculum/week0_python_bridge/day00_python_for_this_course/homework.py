"""Day 00 homework — the Python this course is written in.

Fill in every ``TODO``. Run ``pytest -q`` in this folder until it's green.

One small class + one function that together use everything Day 01 assumes:
classes/self, dunder methods, @property, and raising exceptions.
(The ``: int`` / ``-> None`` annotations are just documentation — ignore them freely.)
"""
from __future__ import annotations


class Backpack:
    """A container with a fixed capacity."""

    def __init__(self, capacity: int) -> None:
        # TODO: store capacity on self, and start self.items as an empty list
        raise NotImplementedError

    def add(self, item) -> None:
        """Add an item. If the backpack already holds ``capacity`` items,
        raise ValueError("backpack is full")."""
        # TODO
        raise NotImplementedError

    def __len__(self) -> int:
        """How many items are inside (this is what makes ``len(b)`` work)."""
        # TODO
        raise NotImplementedError

    def __getitem__(self, i: int):
        """Return item ``i`` (this is what makes ``b[0]`` work).
        Raise IndexError for a bad index — check 0 <= i < the item count yourself."""
        # TODO
        raise NotImplementedError

    @property
    def is_full(self) -> bool:
        """True when the backpack holds ``capacity`` items. Note: read WITHOUT parentheses."""
        # TODO
        raise NotImplementedError


def safe_divide(a: float, b: float) -> float:
    """Return a / b, but raise ValueError("cannot divide by zero") when b == 0."""
    # TODO
    raise NotImplementedError
