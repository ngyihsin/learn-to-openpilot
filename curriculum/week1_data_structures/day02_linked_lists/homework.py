"""Day 02 homework — a doubly-linked list.

Yesterday's dynamic array stores items in one contiguous block, so inserting in the middle
means shifting everything after it (O(n)). A **linked list** instead threads items together
with pointers: each node knows its neighbours. That makes inserting or deleting at a known
node O(1) — no shifting — at the cost of losing O(1) random access.

You'll build a *doubly*-linked list (each node has both `prev` and `next`), which is what
lets you walk the list backwards and delete a node in O(1).

Fill in every ``TODO`` and run ``pytest -q``. The traversal helpers are written for you;
your job is the pointer surgery.
"""
from __future__ import annotations

from typing import Any, Iterator


class _Node:
    __slots__ = ("value", "prev", "next")

    def __init__(self, value: Any) -> None:
        self.value = value
        self.prev: _Node | None = None
        self.next: _Node | None = None


class DoublyLinkedList:
    def __init__(self) -> None:
        self._head: _Node | None = None
        self._tail: _Node | None = None
        self._n = 0

    def __len__(self) -> int:
        return self._n

    def __iter__(self) -> Iterator[Any]:
        cur = self._head
        while cur is not None:
            yield cur.value
            cur = cur.next

    def to_list(self) -> list[Any]:
        """Values from head to tail (follows `next` pointers)."""
        return list(self)

    def to_list_reverse(self) -> list[Any]:
        """Values from tail to head (follows `prev` pointers).

        This is how the grader checks that you kept the `prev` links correct — a subtly
        broken insert/delete often still reads fine forward but breaks going backward.
        """
        out: list[Any] = []
        cur = self._tail
        while cur is not None:
            out.append(cur.value)
            cur = cur.prev
        return out

    # ----- implement the methods below -----

    def push_back(self, value: Any) -> None:
        """Append to the tail in O(1)."""
        # TODO: make a _Node; link it after self._tail; fix head/tail; bump self._n
        raise NotImplementedError

    def push_front(self, value: Any) -> None:
        """Prepend to the head in O(1)."""
        # TODO: mirror push_back on the head side
        raise NotImplementedError

    def pop_front(self) -> Any:
        """Remove and return the head value. Raise IndexError if empty."""
        # TODO: unlink self._head; fix the new head's prev (or clear tail if now empty)
        raise NotImplementedError

    def pop_back(self) -> Any:
        """Remove and return the tail value. Raise IndexError if empty."""
        # TODO: mirror pop_front on the tail side
        raise NotImplementedError

    def delete_value(self, value: Any) -> bool:
        """Delete the first node whose value == `value`. Return True if one was removed.

        The whole point of a doubly-linked list: once you've found the node, unlinking it
        is O(1) — just rewire its neighbours' pointers around it.
        """
        # TODO: walk from head; on match, rewire node.prev.next and node.next.prev
        #       (careful at the ends, where prev or next is None -> update head/tail)
        raise NotImplementedError

    def reverse(self) -> None:
        """Reverse the list in place, in O(n), by swapping each node's prev/next."""
        # TODO: swap prev/next on every node, then swap head and tail
        raise NotImplementedError
