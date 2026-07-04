"""Day 02 reference solution — a doubly-linked list.

Key ideas: every operation is O(1) except the O(n) search inside delete_value and the O(n)
reverse. The fiddly part is the boundaries — when prev or next is None you're at an end, so
you must update head/tail instead of a neighbour.
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
        return list(self)

    def to_list_reverse(self) -> list[Any]:
        out: list[Any] = []
        cur = self._tail
        while cur is not None:
            out.append(cur.value)
            cur = cur.prev
        return out

    def push_back(self, value: Any) -> None:
        node = _Node(value)
        node.prev = self._tail
        if self._tail is not None:
            self._tail.next = node
        else:
            self._head = node
        self._tail = node
        self._n += 1

    def push_front(self, value: Any) -> None:
        node = _Node(value)
        node.next = self._head
        if self._head is not None:
            self._head.prev = node
        else:
            self._tail = node
        self._head = node
        self._n += 1

    def pop_front(self) -> Any:
        if self._head is None:
            raise IndexError("pop_front from empty list")
        node = self._head
        self._head = node.next
        if self._head is not None:
            self._head.prev = None
        else:
            self._tail = None
        self._n -= 1
        return node.value

    def pop_back(self) -> Any:
        if self._tail is None:
            raise IndexError("pop_back from empty list")
        node = self._tail
        self._tail = node.prev
        if self._tail is not None:
            self._tail.next = None
        else:
            self._head = None
        self._n -= 1
        return node.value

    def _unlink(self, node: _Node) -> None:
        if node.prev is not None:
            node.prev.next = node.next
        else:
            self._head = node.next
        if node.next is not None:
            node.next.prev = node.prev
        else:
            self._tail = node.prev
        self._n -= 1

    def delete_value(self, value: Any) -> bool:
        cur = self._head
        while cur is not None:
            if cur.value == value:
                self._unlink(cur)
                return True
            cur = cur.next
        return False

    def reverse(self) -> None:
        cur = self._head
        while cur is not None:
            cur.prev, cur.next = cur.next, cur.prev
            cur = cur.prev  # prev now points to the old next
        self._head, self._tail = self._tail, self._head
