"""Day 05 reference solution — Binary Search Tree."""
from __future__ import annotations

from typing import Any


class _Node:
    __slots__ = ("value", "left", "right")

    def __init__(self, value: Any) -> None:
        self.value = value
        self.left: _Node | None = None
        self.right: _Node | None = None


class BST:
    def __init__(self) -> None:
        self._root: _Node | None = None
        self._n = 0

    def __len__(self) -> int:
        return self._n

    def _min_node(self, node: _Node) -> _Node:
        while node.left is not None:
            node = node.left
        return node

    def min(self) -> Any:
        if self._root is None:
            raise ValueError("min of empty tree")
        return self._min_node(self._root).value

    def max(self) -> Any:
        if self._root is None:
            raise ValueError("max of empty tree")
        node = self._root
        while node.right is not None:
            node = node.right
        return node.value

    def height(self) -> int:
        def h(node: _Node | None) -> int:
            return -1 if node is None else 1 + max(h(node.left), h(node.right))
        return h(self._root)

    def pre_order(self) -> list[Any]:
        out: list[Any] = []
        def walk(node: _Node | None) -> None:
            if node:
                out.append(node.value); walk(node.left); walk(node.right)
        walk(self._root)
        return out

    def insert(self, value: Any) -> None:
        def ins(node: _Node | None, value: Any) -> tuple[_Node, bool]:
            if node is None:
                return _Node(value), True
            if value < node.value:
                node.left, added = ins(node.left, value)
            elif value > node.value:
                node.right, added = ins(node.right, value)
            else:
                return node, False  # duplicate
            return node, added

        self._root, added = ins(self._root, value)
        if added:
            self._n += 1

    def contains(self, value: Any) -> bool:
        node = self._root
        while node is not None:
            if value == node.value:
                return True
            node = node.left if value < node.value else node.right
        return False

    def in_order(self) -> list[Any]:
        out: list[Any] = []
        def walk(node: _Node | None) -> None:
            if node:
                walk(node.left); out.append(node.value); walk(node.right)
        walk(self._root)
        return out

    def delete(self, value: Any) -> bool:
        removed = False

        def dele(node: _Node | None, value: Any) -> _Node | None:
            nonlocal removed
            if node is None:
                return None
            if value < node.value:
                node.left = dele(node.left, value)
            elif value > node.value:
                node.right = dele(node.right, value)
            else:
                removed = True
                if node.left is None:
                    return node.right
                if node.right is None:
                    return node.left
                succ = self._min_node(node.right)
                node.value = succ.value
                node.right = dele(node.right, succ.value)
            return node

        self._root = dele(self._root, value)
        if removed:
            self._n -= 1
        return removed
