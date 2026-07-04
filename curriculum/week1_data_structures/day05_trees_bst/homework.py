"""Day 05 homework — a Binary Search Tree.

A BST keeps values ordered: everything in a node's left subtree is smaller, everything in its
right subtree is larger. That invariant makes `contains`, `insert`, and `delete` run in O(height)
— O(log n) when the tree is balanced — and makes an in-order traversal spit values out *sorted*.

The famous hard part is **delete**, which has three cases (leaf, one child, two children).
You get the traversals and min/max helpers; implement the operations. Run ``pytest -q``.
"""
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

    # ----- provided helpers -----

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
        """-1 for an empty tree, 0 for a single node."""
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

    # ----- implement below -----

    def insert(self, value: Any) -> None:
        """Insert value, keeping the BST invariant. Duplicates are ignored (no-op)."""
        # TODO: walk/recurse left for smaller, right for larger; create a node at the empty
        #       spot. Only increment self._n when you actually add a node.
        raise NotImplementedError

    def contains(self, value: Any) -> bool:
        # TODO: walk down comparing value to node.value; O(height), never scan everything
        raise NotImplementedError

    def in_order(self) -> list[Any]:
        """Left, node, right — for a BST this yields the values in **sorted** order."""
        # TODO: recurse left, append node.value, recurse right
        raise NotImplementedError

    def delete(self, value: Any) -> bool:
        """Remove value if present; return True if something was removed.

        Three cases when you find the node:
          - no children  -> just remove it
          - one child    -> replace it with that child
          - two children -> copy in its in-order successor (min of the right subtree),
                            then delete that successor from the right subtree
        """
        # TODO: implement the three cases (recursion is cleanest). Decrement self._n on removal.
        raise NotImplementedError
