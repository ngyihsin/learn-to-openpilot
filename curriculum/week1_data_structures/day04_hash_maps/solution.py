"""Day 04 reference solution — open-addressing hash map with linear probing + tombstones."""
from __future__ import annotations

from typing import Any, Iterator

_EMPTY = object()
_DELETED = object()


class HashMap:
    def __init__(self, initial_capacity: int = 8) -> None:
        self._cap = initial_capacity
        self._keys: list[Any] = [_EMPTY] * self._cap
        self._values: list[Any] = [None] * self._cap
        self._n = 0
        self._used = 0

    def __len__(self) -> int:
        return self._n

    @property
    def capacity(self) -> int:
        return self._cap

    @property
    def load_factor(self) -> float:
        return self._n / self._cap

    def items(self) -> Iterator[tuple[Any, Any]]:
        for k, v in zip(self._keys, self._values):
            if k is not _EMPTY and k is not _DELETED:
                yield k, v

    def __contains__(self, key: Any) -> bool:
        try:
            self.get(key)
            return True
        except KeyError:
            return False

    def get(self, key: Any) -> Any:
        idx = hash(key) % self._cap
        for _ in range(self._cap):
            k = self._keys[idx]
            if k is _EMPTY:
                raise KeyError(key)
            if k is not _DELETED and k == key:
                return self._values[idx]
            idx = (idx + 1) % self._cap
        raise KeyError(key)

    def _find_slot(self, key: Any) -> int:
        first_deleted = -1
        idx = hash(key) % self._cap
        for _ in range(self._cap):
            k = self._keys[idx]
            if k is _EMPTY:
                return first_deleted if first_deleted != -1 else idx
            if k is _DELETED:
                if first_deleted == -1:
                    first_deleted = idx
            elif k == key:
                return idx
            idx = (idx + 1) % self._cap
        return first_deleted  # table saturated with tombstones (resize will fix)

    def put(self, key: Any, value: Any) -> None:
        if (self._used + 1) > 0.7 * self._cap:
            self._resize(self._cap * 2)
        idx = self._find_slot(key)
        slot = self._keys[idx]
        if slot is _EMPTY:
            self._used += 1
            self._n += 1
        elif slot is _DELETED:
            self._n += 1
        self._keys[idx] = key
        self._values[idx] = value

    def remove(self, key: Any) -> None:
        idx = hash(key) % self._cap
        for _ in range(self._cap):
            k = self._keys[idx]
            if k is _EMPTY:
                raise KeyError(key)
            if k is not _DELETED and k == key:
                self._keys[idx] = _DELETED
                self._values[idx] = None
                self._n -= 1
                return
            idx = (idx + 1) % self._cap
        raise KeyError(key)

    def _resize(self, new_cap: int) -> None:
        old_keys, old_values = self._keys, self._values
        self._cap = new_cap
        self._keys = [_EMPTY] * new_cap
        self._values = [None] * new_cap
        self._n = 0
        self._used = 0
        for k, v in zip(old_keys, old_values):
            if k is not _EMPTY and k is not _DELETED:
                self.put(k, v)
