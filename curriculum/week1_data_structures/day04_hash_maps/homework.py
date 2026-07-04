"""Day 04 homework — a hash map with open addressing.

A hash map gives you (amortized) O(1) lookup by turning a key into an array index via a hash
function. Collisions are inevitable, so you need a strategy. Today you'll use **open
addressing with linear probing**: on a collision, walk to the next slot until you find room.

Two subtleties make or break it:
  1. **Deletion** can't just blank a slot — that would break probe chains for other keys. You
     leave a **tombstone** instead.
  2. As the table fills, probe chains get long, so you **resize** (rehash everything into a
     bigger table) once the load factor crosses ~0.7.

You're given ``get`` as a worked example of probing. Implement the rest. Run ``pytest -q``.
"""
from __future__ import annotations

from typing import Any, Iterator

_EMPTY = object()      # a slot that has never held a key
_DELETED = object()    # a tombstone: held a key that was removed


class HashMap:
    def __init__(self, initial_capacity: int = 8) -> None:
        self._cap = initial_capacity
        self._keys: list[Any] = [_EMPTY] * self._cap
        self._values: list[Any] = [None] * self._cap
        self._n = 0        # active entries
        self._used = 0     # active entries + tombstones (drives resizing)

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
        """Worked example: probe from hash(key) until we find the key or an empty slot."""
        idx = hash(key) % self._cap
        for _ in range(self._cap):
            k = self._keys[idx]
            if k is _EMPTY:
                raise KeyError(key)                 # hit an empty slot -> not present
            if k is not _DELETED and k == key:
                return self._values[idx]
            idx = (idx + 1) % self._cap             # tombstone or other key -> keep probing
        raise KeyError(key)

    # ----- implement below -----

    def _find_slot(self, key: Any) -> int:
        """Return the index to use for ``key``: the slot where it already lives, or the first
        insertable slot (a tombstone if one was seen on the way, else the first empty slot)."""
        # TODO:
        #   - probe from hash(key) % cap
        #   - remember the FIRST tombstone index you pass
        #   - if you find the key, return its index (so put overwrites)
        #   - if you hit an empty slot, return the remembered tombstone if any, else this slot
        raise NotImplementedError

    def put(self, key: Any, value: Any) -> None:
        """Insert or update. Resize first if this would push the table past ~0.7 full."""
        # TODO:
        #   - if (self._used + 1) > 0.7 * self._cap: self._resize(self._cap * 2)
        #   - idx = self._find_slot(key)
        #   - if slot is _EMPTY: it's a brand-new slot -> self._used += 1
        #   - if slot is _EMPTY or _DELETED: it's a new entry -> self._n += 1
        #   - store key & value
        raise NotImplementedError

    def remove(self, key: Any) -> None:
        """Delete a key by leaving a tombstone. Raise KeyError if absent."""
        # TODO: probe like get; on match set self._keys[idx] = _DELETED, clear the value,
        #       decrement self._n. On empty slot, raise KeyError.
        raise NotImplementedError

    def _resize(self, new_cap: int) -> None:
        """Allocate a bigger table and re-insert every active entry (dropping tombstones)."""
        # TODO: stash old keys/values, reset arrays to new_cap with counts at 0,
        #       then self.put(k, v) for every active (non-empty, non-tombstone) entry
        raise NotImplementedError
