# Day 00 — The Python This Course Is Written In

> **Week 0 · Python Bridge** — every homework file ahead uses five Python features that basic
> tutorials skip. Learn them here in one sitting, and Day 01 stops being a wall.
>
> *Assumes only basic Python: variables, functions, lists, for-loops, if/else.*

## Why today matters

Open any homework file in this course and you'll see things like:

```python
class DynamicArray:
    def __init__(self) -> None:
        self._n: int = 0

    @property
    def capacity(self) -> int: ...

    def __len__(self) -> int: ...

    def pop(self):
        raise IndexError("empty")
```

Five unfamiliar things at once: `class`/`self`, `__len__`-style names, `@property`, `raise`, and
`: int` annotations. None of them is hard — but meeting all five unexplained on Day 01 is a wall.
Today tears it down.

## 1 · Classes and `self` — a blueprint plus its data

A **class** bundles data with the functions that work on that data. Think of a *blueprint*: the class
describes what every "backpack" has; each actual backpack you create is an **object**.

```python
class Backpack:
    def __init__(self, capacity):     # runs when you create one: Backpack(3)
        self.capacity = capacity      # store data ON this object
        self.items = []

    def add(self, item):              # a method: a function that lives in the class
        self.items.append(item)

b = Backpack(3)      # __init__ runs; b.capacity is 3, b.items is []
b.add("rope")        # note: you don't pass self — Python fills it in
```

**`self` is simply "this particular object."** When you call `b.add("rope")`, Python translates it to
`Backpack.add(b, "rope")` — `self` *is* `b`. That's the whole trick. A leading underscore
(`self._n`) is just a naming convention meaning "internal — please don't touch from outside."

## 2 · Dunder methods — teaching your class to speak Python

Names with **d**ouble **under**scores (`__len__`, `__getitem__`) are hooks Python calls for you:

| You write | Python calls | Meaning |
|-----------|--------------|---------|
| `len(b)` | `b.__len__()` | how many things inside |
| `b[0]` | `b.__getitem__(0)` | read by index |
| `b[0] = x` | `b.__setitem__(0, x)` | write by index |
| `print(b)` | `b.__repr__()` | how it displays |

Implement `__len__` and suddenly `len()` works on *your* class exactly like on a list. The course
uses this constantly: you'll *build* list-like and dict-like things.

## 3 · Exceptions — how code says "I can't do that"

`raise` throws an error object up to whoever called you; `try/except` catches one.

```python
def pop(self):
    if len(self.items) == 0:
        raise IndexError("pop from empty backpack")   # stop, with a clear reason
    return self.items.pop()
```

Two idioms you'll meet immediately:
- Every unsolved TODO in this course is `raise NotImplementedError` — it means **"your code goes
  here; until then, calling this fails on purpose."** That's why fresh homework fails its tests.
- Graders often check errors: *"popping an empty stack must raise `IndexError`"* is a real test.

## 4 · `@property` — a method that looks like a variable

```python
class Backpack:
    @property
    def is_full(self):
        return len(self.items) >= self.capacity

b.is_full        # no parentheses! reads like a variable, computes like a method
```

That `@...` line is a **decorator** — it wraps the function below it. You only need to *recognize*
two in this course: `@property` (above) and `@dataclass` (Week 2 — it auto-writes `__init__` for a
class that's mostly data).

## 5 · Type hints — documentation you can ignore

```python
def append(self, value: Any) -> None:
    self._data: list[Any] = []
```

The `: Any`, `-> None`, `: list[Any]` parts are **hints about what type goes where — for human
readers and tools. Python does not enforce them, and you can ignore them completely.** Read
`def f(x: int) -> str` as "f takes an int-ish thing and returns a string-ish thing," and move on.
(`from __future__ import annotations` at the top of files just makes these hints more flexible —
also ignorable.)

## 6 · Big-O in ten minutes — counting steps, not seconds

**O(...)** answers: *if the input gets 10× bigger, how much more work is it?*

| Notation | Name | Example |
|----------|------|---------|
| O(1) | constant | `x[5]` — same cost at any list size |
| O(n) | linear | a loop over the list |
| O(n²) | quadratic | a loop inside a loop |

**"Amortized O(1)"** (Day 01's headline) means: *usually* constant, with a rare expensive operation
whose cost, spread over many calls, averages out to constant. Like a bus pass: one big payment, tiny
cost per ride.

## Do this — build one small class

`homework.py` has one class and one function that exercise everything above. After each step, run
its check here:

**Step 1 · `Backpack.__init__` + `add`** — store `capacity`, start `items` empty; `add` appends but
raises `ValueError` when full.
```bash
python3 -c "from homework import Backpack; b=Backpack(2); b.add('rope'); print(len(b.items))"   # expect: 1
```

**Step 2 · `__len__` and `__getitem__`** — `len(b)` = item count; `b[i]` returns item `i`, raising
`IndexError` for a bad index.
```bash
python3 -c "from homework import Backpack; b=Backpack(2); b.add('rope'); print(len(b), b[0])"   # expect: 1 rope
```

**Step 3 · `@property is_full`** — True when at capacity (no parentheses when reading it).
```bash
python3 -c "from homework import Backpack; b=Backpack(1); b.add('x'); print(b.is_full)"          # expect: True
```

**Step 4 · `safe_divide(a, b)`** — a plain function: return `a / b`, but raise `ValueError` (not
ZeroDivisionError) with a clear message when `b == 0`.
```bash
python3 -c "from homework import safe_divide; print(safe_divide(6, 3))"                          # expect: 2.0
```

**Grade the whole day:** `pytest -q`  ·  or from the repo root `python tools/grade.py day 0`.

## Check yourself

- In `b.add("rope")`, what is `self` inside `add`? What did Python pass without you writing it?
- What does implementing `__len__` buy you that a normal method called `length()` doesn't?
- What does `raise NotImplementedError` in fresh homework mean — is it a bug?
- A list append is "amortized O(1)". In the bus-pass analogy, what's the big payment? (You'll build
  exactly this on Day 01.)

## Where this shows up later

Everywhere. Day 01 is a class with dunders and `@property`. Day 04 tests your exceptions. Week 2
adds `@dataclass`. Week 4's `nn.Module` is "write a subclass with a method." You now read all of it.

**Next:** Day 01 — Dynamic arrays.
