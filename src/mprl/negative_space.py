"""Runtime checks that survive ``python -O``.

Copy this module into your project. Standard library only, Python 3.10+.

Why not ``assert``: ``python -O`` / ``PYTHONOPTIMIZE=1`` removes every ``assert``
statement from the bytecode, condition and message included. Checks that encode
your contracts must not disappear when someone flips an interpreter flag.

Why ``expect`` exists: a type checker treats ``require(x is not None)`` as an
ordinary call returning ``None``, so it does not narrow ``x``. Pyright/Pylance
then report ``reportOptionalSubscript`` / ``reportOptionalMemberAccess`` on the
next line. ``expect(x, msg)`` performs the same check *and* returns ``x`` typed
without ``None``, so one call satisfies both the runtime and the checker.

Use for programmer errors (a state your own code should have made impossible).
For operating errors -- bad user input, missing file, network failure -- raise a
typed exception instead and handle it.
"""

from __future__ import annotations

import importlib
import math
from typing import TYPE_CHECKING, NoReturn, SupportsFloat, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

__all__ = [
    "CheckFailed",
    "bounded",
    "check_finite",
    "check_shape",
    "expect",
    "require",
    "unreachable",
]

T = TypeVar("T")

# A shape spec entry: fixed size, named dimension, or None for "any size".
Dim = int | str | None


class CheckFailed(AssertionError):  # noqa: N818  public name; reads as an event
    """A contract was violated. Subclasses AssertionError so existing handlers
    and ``pytest.raises(AssertionError)`` keep working."""


def require(condition: object, message: str = "") -> None:
    """Fail unless ``condition`` is truthy.

    >>> require(1 < 2)
    >>> require(2 < 1, "ordering broken")
    Traceback (most recent call last):
        ...
    negative_space.CheckFailed: ordering broken

    Keep one predicate per call: ``require(a); require(b)`` reports which half
    failed, ``require(a and b)`` does not.

    This does NOT narrow types. After ``require(x is not None)`` a type checker
    still thinks ``x`` may be ``None``; use ``x = expect(x, msg)`` instead.
    """
    if not condition:
        raise CheckFailed(message or "requirement failed")


def expect(value: T | None, message: str = "") -> T:  # nsp: ignore NSP001 if-raise is the contract
    """Return ``value`` if it is not ``None``; fail otherwise.

    The narrowing counterpart of ``require``: the return type has ``None``
    removed, so the checker stops reporting optional access downstream.

    >>> expect({"id": "a1"}.get("id"), "item has no id")
    'a1'
    >>> expect({"id": "a1"}.get("sha256"), "item has no sha256")
    Traceback (most recent call last):
        ...
    negative_space.CheckFailed: item has no sha256
    """
    if value is None:
        raise CheckFailed(message or "expected a value, got None")
    return value


def unreachable(message: str = "") -> NoReturn:
    """Mark a branch that cannot execute -- an exhausted ``match``, an ``else``
    that exists only for symmetry. Reaching it means the code is wrong."""
    raise CheckFailed(message or "reached unreachable code")


def bounded(iterable: Iterable[T], limit: int, name: str = "loop") -> Iterator[T]:
    """Yield from ``iterable``, failing if it produces more than ``limit`` items.

    Wraps any loop whose length you did not compute yourself, so an unbounded
    producer becomes a crash instead of a hang.

    >>> list(bounded(range(3), 5))
    [0, 1, 2]
    >>> list(bounded(range(10), 3, name="retries"))
    Traceback (most recent call last):
        ...
    negative_space.CheckFailed: retries exceeded its bound of 3 iterations
    """
    require(limit >= 1, f"{name}: bound must be at least 1, got {limit}")
    for count, item in enumerate(iterable, start=1):
        if count > limit:
            raise CheckFailed(f"{name} exceeded its bound of {limit} iterations")
        yield item


def check_shape(
    array: object,
    spec: Sequence[Dim],
    name: str = "array",
) -> dict[str, int]:
    """Check a NumPy/PyTorch-style ``.shape`` against ``spec`` and return the
    bindings for any named dimensions.

    ``spec`` entries: an ``int`` for a fixed size, a ``str`` for a named
    dimension that must be consistent across its uses, or ``None`` (or ``-1``)
    for "any size".

    >>> class Fake:  # any object with a .shape tuple
    ...     shape = (8, 3, 32, 32)
    >>> check_shape(Fake(), ("B", 3, "H", "H"), name="images")
    {'B': 8, 'H': 32}
    >>> check_shape(Fake(), ("B", 3, "H", "W", "C"), name="images")
    Traceback (most recent call last):
        ...
    negative_space.CheckFailed: images: expected 5 dims ('B', 3, 'H', 'W', 'C'), got 4 ...
    """
    raw = expect(getattr(array, "shape", None), f"{name} has no .shape attribute")
    shape = tuple(int(d) for d in raw)
    require(
        len(shape) == len(spec),
        f"{name}: expected {len(spec)} dims {tuple(spec)}, got {len(shape)} {shape}",
    )

    bindings: dict[str, int] = {}
    for axis, (actual, expected) in enumerate(zip(shape, spec, strict=True)):
        if expected is None or expected == -1:
            continue
        if isinstance(expected, str):
            if expected in bindings:
                require(
                    bindings[expected] == actual,
                    f"{name}: dim '{expected}' is {bindings[expected]} elsewhere "
                    f"but {actual} at axis {axis}; full shape {shape}",
                )
            else:
                bindings[expected] = actual
        else:
            require(
                actual == expected,
                f"{name}: axis {axis} expected {expected}, got {actual}; "
                f"full shape {shape}",
            )
    return bindings


def check_finite(value: object, name: str = "value") -> None:
    """Fail if ``value`` contains NaN or infinity.

    Accepts, in order: a tensor with an ``.isfinite()`` method (PyTorch), an
    array-like exposing ``__array__`` (NumPy, checked with ``numpy.isfinite``,
    imported only on this path), or anything ``float()`` accepts via
    ``__float__``.

    >>> check_finite(0.5)
    >>> check_finite(float("nan"), name="loss")
    Traceback (most recent call last):
        ...
    negative_space.CheckFailed: loss is not finite: nan
    >>> check_finite("0.5", name="lr")
    Traceback (most recent call last):
        ...
    negative_space.CheckFailed: lr is not numeric: str
    """
    isfinite_method = getattr(value, "isfinite", None)
    if callable(isfinite_method):
        result = isfinite_method()
        all_method = getattr(result, "all", None)
        ok = bool(all_method()) if callable(all_method) else bool(result)
        require(ok, f"{name} is not finite: {value}")
        return
    if hasattr(value, "__array__"):
        np = importlib.import_module("numpy")
        require(bool(np.isfinite(np.asarray(value)).all()), f"{name} is not finite")
        return
    if not isinstance(value, SupportsFloat):
        raise CheckFailed(f"{name} is not numeric: {type(value).__name__}")
    require(math.isfinite(float(value)), f"{name} is not finite: {value}")


if __name__ == "__main__":
    import doctest

    # Test the importable module, not __main__, so exception lines read
    # `negative_space.CheckFailed: <message>` and the messages are compared too.
    import negative_space  # pyright: ignore[reportMissingImports]  doctest self-run only

    failures, _ = doctest.testmod(negative_space, optionflags=doctest.ELLIPSIS)
    raise SystemExit(1 if failures else 0)
