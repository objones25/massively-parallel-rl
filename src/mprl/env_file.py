"""``.env.example`` parsing, prompting for values, and writing ``.env`` (mode 0600).

``.env.example`` format: a ``# comment`` block directly above each ``NAME=default`` line (an empty
default means none). A comment block followed by a blank line belongs to no key. Values are written
unquoted, so a value python-dotenv would alter (whitespace, quotes, ``#``, ``$``, backslash) is refused.
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from mprl.negative_space import bounded, expect, require

_SECRET_SUFFIXES: Final = ("_KEY", "_TOKEN")
_ASSIGNMENT: Final = re.compile(r"([A-Z][A-Z0-9_]*)=(.*)")
_SAFE_VALUE: Final = re.compile(r"[^\s'\"#$\\`]*")
_MAX_LINES: Final = 1_000
_MAX_ATTEMPTS: Final = 3


@dataclass(frozen=True)
class EnvVar:
    name: str
    default: str | None
    comment: str
    secret: bool


def parse_env_example(text: str) -> list[EnvVar]:
    lines = text.splitlines()
    parsed: list[EnvVar] = []
    comment: list[str] = []
    for line in bounded(lines, _MAX_LINES, ".env.example lines"):
        stripped = line.strip()
        if not stripped:
            comment = []
        elif stripped.startswith("#"):
            comment.append(stripped.removeprefix("#").strip())
        else:
            match = expect(_ASSIGNMENT.fullmatch(stripped), f".env.example: not NAME=value: {stripped!r}")
            name, default = match.group(1), match.group(2)
            require(bool(comment), f".env.example: {name} has no comment saying where to get it")
            parsed.append(EnvVar(name, default or None, "\n".join(comment), name.endswith(_SECRET_SUFFIXES)))
            comment = []
    require(len({v.name for v in parsed}) == len(parsed), ".env.example: duplicate key")
    return parsed


def _check_value(name: str, value: str) -> None:
    if _SAFE_VALUE.fullmatch(value) is None:
        raise ValueError(f"{name}: value must not contain whitespace, quotes, '#', '$', '`' or backslashes")


def render_env(vars: Sequence[EnvVar], values: Mapping[str, str]) -> str:
    """``.env`` text: each key's comment then ``NAME=value`` (the value, else the default, else empty)."""
    blocks: list[str] = []
    for var in vars:
        value = values.get(var.name, var.default or "")
        _check_value(var.name, value)
        comment = "\n".join(f"# {line}" for line in var.comment.splitlines())
        blocks.append(f"{comment}\n{var.name}={value}\n")
    return "\n".join(blocks)


def write_env(path: Path, content: str, *, overwrite: bool) -> None:
    """Write ``content`` with mode 0600; ``FileExistsError`` if ``path`` exists and not ``overwrite``.

    The file is created 0600 from the start (never briefly world-readable); an overwrite goes through
    a temp file in the same directory and an atomic rename.
    """
    if not overwrite:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(content)
        return
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")  # mode 0600
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(content)
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def prompt_values(
    vars: Sequence[EnvVar],
    existing: Mapping[str, str | None],
    *,
    ask: Callable[[str], str],
    ask_secret: Callable[[str], str],
) -> dict[str, str]:
    """Ask for every key, secrets through ``ask_secret`` (getpass); Enter keeps the current value.

    The current value is the existing ``.env``'s, else the default. A secret's current value is never
    shown, only whether it is set. An invalid value is asked again, up to three times.
    """
    values: dict[str, str] = {}
    for var in bounded(vars, len(vars), "env vars"):
        current = existing.get(var.name) or var.default or ""
        print("\n".join(f"# {line}" for line in var.comment.splitlines()))
        if var.secret:
            prompt = f"{var.name} [{'set; Enter keeps it' if current else 'not set'}]: "
        else:
            prompt = f"{var.name} [{current}]: "
        for attempt in bounded(range(_MAX_ATTEMPTS), _MAX_ATTEMPTS, f"{var.name} attempts"):
            value = (ask_secret if var.secret else ask)(prompt).strip() or current
            try:
                _check_value(var.name, value)
            except ValueError as error:
                if attempt == _MAX_ATTEMPTS - 1:
                    raise
                print(f"{error}; try again")
                continue
            values[var.name] = value
            break
    return values
