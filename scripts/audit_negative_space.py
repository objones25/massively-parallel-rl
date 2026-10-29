#!/usr/bin/env python3
"""Audit Python source for missing negative space.

Checks (all AST-based, standard library only, Python 3.10+):

  NSP001  function with fewer than the minimum number of runtime checks
  NSP002  assert on a tuple literal -- always true, never fires
  NSP003  side effect inside a check -- behaviour changes when checks are off
  NSP004  compound check `a and b` -- split so the failure names itself
  NSP005  unbounded loop -- `while True` with no bound, or `for` over an
          infinite iterator (itertools.count/cycle, repeat(x), iter(f, sentinel))
  NSP006  bare `except:` -- catches SystemExit and KeyboardInterrupt too
  NSP007  swallowed exception -- handler neither raises nor logs
  NSP008  bare `assert` outside tests (--strict only) -- removed by python -O
  NSP009  unused `# nsp: ignore` -- the suppression no longer matches a finding

Usage:
    python audit_negative_space.py PATH [PATH ...]
    python audit_negative_space.py src/ --min-assertions 2
    python audit_negative_space.py src/ --strict --json
    python audit_negative_space.py src/ --select NSP005,NSP007

Exits 1 if any finding is reported, 0 if clean. Suppress a finding with a
trailing comment on the line it is reported on: `# nsp: ignore` (all codes) or
`# nsp: ignore NSP005 <reason>`. Only real comments count; the text inside a
string literal is not a suppression.

This file audits clean against itself at `--min-assertions 1`, the right dial
for dispatch-heavy code; `assets/examples/after.py` is clean at the default 2.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import sys
import tokenize
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import override
else:
    try:
        from typing import override
    except ImportError:  # Python < 3.12: the decorator only matters to checkers

        def override(func):  # noqa: ANN001, ANN201
            return func


CHECK_FUNCS = frozenset({"require", "invariant", "precondition", "postcondition",
                         "ensure", "check", "unreachable", "verify", "expect",
                         "_require"})
LOG_ATTRS = frozenset({"debug", "info", "warning", "warn", "error", "exception",
                       "critical", "log", "print", "capture_exception"})
MUTATORS = frozenset({"append", "extend", "insert", "pop", "remove", "clear",
                      "update", "add", "discard", "sort", "write", "writelines",
                      "send", "close", "commit", "execute", "setdefault",
                      "popitem", "seek", "flush", "put", "get_nowait"})
BOUND_HINTS = frozenset({"bounded", "range", "islice", "takewhile", "zip"})
INFINITE_ITERATORS = frozenset({"count", "cycle"})
DENSITY_EXEMPT_DECORATORS = frozenset({"overload", "abstractmethod", "property",
                                       "setter", "fixture", "singledispatch"})
SKIP_DIRS = frozenset({".venv", "venv", "__pycache__", "build", ".git",
                       "site-packages", ".tox", ".mypy_cache"})
SUPPRESS_MARKER = "nsp: ignore"
ALL_CODES = "*"

SEVERITY = {
    "NSP000": "error",
    "NSP001": "warning", "NSP002": "error", "NSP003": "error", "NSP004": "warning",
    "NSP005": "error", "NSP006": "error", "NSP007": "error", "NSP008": "warning",
    "NSP009": "warning",
}

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef
# Nodes that carry a line number a finding can point at.
LocatedNode = ast.stmt | ast.expr | ast.excepthandler


class AuditBug(AssertionError):  # noqa: N818  mirrors CheckFailed in the helpers
    """The auditor itself is in an impossible state."""


def _require(condition: object, message: str) -> None:
    """Local `require` so this script stays dependency-free. Survives -O."""
    if not condition:
        raise AuditBug(message)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    code: str
    severity: str
    symbol: str
    message: str

    def render(self) -> str:
        return (f"{self.path}:{self.line}: {self.code} [{self.severity}] "
                f"{self.symbol}: {self.message}")


def _is_test_file(path: Path) -> bool:  # nsp: ignore NSP001  pure name predicate
    name = path.name
    return (name.startswith("test_") or name.endswith("_test.py")
            or "tests" in path.parts or "test" in path.parts
            or name == "conftest.py")


def _suppressions(source: str) -> dict[int, set[str]]:
    """Map line number -> suppressed codes ('*' means every code).

    Reads COMMENT tokens only, so a marker inside a string literal is ignored.
    The caller has already parsed `source`, so tokenizing it cannot fail.
    """
    out: dict[int, set[str]] = {}
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type != tokenize.COMMENT:
            continue
        text = tok.string.lstrip("#").strip()
        if not text.startswith(SUPPRESS_MARKER):
            continue
        words = text[len(SUPPRESS_MARKER):].replace(",", " ").split()
        codes = {w for w in words if w.startswith("NSP")}
        out[tok.start[0]] = codes or {ALL_CODES}
    _require(all(out.values()), "empty suppression set would silence nothing")
    _require(all(line >= 1 for line in out), "tokenize reported a line before 1")
    return out


def _call_name(func: ast.AST) -> str:  # nsp: ignore NSP001  pure total mapping
    """`f` -> 'f', `obj.f` -> 'f', anything else -> ''. Used on call targets
    and decorator expressions."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _calls_named(node: ast.AST, names: frozenset[str]) -> bool:
    _require(names, "empty name set would always return False")
    return any(isinstance(sub, ast.Call) and _call_name(sub.func) in names
               for sub in ast.walk(node))


def _is_check_stmt(node: ast.stmt) -> bool:  # nsp: ignore NSP001  pure predicate
    """An assert, a call to a check helper, or a guard clause that raises."""
    if isinstance(node, ast.Assert):
        return True
    if isinstance(node, ast.Expr | ast.Assign | ast.AnnAssign) \
            and isinstance(node.value, ast.Call):
        # `require(...)` as a statement, or `x = expect(...)` as an assignment.
        return _call_name(node.value.func) in CHECK_FUNCS
    if isinstance(node, ast.If) and len(node.body) == 1:
        return isinstance(node.body[0], ast.Raise)
    return False


def _count_checks(fn: FunctionNode) -> int:
    """Checks anywhere in the function, excluding nested function bodies."""
    total = 0
    visited = 0
    stack: list[ast.AST] = [fn]
    while stack:
        node = stack.pop()
        visited += 1
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda
                          | ast.ClassDef):
                continue
            if isinstance(child, ast.stmt) and _is_check_stmt(child):
                total += 1
            stack.append(child)
    _require(visited >= 1, "walk did not visit the function itself")
    _require(total <= visited, "counted more checks than nodes visited")
    return total


def _body_statements(fn: FunctionNode) -> list[ast.stmt]:
    """Body with the docstring (first statement, if a string literal) removed."""
    first = fn.body[0]
    has_doc = (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
               and isinstance(first.value.value, str))
    body = fn.body[1:] if has_doc else fn.body
    _require(len(fn.body) - len(body) == int(has_doc), "docstring removal miscounted")
    return body


def _is_trivial(fn: FunctionNode) -> bool:  # nsp: ignore NSP001  pure predicate
    """Stubs, protocol members, and one-statement wrappers are exempt from density."""
    body = _body_statements(fn)
    if not body:
        return True
    if len(body) == 1:
        # A single-statement wrapper has little space to get wrong -- unless that
        # statement is a loop or a try, which carry their own negative space.
        return not isinstance(body[0], ast.While | ast.For | ast.AsyncFor | ast.Try)
    return False


def _decorator_names(fn: FunctionNode) -> set[str]:
    return {name for dec in fn.decorator_list for sub in ast.walk(dec)
            if (name := _call_name(sub))}


def _handler_swallows(handler: ast.ExceptHandler) -> bool:
    """True when the handler neither re-raises nor reports."""
    _require(handler.body, "an except handler cannot have an empty body")
    if any(isinstance(sub, ast.Raise) for sub in ast.walk(handler)):
        return False
    if _calls_named(handler, LOG_ATTRS):
        return False
    meaningful = [s for s in handler.body
                  if not isinstance(s, ast.Pass | ast.Continue | ast.Break)]
    return not meaningful


def _loop_is_bounded(loop: ast.While) -> bool:
    """A `while True` counts as bounded when its body checks an iteration count
    or walks a bounded iterator. Heuristic: any check or bound-hint call counts."""
    _require(loop.body, "a while loop cannot have an empty body")
    for sub in ast.walk(loop):
        if isinstance(sub, ast.stmt) and _is_check_stmt(sub):
            return True
        if isinstance(sub, ast.Call) and _call_name(sub.func) in BOUND_HINTS:
            return True
    return False


def _is_infinite_iterator(node: ast.expr) -> bool:  # nsp: ignore NSP001  pure predicate
    """`count()`, `cycle(x)`, `repeat(x)` without times, `iter(f, sentinel)`."""
    if not isinstance(node, ast.Call):
        return False
    name = _call_name(node.func)
    if name in INFINITE_ITERATORS:
        return True
    if name == "repeat":
        no_times = len(node.args) <= 1
        return no_times and not any(k.arg == "times" for k in node.keywords)
    return name == "iter" and len(node.args) > 1  # iter(callable, sentinel)


class Auditor(ast.NodeVisitor):
    def __init__(self, path: Path, *, min_assertions: int, strict: bool) -> None:
        _require(min_assertions >= 0,
                 f"min_assertions must be non-negative, got {min_assertions}")
        _require(path.suffix == ".py", f"not a Python file: {path}")
        self.path = path
        self.min_assertions = min_assertions
        self.strict = strict
        self.is_test = _is_test_file(path)
        self.findings: list[Finding] = []
        self._scope: list[str] = []

    def add(self, node: LocatedNode, code: str, message: str) -> None:
        _require(code in SEVERITY, f"unknown finding code {code}")
        _require(node.lineno > 0, f"{code} on a node with no line number")
        self.findings.append(Finding(
            path=str(self.path), line=node.lineno, code=code,
            severity=SEVERITY[code], symbol=".".join(self._scope) or "<module>",
            message=message,
        ))

    # --- scopes -----------------------------------------------------------
    @override
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        depth = len(self._scope)
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()
        _require(len(self._scope) == depth, "scope stack unbalanced after class")

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: FunctionNode) -> None:
        depth = len(self._scope)
        self._scope.append(node.name)
        if not self.is_test:
            self._check_density(node)
        self.generic_visit(node)
        self._scope.pop()
        _require(len(self._scope) == depth, "scope stack unbalanced after function")

    def _check_density(self, node: FunctionNode) -> None:
        if _is_trivial(node) or node.name.startswith("__"):
            return
        if _decorator_names(node) & DENSITY_EXEMPT_DECORATORS:
            return
        count = _count_checks(node)
        if count < self.min_assertions:
            stmts = len(_body_statements(node))
            _require(stmts >= 1, "a non-trivial function has a statement")
            self.add(node, "NSP001",
                     f"{count} runtime check(s) in {stmts} statements; minimum is "
                     f"{self.min_assertions}. What must never be true when this "
                     f"returns?")

    # --- checks -----------------------------------------------------------
    @override
    def visit_Assert(self, node: ast.Assert) -> None:  # nsp: ignore NSP001 dispatch
        if isinstance(node.test, ast.Tuple) and node.test.elts:
            self.add(node, "NSP002",
                     "assert on a tuple literal is always true; drop the parentheses "
                     "or use require(cond, msg)")
        self._check_compound(node, node.test, "assert")
        self._check_side_effect(node, node.test, "assert")
        if self.strict and not self.is_test:
            self.add(node, "NSP008",
                     "plain assert is removed by python -O; use require() for a check "
                     "that must survive production")
        self.generic_visit(node)

    @override
    def visit_Expr(self, node: ast.Expr) -> None:  # nsp: ignore NSP001 dispatch
        if isinstance(node.value, ast.Call):
            name = _call_name(node.value.func)
            if name in CHECK_FUNCS and node.value.args:
                self._check_compound(node, node.value.args[0], name)
                self._check_side_effect(node, node.value.args[0], name)
        self.generic_visit(node)

    def _check_compound(self, node: LocatedNode, test: ast.expr, kind: str) -> None:
        _require(kind, "every finding needs the kind of check that produced it")
        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
            _require(len(test.values) > 1, "an `and` has at least two operands")
            self.add(node, "NSP004",
                     f"compound {kind}: split into {len(test.values)} checks so a "
                     f"failure names the half that broke")

    def _check_side_effect(self, node: LocatedNode, test: ast.expr, kind: str) -> None:
        _require(kind, "every finding needs the kind of check that produced it")
        for sub in ast.walk(test):
            if isinstance(sub, ast.NamedExpr):
                self.add(node, "NSP003",
                         f"walrus assignment inside {kind}; the binding vanishes when "
                         f"checks are disabled")
                return
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) \
                    and sub.func.attr in MUTATORS:
                self.add(node, "NSP003",
                         f"{kind} calls .{sub.func.attr}(), which mutates state; "
                         f"checks must be pure")
                return

    @override
    def visit_While(self, node: ast.While) -> None:  # nsp: ignore NSP001 dispatch
        test = node.test
        infinite = isinstance(test, ast.Constant) and bool(test.value) is True
        if infinite and not _loop_is_bounded(node):
            self.add(node, "NSP005",
                     "`while True` with no iteration bound; add an explicit maximum "
                     "and assert it, or mark it `# nsp: ignore NSP005` if it is a "
                     "deliberate event loop")
        self.generic_visit(node)

    @override
    def visit_For(self, node: ast.For) -> None:  # nsp: ignore NSP001 dispatch
        self._check_for_iterator(node)
        self.generic_visit(node)

    @override
    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:  # nsp: ignore NSP001 dispatch
        self._check_for_iterator(node)
        self.generic_visit(node)

    def _check_for_iterator(self, node: ast.For | ast.AsyncFor) -> None:
        if _is_infinite_iterator(node.iter):
            self.add(node, "NSP005",
                     "`for` over an infinite iterator; wrap it in bounded(it, MAX) "
                     "or islice(it, MAX)")

    @override
    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:  # nsp: ignore NSP001 dispatch
        if node.type is None:
            self.add(node, "NSP006",
                     "bare `except:` also catches SystemExit and KeyboardInterrupt; "
                     "name the exceptions you expect")
        elif _handler_swallows(node):
            self.add(node, "NSP007",
                     "handler neither raises nor logs; the failure is erased. "
                     "Re-raise, log with context, or put `# nsp: ignore NSP007 "
                     "<reason>` on the except line if it is truly ignorable")
        self.generic_visit(node)


def _apply_suppressions(findings: list[Finding], suppressed: dict[int, set[str]],
                        path: Path, *, strict: bool) -> list[Finding]:
    """Drop suppressed findings; report suppressions that matched nothing."""
    used: set[tuple[int, str]] = set()
    kept: list[Finding] = []
    for f in findings:
        codes = suppressed.get(f.line, set())
        hit = ALL_CODES if ALL_CODES in codes else (f.code if f.code in codes else None)
        if hit is None:
            kept.append(f)
        else:
            used.add((f.line, hit))
    for line, codes in sorted(suppressed.items()):
        for code in sorted(codes):
            if (line, code) in used or (code == "NSP008" and not strict):
                continue
            label = "all codes" if code == ALL_CODES else code
            kept.append(Finding(str(path), line, "NSP009", SEVERITY["NSP009"],
                                "<suppression>",
                                f"`# nsp: ignore` for {label} matches no finding; "
                                f"delete it so it cannot hide a future one"))
    _require(len(kept) >= len(findings) - len(used),
             "suppression dropped more findings than it matched")
    return kept


def audit_file(path: Path, *, min_assertions: int, strict: bool) -> list[Finding]:
    _require(path.is_file(), f"not a file: {path}")
    _require(min_assertions >= 0,
             f"min_assertions must be non-negative, got {min_assertions}")
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [Finding(str(path), exc.lineno or 0, "NSP000", "error",
                        "<module>", f"could not parse: {exc.msg}")]

    auditor = Auditor(path, min_assertions=min_assertions, strict=strict)
    auditor.visit(tree)
    return _apply_suppressions(auditor.findings, _suppressions(source), path,
                               strict=strict)


def collect(paths: list[str]) -> list[Path]:
    _require(paths, "no paths given")
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        _require(p.exists(), f"path does not exist: {raw}")
        if p.is_dir():
            files.extend(sorted(q for q in p.rglob("*.py")
                                if not (SKIP_DIRS & set(q.parts))))
        elif p.suffix == ".py":
            files.append(p)
    _require(all(f.suffix == ".py" for f in files), "collected a non-Python file")
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="files or directories to audit")
    parser.add_argument("--min-assertions", type=int, default=2,
                        help="minimum runtime checks per non-trivial function "
                             "(default: 2)")
    parser.add_argument("--strict", action="store_true",
                        help="also flag plain `assert` outside tests (NSP008)")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument("--select", default="",
                        help="comma-separated codes to report, e.g. NSP005,NSP007")
    args = parser.parse_args(argv)
    min_assertions: int = args.min_assertions
    select: str = args.select
    paths: list[str] = args.paths

    # Operating errors: these come from the command line, so they are reported
    # through argparse (exit 2), not asserted.
    if min_assertions < 0:
        parser.error("--min-assertions cannot be negative")
    selected = {c.strip().upper() for c in select.split(",") if c.strip()}
    if not selected <= set(SEVERITY):
        parser.error(f"unknown code(s): {sorted(selected - set(SEVERITY))}")
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        parser.error(f"path(s) do not exist: {missing}")

    files = collect(paths)
    findings: list[Finding] = []
    for path in files:
        findings.extend(audit_file(path, min_assertions=min_assertions,
                                   strict=bool(args.strict)))
    if selected:
        findings = [f for f in findings if f.code in selected]
    findings.sort(key=lambda f: (f.path, f.line, f.code))
    scanned = {str(p) for p in files}
    _require({f.path for f in findings} <= scanned, "finding for a file never scanned")

    if args.json:
        print(json.dumps({"files_scanned": len(files),
                          "findings": [asdict(f) for f in findings]}, indent=2))
    else:
        for f in findings:
            print(f.render())
        counts: dict[str, int] = {}
        for f in findings:
            counts[f.code] = counts.get(f.code, 0) + 1
        summary = ", ".join(f"{c}={n}" for c, n in sorted(counts.items())) or "none"
        print(f"\n{len(files)} file(s) scanned, {len(findings)} finding(s): {summary}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
