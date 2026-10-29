#!/usr/bin/env python3
"""Static audit of a pytest suite against the pytest-expert hard gates.

Pure stdlib (ast, tokenize, tomllib). It never imports the code under test, so it
is safe on any repo and fast enough for a pre-commit hook or a Claude Code hook.

Usage:
    python audit_tests.py tests/
    python audit_tests.py tests/test_wallet.py
    python audit_tests.py tests/ --ignore PRINT,BRANCHING
    python audit_tests.py tests/ --format json|github|sarif
    python audit_tests.py --list-codes

Inline suppression (a reason after the bracket is required):
    x = compute()  # audit: ignore[FLOAT_EQ] exact binary fraction
    def test_x():  # audit: ignore[NO_ASSERT] smoke test: must not raise
    # audit: ignore-file[PRINT] demo module

Exit status: 1 if any error-severity finding remains (or any warning, with
--warnings-as-errors), 2 for usage errors, 0 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import configparser
import io
import json
import re
import sys
import tokenize
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    tomllib = None

ERROR, WARNING = "error", "warning"

CODES: dict[str, tuple[str, str]] = {
    "NO_ASSERT": (ERROR, "test contains no assert, pytest.raises/warns/fail, or real mock assertion"),
    "ALWAYS_TRUE": (ERROR, "assertion can never fail (truthy literal, non-empty tuple/list/dict/set)"),
    "BARE_COMPARISON": (ERROR, "comparison used as a statement; it is computed and thrown away, add assert"),
    "NAN_COMPARE": (ERROR, "== / != against NaN is always False / True; use math.isnan or nan_ok"),
    "FLOAT_EQ": (ERROR, "float literal compared with == or !=; use pytest.approx"),
    "SWALLOWED_FAILURE": (ERROR, "except AssertionError/Exception/bare without re-raise; failures vanish"),
    "FAKE_MOCK_ASSERT": (ERROR, "mock method that silently passes; Mock auto-creates any attribute"),
    "BROAD_RAISES": (ERROR, "pytest.raises on Exception/BaseException with no match=; passes on the wrong error"),
    "WEAK_ASSERT": (WARNING, "every assertion is existence-only (is not None, isinstance, len > 0)"),
    "SLEEP": (ERROR, "time.sleep in a test or fixture; slow and flaky, inject a clock"),
    "NETWORK": (ERROR, "real network call in a non-integration test"),
    "UNSEEDED_RANDOM": (ERROR, "unseeded randomness and no seeding conftest; failures become unreproducible"),
    "PRINT": (WARNING, "print() in a test; assert instead, or use capsys/caplog"),
    "BRANCHING": (ERROR, "if/while or for-over-cases in a test body; use parametrize"),
    "ENV_MUTATION": (ERROR, "os.environ mutated directly; use monkeypatch.setenv/delenv"),
    "SYS_PATH": (ERROR, "sys.path mutated in a test; fix packaging or use monkeypatch.syspath_prepend"),
    "SKIP_NO_REASON": (ERROR, "skip/skipif/xfail without a reason; the report will not say why"),
    "XFAIL_NOT_STRICT": (ERROR, "xfail without strict=True and config does not set strict_xfail"),
    "PATCH_NO_SPEC": (ERROR, "mock.patch / patch.object without autospec/spec/new; the double drifts from the API"),
    "UNITTEST_CLASS": (ERROR, "unittest.TestCase subclass; plain functions get fixtures and assert rewriting"),
    "MUTABLE_MODULE_STATE": (ERROR, "module-level mutable is mutated by tests; state leaks between tests"),
    "SHADOWED_TEST_NAME": (ERROR, "two tests with the same name in one scope; the first never runs"),
    "SUPPRESSION_NO_REASON": (ERROR, "audit: ignore[...] pragma without a reason after the bracket"),
    "SYNTAX": (ERROR, "file does not parse"),
}

REAL_MOCK_ASSERTIONS = frozenset({
    "assert_called", "assert_called_once", "assert_called_with", "assert_called_once_with",
    "assert_any_call", "assert_has_calls", "assert_not_called", "assert_awaited",
    "assert_awaited_once", "assert_awaited_with", "assert_awaited_once_with",
    "assert_any_await", "assert_has_awaits", "assert_not_awaited",
})
FAKE_MOCK_ATTRS = frozenset({"called_once", "called_once_with", "called_with", "not_called", "has_calls"})
ASSERTING_CALL_TAILS = frozenset({"raises", "warns", "deprecated_call", "fail"})
NETWORK_ROOTS = frozenset({"requests", "httpx", "urllib", "urllib3", "socket", "aiohttp", "http"})
SWALLOWING = frozenset({"AssertionError", "Exception", "BaseException"})
NAN_EXPRS = frozenset({"math.nan", "np.nan", "numpy.nan", "torch.nan", "cmath.nan"})
INTEGRATION_MARKERS = frozenset({"integration", "e2e"})
SEED_PATTERN = re.compile(
    r"random\.seed\(|manual_seed\(|np\.random\.seed\(|numpy\.random\.seed\(|default_rng\(\s*[\w.]"
)
PRAGMA = re.compile(r"#\s*audit:\s*(ignore|ignore-file)\[([A-Z_,\s]+)\]\s*(.*)$")


@dataclass(frozen=True)
class Finding:
    file: str
    line: int
    code: str
    detail: str
    severity: str

    def render(self) -> str:
        return f"{self.file}:{self.line}: {self.code}: {self.detail}"


def dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def call_name(node: ast.AST) -> str:
    return dotted(node.func) if isinstance(node, ast.Call) else dotted(node)


def has_kw(call: ast.Call, *names: str) -> bool:
    return any(kw.arg in names for kw in call.keywords)


def is_float_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        node = node.operand
    return isinstance(node, ast.Constant) and isinstance(node.value, float)


def is_nan_expr(node: ast.AST) -> bool:
    if dotted(node) in NAN_EXPRS:
        return True
    if isinstance(node, ast.Call) and call_name(node) in {"float", "np.float64", "numpy.float64"}:
        return bool(node.args) and isinstance(node.args[0], ast.Constant) and str(node.args[0].value).lower() == "nan"
    return False


def is_test_function(node: ast.AST) -> bool:
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test")


def is_fixture(node: ast.AST) -> bool:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    return any(call_name(d).endswith("fixture") for d in node.decorator_list)


def marker_names(decorators: list[ast.expr]) -> set[str]:
    names = set()
    for dec in decorators:
        name = call_name(dec)
        if name.startswith(("pytest.mark.", "mark.")):
            names.add(name.rsplit(".", 1)[-1])
    return names


def module_markers(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in stmt.targets):
            values = stmt.value.elts if isinstance(stmt.value, (ast.List, ast.Tuple)) else [stmt.value]
            names |= marker_names(values)
    return names


# --------------------------------------------------------------------------- config
def find_config_strict_xfail(start: Path) -> bool:
    """True if the nearest pytest config makes xfail strict by default."""
    for directory in [start, *start.parents]:
        for name in ("pytest.toml", ".pytest.toml", "pyproject.toml"):
            path = directory / name
            if path.is_file() and tomllib is not None:
                try:
                    data = tomllib.loads(path.read_text(encoding="utf-8"))
                except (tomllib.TOMLDecodeError, OSError):
                    continue
                section = data.get("pytest") if name != "pyproject.toml" else data.get("tool", {}).get("pytest", {})
                if name == "pyproject.toml":
                    section = data.get("tool", {}).get("pytest", {}).get("ini_options", section) or section
                if section:
                    return _truthy(section, ("strict", "strict_xfail", "xfail_strict"))
        for name, sect in (("pytest.ini", "pytest"), ("tox.ini", "pytest"), ("setup.cfg", "tool:pytest")):
            path = directory / name
            if path.is_file():
                parser = configparser.ConfigParser()
                try:
                    parser.read(path, encoding="utf-8")
                except configparser.Error:
                    continue
                if parser.has_section(sect):
                    return _truthy(dict(parser[sect]), ("strict", "strict_xfail", "xfail_strict"))
    return False


def _truthy(section: dict, keys: tuple[str, ...]) -> bool:
    return any(str(section.get(k, "")).strip().lower() in {"true", "1", "yes"} for k in keys)


def suite_is_seeded(root: Path) -> bool:
    base = root if root.is_dir() else root.parent
    candidates = list(base.rglob("conftest.py")) + [p / "conftest.py" for p in base.parents]
    return any(p.is_file() and SEED_PATTERN.search(p.read_text(encoding="utf-8", errors="replace")) for p in candidates)


# --------------------------------------------------------------------------- pragmas
def read_pragmas(source: str) -> tuple[dict[int, set[str]], set[str], list[int]]:
    line_codes: dict[int, set[str]] = {}
    file_codes: set[str] = set()
    missing_reason: list[int] = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return line_codes, file_codes, missing_reason
    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        match = PRAGMA.search(tok.string)
        if not match:
            continue
        kind, codes_text, reason = match.groups()
        codes = {c.strip() for c in codes_text.split(",") if c.strip()}
        if not reason.strip():
            missing_reason.append(tok.start[0])
            continue
        if kind == "ignore-file":
            file_codes |= codes
        else:
            line_codes.setdefault(tok.start[0], set()).update(codes)
    return line_codes, file_codes, missing_reason


# --------------------------------------------------------------------------- auditor
class Auditor(ast.NodeVisitor):
    def __init__(self, path: Path, seeded: bool, strict_xfail: bool, tree: ast.Module) -> None:
        self.path = str(path)
        self.seeded = seeded
        self.strict_xfail = strict_xfail
        self.findings: list[Finding] = []
        self.fn_stack: list[ast.AST] = []
        self.scopes: list[dict[str, int]] = [{}]
        parts = {p.lower() for p in path.parts}
        self.integration_module = bool(parts & INTEGRATION_MARKERS) or bool(module_markers(tree) & INTEGRATION_MARKERS)

    def add(self, node: ast.AST, code: str, detail: str = "") -> None:
        severity, default = CODES[code]
        self.findings.append(Finding(self.path, getattr(node, "lineno", 0), code, detail or default, severity))

    def current_test(self):
        return next((f for f in reversed(self.fn_stack) if is_test_function(f)), None)

    def in_test_or_fixture(self) -> bool:
        return any(is_test_function(f) or is_fixture(f) for f in self.fn_stack)

    # module ---------------------------------------------------------------
    def visit_Module(self, node: ast.Module) -> None:
        globals_: dict[str, int] = {}
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, (ast.List, ast.Dict, ast.Set)):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id != "pytestmark":
                        globals_[target.id] = stmt.lineno
        mutators = {"append", "extend", "add", "update", "insert", "pop", "clear", "setdefault", "remove", "discard"}
        for sub in ast.walk(node):
            name = ""
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and isinstance(sub.func.value, ast.Name):
                if sub.func.attr in mutators:
                    name = sub.func.value.id
            elif isinstance(sub, (ast.Assign, ast.AugAssign, ast.Delete)):
                targets = sub.targets if isinstance(sub, (ast.Assign, ast.Delete)) else [sub.target]
                for target in targets:
                    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                        name = target.value.id
            if name in globals_:
                self.add(sub, "MUTABLE_MODULE_STATE",
                         f"{name!r} (defined line {globals_[name]}) is mutated; move it into a fixture")
                globals_.pop(name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if any(dotted(b) in {"unittest.TestCase", "TestCase", "unittest.IsolatedAsyncioTestCase"} for b in node.bases):
            self.add(node, "UNITTEST_CLASS", f"{node.name} subclasses unittest.TestCase")
        self.scopes.append({})
        self.generic_visit(node)
        self.scopes.pop()

    # functions ------------------------------------------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function(node)

    def _function(self, node) -> None:
        if is_test_function(node) and not self.fn_stack:
            scope = self.scopes[-1]
            if node.name in scope:
                self.add(node, "SHADOWED_TEST_NAME", f"{node.name} already defined at line {scope[node.name]}; that one never runs")
            scope[node.name] = node.lineno
            self._check_markers(node)
            self._check_assertions(node)
            self._check_branching(node)
        self.fn_stack.append(node)
        self.generic_visit(node)
        self.fn_stack.pop()

    def _check_markers(self, node) -> None:
        for dec in node.decorator_list:
            name = call_name(dec)
            if not name.startswith(("pytest.mark.", "mark.")):
                continue
            short = name.rsplit(".", 1)[-1]
            if short in {"skip", "skipif", "xfail"}:
                positional_reason = (
                    short == "skip" and isinstance(dec, ast.Call) and dec.args
                    and isinstance(dec.args[0], ast.Constant) and isinstance(dec.args[0].value, str)
                )
                if not (isinstance(dec, ast.Call) and (has_kw(dec, "reason") or positional_reason)):
                    self.add(dec, "SKIP_NO_REASON", f"@{name} has no reason")
            if short == "xfail" and not self.strict_xfail and not (isinstance(dec, ast.Call) and has_kw(dec, "strict")):
                self.add(dec, "XFAIL_NOT_STRICT")

    def _check_assertions(self, node) -> None:
        asserts: list[ast.Assert] = []
        other_assertion = False
        for sub in ast.walk(node):
            if isinstance(sub, ast.Assert):
                asserts.append(sub)
            elif isinstance(sub, ast.Call):
                name = call_name(sub)
                tail = name.rsplit(".", 1)[-1]
                if (tail in ASSERTING_CALL_TAILS and (name.startswith("pytest.") or name == tail)) or (tail.startswith("assert") and tail not in FAKE_MOCK_ATTRS):
                    other_assertion = True
            elif isinstance(sub, ast.withitem) and "raises" in call_name(sub.context_expr):
                other_assertion = True
        if not asserts and not other_assertion:
            self.add(node, "NO_ASSERT", f"{node.name} asserts nothing")
        elif asserts and not other_assertion and all(self._is_weak(a.test) for a in asserts):
            self.add(node, "WEAK_ASSERT", f"{node.name}: every assertion only checks existence or type")

    @staticmethod
    def _is_weak(test: ast.AST) -> bool:
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            op, right, left = test.ops[0], test.comparators[0], test.left
            if isinstance(op, ast.IsNot) and isinstance(right, ast.Constant) and right.value is None:
                return True
            if call_name(left) == "len" and isinstance(right, ast.Constant) and right.value in (0, 1):
                return isinstance(op, (ast.Gt, ast.NotEq)) or (isinstance(op, ast.GtE) and right.value == 1)
        return isinstance(test, ast.Call) and call_name(test) in {"isinstance", "hasattr", "callable", "len"}

    def _check_branching(self, node) -> None:
        for stmt in node.body:
            for inner in ast.walk(stmt):
                if isinstance(inner, (ast.If, ast.While)):
                    self.add(inner, "BRANCHING", "if/while in a test body; parametrize the cases")
                    return
                if isinstance(inner, ast.For) and not isinstance(inner.iter, ast.Call):
                    self.add(inner, "BRANCHING", "for-loop over cases; parametrize the cases")
                    return

    # statements -----------------------------------------------------------
    def visit_Expr(self, node: ast.Expr) -> None:
        if self.current_test() is not None and isinstance(node.value, ast.Compare):
            self.add(node, "BARE_COMPARISON", f"`{ast.unparse(node.value)[:60]}` is not asserted")
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        if self.current_test() is not None:
            for handler in node.handlers:
                caught = self._caught_names(handler.type)
                if caught is None or caught & SWALLOWING:
                    reraises = any(
                        isinstance(s, ast.Raise)
                        or (isinstance(s, ast.Call) and call_name(s) in {"pytest.fail", "fail"})
                        for s in ast.walk(ast.Module(body=handler.body, type_ignores=[]))
                    )
                    if not reraises:
                        label = "bare except" if caught is None else "except " + "/".join(sorted(caught & SWALLOWING))
                        self.add(handler, "SWALLOWED_FAILURE", f"{label} without re-raise hides failures")
        self.generic_visit(node)

    visit_TryStar = visit_Try

    @staticmethod
    def _caught_names(type_node) -> set[str] | None:
        if type_node is None:
            return None
        nodes = type_node.elts if isinstance(type_node, ast.Tuple) else [type_node]
        return {dotted(n).rsplit(".", 1)[-1] for n in nodes}

    def visit_Assert(self, node: ast.Assert) -> None:
        test = node.test
        if isinstance(test, ast.Constant) and test.value is not None and bool(test.value):
            self.add(node, "ALWAYS_TRUE", f"assert {ast.unparse(test)[:40]} always passes")
        if isinstance(test, (ast.Tuple, ast.List, ast.Set, ast.Dict)) and (getattr(test, "elts", None) or getattr(test, "keys", None)):
            self.add(node, "ALWAYS_TRUE", "asserting a non-empty literal container always passes")
        if isinstance(test, ast.Attribute) and test.attr in FAKE_MOCK_ATTRS:
            self.add(node, "FAKE_MOCK_ASSERT", f"assert x.{test.attr} is truthy for any Mock")
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if self.current_test() is not None:
            operands = [node.left, *node.comparators]
            for i, op in enumerate(node.ops):
                if not isinstance(op, (ast.Eq, ast.NotEq)):
                    continue
                left, right = operands[i], operands[i + 1]
                if is_nan_expr(left) or is_nan_expr(right):
                    self.add(node, "NAN_COMPARE")
                elif is_float_literal(left) or is_float_literal(right):
                    literal = left if is_float_literal(left) else right
                    self.add(node, "FLOAT_EQ", f"float {ast.unparse(literal)} compared with ==; use pytest.approx")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = call_name(node)
        tail = name.rsplit(".", 1)[-1]
        root = name.split(".", 1)[0]
        test = self.current_test()

        if tail.startswith("assert_") and tail not in REAL_MOCK_ASSERTIONS and ("call" in tail or "await" in tail):
            self.add(node, "FAKE_MOCK_ASSERT", f"{name}() is not a real Mock assertion")
        if tail in FAKE_MOCK_ATTRS and isinstance(node.func, ast.Attribute):
            self.add(node, "FAKE_MOCK_ASSERT", f"{name}() returns a truthy Mock and asserts nothing")

        if tail == "raises" and (name.startswith("pytest.") or name == "raises") and node.args:
            exc = node.args[0]
            names = {dotted(e) for e in (exc.elts if isinstance(exc, ast.Tuple) else [exc])}
            if names & {"Exception", "BaseException"} and not has_kw(node, "match"):
                self.add(node, "BROAD_RAISES", f"pytest.raises({ast.unparse(exc)}) with no match=")

        if name in {"time.sleep", "sleep", "asyncio.sleep", "anyio.sleep", "trio.sleep"} and self.in_test_or_fixture():
            zero = node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == 0
            if not zero:
                self.add(node, "SLEEP", f"{name}() in a test or fixture")

        if test is not None and root in NETWORK_ROOTS:
            integration = self.integration_module or bool(marker_names(test.decorator_list) & INTEGRATION_MARKERS)
            if not integration:
                self.add(node, "NETWORK", f"{name}() reaches the network; stub it or mark the test integration")

        if test is not None and not self.seeded:
            seeded_ctor = name in {"random.Random", "np.random.default_rng", "numpy.random.default_rng", "np.random.RandomState"} and node.args
            if not seeded_ctor and (root == "random" or name.startswith(("np.random.", "numpy.random."))):
                self.add(node, "UNSEEDED_RANDOM", f"{name}() with no seed")

        if test is not None and name == "print":
            self.add(node, "PRINT")

        patch_like = name in {"patch", "mock.patch", "unittest.mock.patch", "mocker.patch", "patch.object",
                              "mock.patch.object", "unittest.mock.patch.object", "mocker.patch.object"}
        if patch_like and not has_kw(node, "autospec", "spec", "spec_set", "new", "new_callable"):
            positional_new = (name.endswith(".object") and len(node.args) >= 3) or (not name.endswith(".object") and len(node.args) >= 2)
            if not positional_new:
                self.add(node, "PATCH_NO_SPEC", f"{name}() without autospec=True")

        if name in {"sys.path.insert", "sys.path.append", "sys.path.extend"}:
            self.add(node, "SYS_PATH", f"{name}()")

        if name in {"os.environ.update", "os.environ.setdefault", "os.environ.pop", "os.environ.clear", "os.putenv", "os.unsetenv"}:
            self.add(node, "ENV_MUTATION", f"{name}(); use monkeypatch")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Subscript) and dotted(target.value) == "os.environ":
                self.add(node, "ENV_MUTATION", "os.environ[...] = ...; use monkeypatch.setenv")
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            if isinstance(target, ast.Subscript) and dotted(target.value) == "os.environ":
                self.add(node, "ENV_MUTATION", "del os.environ[...]; use monkeypatch.delenv")
        self.generic_visit(node)


# --------------------------------------------------------------------------- driver
def collect(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    skip = {".venv", "venv", ".tox", ".nox", "node_modules", ".git", "build", "dist", "mutants", "site-packages"}
    return sorted(
        p for p in root.rglob("*.py")
        if not skip & set(p.parts) and (p.name.startswith("test_") or p.name.endswith("_test.py") or p.name == "conftest.py")
    )


def function_ranges(tree: ast.Module) -> list[tuple[int, int, int]]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            out.append((first, node.lineno, node.end_lineno or node.lineno))
    return out


def audit_file(path: Path, seeded: bool, strict_xfail: bool) -> list[Finding]:
    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [Finding(str(path), exc.lineno or 0, "SYNTAX", str(exc.msg), ERROR)]
    auditor = Auditor(path, seeded, strict_xfail, tree)
    auditor.visit(tree)
    line_codes, file_codes, missing = read_pragmas(source)
    ranges = function_ranges(tree)

    def suppressed(f: Finding) -> bool:
        if f.code in file_codes or f.code in line_codes.get(f.line, set()):
            return True
        for first, def_line, end in ranges:
            if first <= f.line <= end:
                for line in range(first, def_line + 1):
                    if f.code in line_codes.get(line, set()):
                        return True
        return False

    kept = [f for f in auditor.findings if not suppressed(f)]
    kept += [Finding(str(path), line, "SUPPRESSION_NO_REASON", CODES["SUPPRESSION_NO_REASON"][1], ERROR) for line in missing]
    return kept


def audit(root: Path, ignore: frozenset[str] = frozenset()) -> list[Finding]:
    base = root if root.is_dir() else root.parent
    seeded = suite_is_seeded(root)
    strict_xfail = find_config_strict_xfail(base.resolve())
    findings = [f for p in collect(root) for f in audit_file(p, seeded, strict_xfail) if f.code not in ignore]
    return sorted(set(findings), key=lambda f: (f.file, f.line, f.code))


def to_sarif(findings: list[Finding]) -> dict:
    rules = [{"id": c, "shortDescription": {"text": d}, "defaultConfiguration": {"level": s}} for c, (s, d) in sorted(CODES.items())]
    results = [{
        "ruleId": f.code, "level": f.severity, "message": {"text": f.detail},
        "locations": [{"physicalLocation": {"artifactLocation": {"uri": Path(f.file).as_posix()},
                                            "region": {"startLine": max(f.line, 1)}}}],
    } for f in findings]
    return {"version": "2.1.0", "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "runs": [{"tool": {"driver": {"name": "pytest-expert-audit", "rules": rules}}, "results": results}]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, nargs="?", help="test directory or single test file")
    parser.add_argument("--ignore", default="", help="comma-separated codes to suppress everywhere")
    parser.add_argument("--format", choices=("text", "json", "github", "sarif"), default="text")
    parser.add_argument("--warnings-as-errors", action="store_true", help="exit 1 on warning-severity findings too")
    parser.add_argument("--list-codes", action="store_true", help="print every code and exit")
    args = parser.parse_args(argv)

    if args.list_codes:
        for code, (severity, why) in sorted(CODES.items()):
            print(f"{code:23} {severity:8} {why}")
        return 0
    if args.path is None or not args.path.exists():
        print(f"audit_tests: no such path: {args.path}", file=sys.stderr)
        return 2

    ignore = frozenset(c.strip().upper() for c in args.ignore.split(",") if c.strip())
    findings = audit(args.path, ignore)

    if args.format == "json":
        print(json.dumps([asdict(f) for f in findings], indent=2))
    elif args.format == "sarif":
        print(json.dumps(to_sarif(findings), indent=2))
    elif args.format == "github":
        for f in findings:
            level = "error" if f.severity == ERROR else "warning"
            print(f"::{level} file={f.file},line={max(f.line, 1)},title={f.code}::{f.detail}")
    else:
        for f in findings:
            print(f.render() + ("" if f.severity == ERROR else "  [warning]"))
        counts: dict[str, int] = {}
        for f in findings:
            counts[f.code] = counts.get(f.code, 0) + 1
        print(f"\n{len(findings)} finding(s): " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) if findings else "clean: no findings")

    failing = [f for f in findings if f.severity == ERROR or args.warnings_as_errors]
    return 1 if failing else 0


if __name__ == "__main__":
    raise SystemExit(main())
