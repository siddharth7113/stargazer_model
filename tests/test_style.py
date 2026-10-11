"""Style rules that ruff does not cover, checked over the package source with the ast module."""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "stargazer"

# Names that are conventional and readable on their own.
ALLOWED_SHORT = {"_", "t0", "u1", "u2", "b", "rp_rs", "a_rs"}


def _short_names(tree: ast.AST) -> set[tuple[str, int]]:
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            found.add((node.id, node.lineno))
        elif isinstance(node, ast.arg):
            found.add((node.arg, node.lineno))
    return {(n, line) for n, line in found if len(n) == 1 and n not in ALLOWED_SHORT}


def test_no_single_letter_names_in_package():
    offenders = {}
    for path in sorted(PACKAGE.glob("*.py")):
        names = _short_names(ast.parse(path.read_text()))
        if names:
            offenders[path.name] = sorted(names, key=lambda x: x[1])
    assert not offenders, "\n".join(f"{f}: {v}" for f, v in offenders.items())
