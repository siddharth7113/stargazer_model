"""Pre-commit hook: refuse single-letter variable names in the package.

Usage: ``python scripts/check_names.py stargazer/*.py``. Exit code 1 lists each offender.
This is a lint rule, so it runs in pre-commit, not in the test suite.
"""

import ast
import sys
from pathlib import Path

ALLOWED_SHORT = {"_", "t0", "u1", "u2", "b", "rp_rs", "a_rs"}


def short_names(tree: ast.AST) -> list[tuple[str, int]]:
    """Return ``(name, line)`` for each single-letter store or argument not in the allow list."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            found.add((node.id, node.lineno))
        elif isinstance(node, ast.arg):
            found.add((node.arg, node.lineno))
    return sorted((name, line) for name, line in found if len(name) == 1 and name not in ALLOWED_SHORT)


def main(paths: list[str]) -> int:
    """Check each path. Return 1 if any offender exists."""
    status = 0
    for path in paths:
        for name, line in short_names(ast.parse(Path(path).read_text())):
            print(f"{path}:{line}: single-letter name '{name}'")
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
