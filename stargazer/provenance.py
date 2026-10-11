"""Helpers that make every output traceable: hashes, git commit, timestamps."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        rev = out.stdout.strip()
        return rev + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_manifest(directory: Path, pattern: str = "**/*", name: str = "manifest.json") -> Path:
    """Write SHA-256 of every file under `directory` (except the manifest itself)."""
    directory = Path(directory)
    entries = {}
    for p in sorted(directory.glob(pattern)):
        if p.is_file() and p.name != name:
            entries[str(p.relative_to(directory))] = sha256_file(p)
    out = directory / name
    out.write_text(json.dumps({"created": utc_now(), "files": entries}, indent=2, sort_keys=True))
    return out


def verify_manifest(directory: Path, name: str = "manifest.json") -> list[str]:
    """Return a list of problems; empty list means every file matches its recorded hash."""
    directory = Path(directory)
    manifest = json.loads((directory / name).read_text())
    problems = []
    for rel, digest in manifest["files"].items():
        p = directory / rel
        if not p.exists():
            problems.append(f"missing: {rel}")
        elif sha256_file(p) != digest:
            problems.append(f"hash mismatch: {rel}")
    return problems
