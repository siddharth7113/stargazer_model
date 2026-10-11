"""Helpers that make every output traceable: hashes, git commit, timestamps."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file. Read the file in 1 MiB chunks.

    Parameters
    ----------
    path : Path
        File to hash.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 of a bytes object."""
    return hashlib.sha256(data).hexdigest()


def git_commit() -> str | None:
    """Return the git HEAD commit. Append ``-dirty`` if the working tree has changes.

    Returns
    -------
    str or None
        None when not inside a git repository.
    """
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return None


def utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string with second precision."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def write_manifest(directory: Path, pattern: str = "**/*", name: str = "manifest.json") -> Path:
    """Write the SHA-256 of each file under a directory to a manifest file.

    Parameters
    ----------
    directory : Path
        Root to scan.
    pattern : str
        Glob pattern relative to ``directory``.
    name : str
        Manifest file name. The manifest does not list itself.

    Returns
    -------
    Path
        The manifest that was written.
    """
    directory = Path(directory)
    entries = {
        str(p.relative_to(directory)): sha256_file(p)
        for p in sorted(directory.glob(pattern))
        if p.is_file() and p.name != name
    }
    out = directory / name
    out.write_text(json.dumps({"created": utc_now(), "files": entries}, indent=2, sort_keys=True))
    return out


def verify_manifest(directory: Path, name: str = "manifest.json") -> list[str]:
    """Recompute hashes and compare with the manifest.

    Returns
    -------
    list of str
        Human-readable problems. An empty list means every file matches.
    """
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
