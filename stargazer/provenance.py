"""Make each output traceable to its inputs.

The benchmark claims that each number is reproducible. These classes supply the
evidence. :class:`Manifest` records and verifies the SHA-256 of each file in a
directory. :class:`GitState` records the commit that produced a file, so a results file can
name the exact code behind its numbers. :func:`utc_timestamp` gives the time.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path


def utc_timestamp() -> str:
    """Return the current UTC time as an ISO 8601 string with second precision."""
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Digest:
    """A SHA-256 digest as a hex string.

    Parameters
    ----------
    hex : str
        64 lower-case hex characters.

    Examples
    --------
    >>> Digest.of_bytes(b"").hex[:16]
    'e3b0c44298fc1c14'
    """

    hex: str

    @classmethod
    def of_file(cls, path: Path) -> Digest:
        """Hash a file with the standard library's streaming digest."""
        with open(path, "rb") as file:
            return cls(hashlib.file_digest(file, "sha256").hexdigest())

    @classmethod
    def of_bytes(cls, data: bytes) -> Digest:
        """Hash a bytes object."""
        return cls(hashlib.sha256(data).hexdigest())

    def __str__(self) -> str:
        return self.hex


@dataclass
class Manifest:
    """The SHA-256 of each file in a directory, with the time it was recorded.

    A manifest lets any reader confirm that a directory of data files is the one a
    results file refers to. :meth:`of_directory` builds one. :meth:`write` saves it
    as ``manifest.json`` inside the directory. :meth:`read` opens a saved one.
    :meth:`verify` recomputes each hash and reports each mismatch.

    Parameters
    ----------
    files : dict of str to str
        Path relative to the directory, mapped to its SHA-256 hex digest.
    created : str
        UTC time the hashes were computed, ISO 8601.
    """

    FILENAME = "manifest.json"

    files: dict[str, str] = field(default_factory=dict)
    created: str = ""

    @classmethod
    def of_directory(cls, directory: Path, pattern: str = "**/*") -> Manifest:
        """Hash each file under a directory. The manifest file itself is skipped.

        Parameters
        ----------
        directory : Path
        pattern : str
            Glob relative to ``directory``.
        """
        directory = Path(directory)
        files = {
            str(path.relative_to(directory)): Digest.of_file(path).hex
            for path in sorted(directory.glob(pattern))
            if path.is_file() and path.name != cls.FILENAME
        }
        return cls(files=files, created=utc_timestamp())

    @classmethod
    def read(cls, directory: Path) -> Manifest:
        """Open the manifest saved in a directory."""
        data = json.loads((Path(directory) / cls.FILENAME).read_text())
        return cls(files=data["files"], created=data["created"])

    def write(self, directory: Path) -> Path:
        """Save as ``manifest.json`` in the directory. Return the path."""
        out = Path(directory) / self.FILENAME
        out.write_text(json.dumps({"created": self.created, "files": self.files}, indent=2, sort_keys=True))
        return out

    def verify(self, directory: Path) -> list[str]:
        """Recompute each hash against the files in ``directory``.

        Returns
        -------
        list of str
            One line for each missing file or hash mismatch. Empty means intact.
        """
        directory = Path(directory)
        problems = []
        for rel, digest in self.files.items():
            path = directory / rel
            if not path.exists():
                problems.append(f"missing: {rel}")
            elif Digest.of_file(path).hex != digest:
                problems.append(f"hash mismatch: {rel}")
        return problems

    @staticmethod
    def digest_of(directory: Path) -> Digest:
        """Return the digest of the saved manifest file. This identifies the directory."""
        return Digest.of_file(Path(directory) / Manifest.FILENAME)


@dataclass(frozen=True)
class GitState:
    """The git commit of the working tree, and whether the tree had uncommitted changes.

    Parameters
    ----------
    commit : str or None
        Full SHA of HEAD. None if the directory is not a git repository.
    dirty : bool
        True if ``git status --porcelain`` printed anything.

    Examples
    --------
    >>> str(GitState(commit="abc123", dirty=True))
    'abc123-dirty'
    """

    commit: str | None
    dirty: bool = False

    @classmethod
    def current(cls, cwd: Path | None = None) -> GitState:
        """Read the state of the repository that contains ``cwd``."""
        try:
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=cwd
            )
            status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=cwd)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return cls(commit=None)
        return cls(commit=head.stdout.strip(), dirty=bool(status.stdout.strip()))

    def __str__(self) -> str:
        if self.commit is None:
            return "no-git"
        return self.commit + ("-dirty" if self.dirty else "")
