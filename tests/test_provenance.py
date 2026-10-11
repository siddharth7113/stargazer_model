"""Unit tests for stargazer.provenance: Digest, Manifest, GitState and the timestamp."""

import hashlib

from stargazer.provenance import Digest, GitState, Manifest, utc_timestamp


def test_digest_of_file_matches_digest_of_bytes(tmp_path):
    """The streaming file hash must equal the in-memory hash of the same bytes."""
    path = tmp_path / "a.bin"
    path.write_bytes(b"star gazer")
    assert Digest.of_file(path) == Digest.of_bytes(b"star gazer")


def test_digest_is_sha256_hex():
    """The digest is the standard SHA-256 hex string, so other tools can recompute it."""
    assert Digest.of_bytes(b"x").hex == hashlib.sha256(b"x").hexdigest()


def test_manifest_lists_files_recursively(tmp_path):
    """Episode sets have files in a subdirectory, so the manifest must walk into it."""
    (tmp_path / "x.txt").write_text("1")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "y.txt").write_text("2")
    assert set(Manifest.of_directory(tmp_path).files) == {"x.txt", "sub/y.txt"}


def test_manifest_excludes_itself(tmp_path):
    """A manifest that listed its own hash could never be consistent."""
    (tmp_path / "x.txt").write_text("1")
    Manifest.of_directory(tmp_path).write(tmp_path)
    assert "manifest.json" not in Manifest.of_directory(tmp_path).files


def test_manifest_write_read_round_trip(tmp_path):
    """What ``write`` saves, ``read`` returns unchanged."""
    (tmp_path / "x.txt").write_text("1")
    manifest = Manifest.of_directory(tmp_path)
    manifest.write(tmp_path)
    assert Manifest.read(tmp_path) == manifest


def test_manifest_verify_passes_on_intact_directory(tmp_path):
    """An unchanged directory verifies with no problems."""
    (tmp_path / "x.txt").write_text("1")
    assert Manifest.of_directory(tmp_path).verify(tmp_path) == []


def test_manifest_verify_reports_mismatch_and_missing(tmp_path):
    """A changed file and a deleted file are reported by name, so a reader knows what to re-fetch."""
    (tmp_path / "x.txt").write_text("1")
    (tmp_path / "y.txt").write_text("2")
    manifest = Manifest.of_directory(tmp_path)
    (tmp_path / "x.txt").write_text("changed")
    (tmp_path / "y.txt").unlink()
    assert sorted(manifest.verify(tmp_path)) == ["hash mismatch: x.txt", "missing: y.txt"]


def test_git_state_outside_repository_has_no_commit(tmp_path):
    """Running outside git must not crash; the commit is None and prints as ``no-git``."""
    state = GitState.current(cwd=tmp_path)
    assert state.commit is None and str(state) == "no-git"


def test_git_state_dirty_suffix():
    """The string form appends ``-dirty`` only when the tree had changes."""
    assert str(GitState(commit="abc", dirty=True)) == "abc-dirty"
    assert str(GitState(commit="abc", dirty=False)) == "abc"


def test_timestamp_is_utc_iso_8601():
    """Timestamps must be comparable across machines, so they carry the UTC offset."""
    stamp = utc_timestamp()
    assert stamp.endswith("+00:00") and len(stamp) == len("2026-10-11T12:00:00+00:00")
