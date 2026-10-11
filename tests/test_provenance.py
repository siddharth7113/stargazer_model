from stargazer.provenance import Digest, GitState, Manifest, utc_timestamp


def test_digest_matches_hashlib(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"star gazer")
    assert Digest.of_file(p) == Digest.of_bytes(b"star gazer")
    assert len(Digest.of_bytes(b"").hex) == 64


def test_manifest_round_trip_and_verify(tmp_path):
    (tmp_path / "x.txt").write_text("1")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "y.txt").write_text("2")
    m = Manifest.of_directory(tmp_path)
    assert set(m.files) == {"x.txt", "sub/y.txt"}
    m.write(tmp_path)
    assert Manifest.read(tmp_path).files == m.files
    assert m.verify(tmp_path) == []
    (tmp_path / "x.txt").write_text("changed")
    (tmp_path / "sub" / "y.txt").unlink()
    assert sorted(m.verify(tmp_path)) == ["hash mismatch: x.txt", "missing: sub/y.txt"]


def test_manifest_excludes_itself(tmp_path):
    (tmp_path / "x.txt").write_text("1")
    Manifest.of_directory(tmp_path).write(tmp_path)
    assert "manifest.json" not in Manifest.of_directory(tmp_path).files


def test_git_state_outside_repo(tmp_path):
    assert GitState.current(cwd=tmp_path).commit is None
    assert str(GitState(commit=None)) == "no-git"


def test_timestamp_is_utc_iso():
    assert utc_timestamp().endswith("+00:00")
