"""Unit tests for stargazer.episodes: building, reading and verifying an episode set on a fake cache."""

import json

import numpy as np
import pytest

from stargazer.episodes import BuildConfig, EpisodeSet


def _build(fake_cache, tmp_path, name="a", **kwargs):
    """Build a small set in ``tmp_path/name`` with 6 episodes and seed 1 unless overridden."""
    cache, targets, targets_path = fake_cache
    cfg = BuildConfig(**{"n_episodes": 6, "seed": 1, **kwargs})
    return EpisodeSet.build(targets, cache, tmp_path / name, cfg, targets_path)


def _episode_hashes(episodes):
    """Hashes of the episode and truth files, excluding config.json which carries a timestamp."""
    files = json.loads((episodes.root / "manifest.json").read_text())["files"]
    return {key: value for key, value in files.items() if key != "config.json"}


def test_same_seed_gives_identical_files(fake_cache, tmp_path):
    """Two builds with the same inputs and seed produce identical episode and truth files."""
    first = _build(fake_cache, tmp_path, "a", seed=7)
    second = _build(fake_cache, tmp_path, "b", seed=7)
    assert _episode_hashes(first) == _episode_hashes(second)


def test_different_seed_gives_different_files(fake_cache, tmp_path):
    """The seed must change the draws, or the determinism test above would be vacuous."""
    first = _build(fake_cache, tmp_path, "a", seed=1)
    second = _build(fake_cache, tmp_path, "b", seed=2)
    assert _episode_hashes(first) != _episode_hashes(second)


def test_episode_file_holds_only_arrays_in_relative_time(fake_cache, tmp_path):
    """The detector-facing file carries nothing that identifies the star, and time starts at zero."""
    episodes = _build(fake_cache, tmp_path)
    with np.load(episodes.root / "episodes" / "ep_0000.npz") as archive:
        assert set(archive.files) == {"time", "flux", "flux_err"}
        assert archive["time"].min() == 0.0


def test_truth_record_fields(fake_cache, tmp_path):
    """A transit record names the host, stores the SNR and transit count, and the epoch in relative time."""
    truth = _build(fake_cache, tmp_path, n_episodes=5, null_frac=0.0).truth
    assert len(truth) == 5
    for record in truth.values():
        assert record["kind"] == "transit" and record["tic"] in (1000, 1001, 1002)
        assert record["snr"] > 0 and record["n_transits"] >= 0
        assert 0 <= record["transit"]["t0"] < record["transit"]["period"]


def test_null_fraction_one_gives_only_nulls(fake_cache, tmp_path):
    """``null_frac`` of 1.0 injects nothing; a null record has no transit key."""
    truth = _build(fake_cache, tmp_path, n_episodes=10, null_frac=1.0).truth
    assert all(record["kind"] == "null" and "transit" not in record for record in truth.values())


def test_injected_flux_differs_from_cache(fake_cache, tmp_path):
    """A transit episode's flux is not the cached flux; a null episode's flux is exactly the cached flux."""
    cache, _, _ = fake_cache
    episodes = _build(fake_cache, tmp_path, n_episodes=10, null_frac=0.5, depth_range=(0.01, 0.02))
    for episode_id, record in episodes.truth.items():
        _, flux, _ = episodes.load(episode_id)
        cached = cache.load(record["tic"], record["sector"]).flux
        assert np.array_equal(flux, cached) == (record["kind"] == "null")


def test_config_records_seed_git_and_targets_hash(fake_cache, tmp_path):
    """config.json holds what a reader needs to rebuild the set."""
    config = _build(fake_cache, tmp_path, seed=7).config
    assert config["build"]["seed"] == 7 and config["build"]["n_episodes"] == 6
    assert len(config["targets_sha256"]) == 64 and config["git_commit"]


def test_ids_len_and_load(fake_cache, tmp_path):
    """Episode ids are zero-padded and sorted; ``load`` returns three arrays of one length."""
    episodes = _build(fake_cache, tmp_path, n_episodes=4)
    assert episodes.ids == ["ep_0000", "ep_0001", "ep_0002", "ep_0003"] and len(episodes) == 4
    time, flux, flux_err = episodes.load("ep_0002")
    assert time.shape == flux.shape == flux_err.shape and time.size > 1000


def test_verify_passes_on_fresh_set(fake_cache, tmp_path):
    """A set that has just been built verifies cleanly."""
    assert _build(fake_cache, tmp_path).verify() == []


def test_verify_detects_changed_file(fake_cache, tmp_path):
    """A 0.01 percent change to one episode is caught and named."""
    episodes = _build(fake_cache, tmp_path, n_episodes=4)
    time, flux, flux_err = episodes.load("ep_0001")
    np.savez_compressed(
        episodes.root / "episodes" / "ep_0001.npz", time=time, flux=flux * 1.0001, flux_err=flux_err
    )
    assert episodes.verify() == ["hash mismatch: episodes/ep_0001.npz"]


def test_hash_is_the_manifest_digest(fake_cache, tmp_path):
    """The set hash is the SHA-256 of manifest.json, and reopening the directory gives the same hash."""
    episodes = _build(fake_cache, tmp_path)
    import hashlib

    expected = hashlib.sha256((episodes.root / "manifest.json").read_bytes()).hexdigest()
    assert episodes.hash == expected == EpisodeSet(episodes.root).hash


def test_open_requires_manifest(tmp_path):
    """A directory without a manifest is not an episode set."""
    with pytest.raises(FileNotFoundError):
        EpisodeSet(tmp_path)


def test_build_skips_hosts_with_nan_parameters(fake_cache, tmp_path):
    """A host with NaN logg is left out of every episode and listed in config.json."""
    cache, targets, _ = fake_cache
    targets["targets"][0]["logg"] = float("nan")
    episodes = EpisodeSet.build(targets, cache, tmp_path / "a", BuildConfig(n_episodes=6, seed=1))
    assert episodes.config["hosts_skipped_missing_params"] == [1000]
    assert all(record["tic"] != 1000 for record in episodes.truth.values())


def test_build_without_cached_hosts_raises(fake_cache, tmp_path):
    """A targets file whose stars are not in the cache cannot be built, and the error says to fetch."""
    cache, _, _ = fake_cache
    targets = {"targets": [{"tic": 999, "sector": 98, "logg": 4.4, "radius_rsun": 1.0}]}
    with pytest.raises(RuntimeError, match="fetch"):
        EpisodeSet.build(targets, cache, tmp_path / "a", BuildConfig(n_episodes=1, seed=1))
