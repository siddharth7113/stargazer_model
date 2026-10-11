import json

import numpy as np

from stargazer.episodes import build_episode_set, episode_set_hash, list_episodes, load_episode, load_truth
from stargazer.provenance import verify_manifest
from tests.conftest import synthetic_lc


def _fake_cache(tmp_path, star, n_stars=3):
    cache = tmp_path / "cache"
    cache.mkdir()
    targets = {"targets": []}
    for i in range(n_stars):
        tic, sector = 1000 + i, 98
        t, f, fe = synthetic_lc(seed=i)
        np.savez_compressed(cache / f"tic{tic}_s{sector:04d}.npz", time=t, flux=f, flux_err=fe,
                            meta=json.dumps({"tic": tic, "sector": sector}))
        targets["targets"].append({"tic": tic, "sector": sector, **star})
    tp = tmp_path / "targets.json"
    tp.write_text(json.dumps(targets))
    return cache, targets, tp


def test_build_is_deterministic_and_truth_is_separated(tmp_path, star):
    cache, targets, tp = _fake_cache(tmp_path, star)
    a = build_episode_set(targets, cache, tmp_path / "a", n=12, seed=7, targets_path=tp)
    b = build_episode_set(targets, cache, tmp_path / "b", n=12, seed=7, targets_path=tp)
    ma = json.loads((a / "manifest.json").read_text())["files"]
    mb = json.loads((b / "manifest.json").read_text())["files"]
    ep_a = {k: v for k, v in ma.items() if k.startswith("episodes/")}
    ep_b = {k: v for k, v in mb.items() if k.startswith("episodes/")}
    assert ep_a == ep_b  # same seed -> byte-identical episode files
    assert ma["truth.json"] == mb["truth.json"]
    assert verify_manifest(a) == []
    assert len(list_episodes(a)) == 12
    # agent-facing file carries nothing but time/flux/flux_err
    with np.load(a / "episodes" / "ep_0000.npz") as z:
        assert set(z.files) == {"time", "flux", "flux_err"}
        assert z["time"].min() == 0.0
    truth = load_truth(a)
    kinds = {v["kind"] for v in truth.values()}
    assert kinds <= {"null", "transit"}
    tr = next(v for v in truth.values() if v["kind"] == "transit")
    assert tr["snr"] > 0 and tr["n_transits"] >= 0 and "period" in tr["transit"]
    assert 0 <= tr["transit"]["t0"] < tr["transit"]["period"]


def test_different_seed_changes_set(tmp_path, star):
    cache, targets, tp = _fake_cache(tmp_path, star)
    a = build_episode_set(targets, cache, tmp_path / "a", n=6, seed=1)
    b = build_episode_set(targets, cache, tmp_path / "b", n=6, seed=2)
    assert load_truth(a) != load_truth(b)


def test_tampering_is_detected(tmp_path, star):
    cache, targets, tp = _fake_cache(tmp_path, star)
    a = build_episode_set(targets, cache, tmp_path / "a", n=4, seed=1)
    h = episode_set_hash(a)
    t, f, fe = load_episode(a, "ep_0001")
    np.savez_compressed(a / "episodes" / "ep_0001.npz", time=t, flux=f * 1.0001, flux_err=fe)
    assert verify_manifest(a) == ["hash mismatch: episodes/ep_0001.npz"]
    assert episode_set_hash(a) == h  # manifest unchanged, so the recorded hash still identifies the original set
