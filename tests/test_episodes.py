import json

import numpy as np

from stargazer.data import LightCurveCache
from stargazer.episodes import BuildConfig, EpisodeSet
from tests.conftest import synthetic_lc


def _fake_cache(tmp_path, star, n_stars=3):
    cache = LightCurveCache(tmp_path / "cache")
    cache.root.mkdir()
    targets = {"targets": []}
    for i in range(n_stars):
        tic, sector = 1000 + i, 98
        t, f, fe = synthetic_lc(seed=i)
        np.savez_compressed(
            cache.path(tic, sector),
            time=t,
            flux=f,
            flux_err=fe,
            meta=json.dumps({"tic": tic, "sector": sector}),
        )
        targets["targets"].append({"tic": tic, "sector": sector, **star})
    tp = tmp_path / "targets.json"
    tp.write_text(json.dumps(targets))
    return cache, targets, tp


def test_build_is_deterministic_and_truth_is_separated(tmp_path, star):
    cache, targets, tp = _fake_cache(tmp_path, star)
    cfg = BuildConfig(n_episodes=12, seed=7)
    a = EpisodeSet.build(targets, cache, tmp_path / "a", cfg, tp)
    b = EpisodeSet.build(targets, cache, tmp_path / "b", cfg, tp)
    ma = json.loads((a.root / "manifest.json").read_text())["files"]
    mb = json.loads((b.root / "manifest.json").read_text())["files"]
    assert {k: v for k, v in ma.items() if k.startswith("episodes/")} == {
        k: v for k, v in mb.items() if k.startswith("episodes/")
    }
    assert ma["truth.json"] == mb["truth.json"]
    assert a.verify() == []
    assert len(a) == 12
    with np.load(a.root / "episodes" / "ep_0000.npz") as z:
        assert set(z.files) == {"time", "flux", "flux_err"}
        assert z["time"].min() == 0.0
    truth = a.truth
    assert {v["kind"] for v in truth.values()} <= {"null", "transit"}
    tr = next(v for v in truth.values() if v["kind"] == "transit")
    assert tr["snr"] > 0 and tr["n_transits"] >= 0 and "period" in tr["transit"]
    assert 0 <= tr["transit"]["t0"] < tr["transit"]["period"]
    assert a.config["build"]["seed"] == 7 and a.config["targets_sha256"]


def test_different_seed_changes_set(tmp_path, star):
    cache, targets, _ = _fake_cache(tmp_path, star)
    a = EpisodeSet.build(targets, cache, tmp_path / "a", BuildConfig(n_episodes=6, seed=1))
    b = EpisodeSet.build(targets, cache, tmp_path / "b", BuildConfig(n_episodes=6, seed=2))
    assert a.truth != b.truth


def test_tampering_is_detected(tmp_path, star):
    cache, targets, _ = _fake_cache(tmp_path, star)
    a = EpisodeSet.build(targets, cache, tmp_path / "a", BuildConfig(n_episodes=4, seed=1))
    h = a.hash
    t, f, fe = a.load("ep_0001")
    np.savez_compressed(a.root / "episodes" / "ep_0001.npz", time=t, flux=f * 1.0001, flux_err=fe)
    assert a.verify() == ["hash mismatch: episodes/ep_0001.npz"]
    assert a.hash == h


def test_evaluate_round_trip(tmp_path, star):
    from stargazer.baseline import BLSDetector
    from stargazer.evaluate import Results, evaluate

    cache, targets, _ = _fake_cache(tmp_path, star)
    es = EpisodeSet.build(
        targets, cache, tmp_path / "a", BuildConfig(n_episodes=3, seed=3, depth_range=(0.01, 0.02))
    )
    res = evaluate(BLSDetector(), es)
    p = res.save(tmp_path / "r.json")
    back = Results.load(p)
    assert back.summary == res.summary
    assert back.episode_set_hash == es.hash
    assert back.config["sde_threshold"] == 7.0 and "period_rtol" in back.config
    assert set(back.preds) == set(es.ids)


def test_evaluate_records_detector_errors(tmp_path, star):
    from stargazer.evaluate import evaluate
    from stargazer.types import Prediction

    class Flaky:
        name = "flaky"
        config = {}
        calls = 0

        def detect(self, time, flux, flux_err):
            Flaky.calls += 1
            if Flaky.calls % 2 == 0:
                raise RuntimeError("boom")
            return Prediction(detected=False)

    cache, targets, _ = _fake_cache(tmp_path, star)
    es = EpisodeSet.build(targets, cache, tmp_path / "a", BuildConfig(n_episodes=4, seed=5))
    res = evaluate(Flaky(), es)
    assert set(res.preds) == set(es.ids)
    assert len(res.config["errors"]) == 2
    assert all("error" in res.preds[k] for k in res.config["errors"])
    import pytest

    with pytest.raises(RuntimeError):
        evaluate(Flaky(), es, record_errors=False)


def test_build_skips_hosts_with_nan_parameters(tmp_path, star):
    cache, targets, _ = _fake_cache(tmp_path, star)
    targets["targets"][0]["logg"] = float("nan")
    es = EpisodeSet.build(targets, cache, tmp_path / "a", BuildConfig(n_episodes=6, seed=1))
    assert es.config["hosts_skipped_missing_params"] == [1000]
    assert all(v["tic"] != 1000 for v in es.truth.values())
    assert all(np.isfinite(v["transit"]["a_rs"]) for v in es.truth.values() if v["kind"] == "transit")
