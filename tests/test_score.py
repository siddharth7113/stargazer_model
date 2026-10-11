import numpy as np

from stargazer.score import score_episode, summarize

T = {"kind": "transit", "snr": 12.0, "n_transits": 4,
     "transit": {"period": 3.0, "t0": 1.0, "duration": 0.1, "depth": 0.001}}
N = {"kind": "null"}


def test_exact_hit():
    r = score_episode({"detected": True, "period": 3.001, "t0": 7.0, "depth": 0.0012}, T)
    assert r["outcome"] == "hit" and r["score"] == 1.0 and r["epoch_ok"] and r["depth_ok"]


def test_epoch_wrong_gives_partial():
    r = score_episode({"detected": True, "period": 3.0, "t0": 2.5}, T)
    assert r["outcome"] == "hit" and r["score"] == 0.7 and not r["epoch_ok"]


def test_double_period_alias():
    r = score_episode({"detected": True, "period": 6.0, "t0": 4.0}, T)
    assert r["outcome"] == "alias" and r["alias"] == 2.0 and r["score"] == 0.5


def test_third_period_alias():
    r = score_episode({"detected": True, "period": 1.0, "t0": 1.0}, T)
    assert r["alias"] == 1.0 / 3.0 and r["score"] == 0.25


def test_wrong_period():
    r = score_episode({"detected": True, "period": 4.2, "t0": 1.0}, T)
    assert r["outcome"] == "wrong_period" and r["score"] == 0.0


def test_miss():
    r = score_episode({"detected": False}, T)
    assert r["outcome"] == "miss" and r["score"] == 0.0


def test_null_correct_and_false_alarm():
    assert score_episode({"detected": False}, N)["score"] == 1.0
    r = score_episode({"detected": True, "period": 2.0, "t0": 0.0}, N)
    assert r["outcome"] == "false_alarm" and r["score"] == 0.0


def test_period_tolerance_is_respected():
    assert score_episode({"detected": True, "period": 3.0 * 1.009, "t0": 1.0}, T)["period_credit"] == 1.0
    assert score_episode({"detected": True, "period": 3.0 * 1.011, "t0": 1.0}, T)["period_credit"] == 0.0


def test_depth_not_scored_only_flagged():
    r = score_episode({"detected": True, "period": 3.0, "t0": 1.0, "depth": 0.01}, T)
    assert r["score"] == 1.0 and r["depth_ok"] is False


def test_summarize_bins():
    truth = {"a": T, "b": {**T, "snr": 3.0, "n_transits": 1}, "c": N, "d": N}
    rows = {
        "a": score_episode({"detected": True, "period": 3.0, "t0": 1.0}, T),
        "b": score_episode({"detected": False}, truth["b"]),
        "c": score_episode({"detected": False}, N),
        "d": score_episode({"detected": True, "period": 1.0, "t0": 0.0}, N),
    }
    s = summarize(rows, truth)
    assert s["false_alarm_rate"] == 0.5
    assert s["completeness_exact"] == 0.5
    assert s["by_snr"]["10-20"]["completeness"] == 1.0
    assert s["by_snr"]["0-5"]["completeness"] == 0.0
    assert s["by_n_transits"]["1-1"]["n"] == 1
