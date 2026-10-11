from stargazer.calibrate import choose_threshold, rescore_at_threshold, threshold_sweep
from stargazer.evaluate import Results

TRUTH = {
    "n0": {"kind": "null"},
    "n1": {"kind": "null"},
    "n2": {"kind": "null"},
    "n3": {"kind": "null"},
    "t0": {
        "kind": "transit",
        "snr": 15,
        "n_transits": 4,
        "transit": {"period": 3.0, "t0": 1.0, "duration": 0.1, "depth": 1e-3},
    },
    "t1": {
        "kind": "transit",
        "snr": 4,
        "n_transits": 2,
        "transit": {"period": 5.0, "t0": 2.0, "duration": 0.1, "depth": 1e-3},
    },
}
PREDS = {
    "n0": {"detected": True, "period": 1.0, "t0": 0.0, "sde": 5.0},
    "n1": {"detected": True, "period": 1.0, "t0": 0.0, "sde": 6.0},
    "n2": {"detected": True, "period": 1.0, "t0": 0.0, "sde": 7.0},
    "n3": {"detected": True, "period": 1.0, "t0": 0.0, "sde": 8.0},
    "t0": {"detected": True, "period": 3.0, "t0": 1.0, "sde": 12.0},
    "t1": {"detected": True, "period": 5.0, "t0": 2.0, "sde": 6.5},
}


def _results():
    return Results("x", {"period_rtol": 0.01, "epoch_frac": 0.5}, "s", "h", PREDS, {}, {})


def test_rescore_moves_false_alarm_rate():
    r = _results()
    assert rescore_at_threshold(r, TRUTH, 4.0)["false_alarm_rate"] == 1.0
    s = rescore_at_threshold(r, TRUTH, 7.5)
    assert s["false_alarm_rate"] == 0.25 and s["completeness_exact"] == 0.5


def test_sweep_is_monotonic():
    rows = threshold_sweep(_results(), TRUTH, [4, 6, 8, 10, 13])
    far = [r["false_alarm_rate"] for r in rows]
    assert far == sorted(far, reverse=True)
    assert rows[-1]["completeness_exact"] == 0.0


def test_choose_threshold_exact_on_null_stats():
    r = _results()
    assert choose_threshold(r, TRUTH, max_false_alarm=0.25) == 8.0
    assert choose_threshold(r, TRUTH, max_false_alarm=0.5) == 7.0
    assert choose_threshold(r, TRUTH, max_false_alarm=0.0) > 8.0
