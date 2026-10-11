"""Unit tests for stargazer.calibrate: re-scoring a results file at other thresholds."""

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


def test_rescore_ignores_recorded_detected_flag():
    """Re-scoring must use the statistic and threshold only, not the flag the original run stored."""
    assert rescore_at_threshold(_results(), TRUTH, 100.0)["false_alarm_rate"] == 0.0


def test_rescore_at_threshold_between_null_statistics():
    """At 7.5 the nulls at 5, 6, 7 are quiet and the one at 8 fires: false-alarm rate 1/4."""
    summary = rescore_at_threshold(_results(), TRUTH, 7.5)
    assert summary["false_alarm_rate"] == 0.25 and summary["completeness_exact"] == 0.5


def test_sweep_false_alarm_rate_is_non_increasing():
    """Raising the threshold can only remove detections, so the false-alarm rate cannot rise."""
    rates = [row["false_alarm_rate"] for row in threshold_sweep(_results(), TRUTH, [4, 6, 8, 10, 13])]
    assert rates == [1.0, 0.75, 0.25, 0.0, 0.0]


def test_sweep_row_holds_all_four_metrics():
    """The CLI table prints these four columns; a missing key would break it."""
    row = threshold_sweep(_results(), TRUTH, [6])[0]
    assert set(row) == {
        "threshold",
        "false_alarm_rate",
        "completeness_exact",
        "completeness_with_aliases",
        "mean_score",
    }


def test_choose_threshold_returns_exact_null_statistic():
    """The chosen threshold is one of the null statistics, so the stated rate is exact for the set."""
    assert choose_threshold(_results(), TRUTH, max_false_alarm=0.25) == 8.0
    assert choose_threshold(_results(), TRUTH, max_false_alarm=0.5) == 7.0


def test_choose_threshold_zero_false_alarms_exceeds_max_null():
    """A target of zero needs a threshold above the loudest null."""
    assert choose_threshold(_results(), TRUTH, max_false_alarm=0.0) > 8.0


def test_choose_threshold_without_nulls_is_infinite():
    """With no null episodes the false-alarm rate is undefined, and the function must say so, not guess."""
    only_transits = {key: value for key, value in TRUTH.items() if value["kind"] == "transit"}
    results = Results("x", {}, "s", "h", {key: PREDS[key] for key in only_transits}, {}, {})
    assert choose_threshold(results, only_transits) == float("inf")
