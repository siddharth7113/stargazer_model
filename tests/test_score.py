"""Unit tests for stargazer.score: one test for each scoring rule, then the set-level summary."""

from stargazer.score import score_episode, summarize

TRANSIT = {
    "kind": "transit",
    "snr": 12.0,
    "n_transits": 4,
    "transit": {"period": 3.0, "t0": 1.0, "duration": 0.1, "depth": 0.001},
}
NULL = {"kind": "null"}


def test_exact_period_and_epoch_scores_one():
    """The best possible answer, period and epoch both right, scores 1.0 with outcome ``hit``."""
    row = score_episode({"detected": True, "period": 3.001, "t0": 7.0}, TRANSIT)
    assert row["outcome"] == "hit" and row["score"] == 1.0 and row["epoch_ok"]


def test_right_period_wrong_epoch_scores_point_seven():
    """Epoch is worth 30 percent; a correct period with a wrong epoch keeps the other 70."""
    row = score_episode({"detected": True, "period": 3.0, "t0": 2.5}, TRANSIT)
    assert row["outcome"] == "hit" and row["score"] == 0.7 and not row["epoch_ok"]


def test_double_period_alias_scores_half():
    """A detector that pairs every second transit gets half credit, labelled ``alias``."""
    row = score_episode({"detected": True, "period": 6.0, "t0": 4.0}, TRANSIT)
    assert row["outcome"] == "alias" and row["alias"] == 2.0 and row["score"] == 0.5


def test_third_period_alias_scores_quarter():
    """The 1/3 harmonic is a weaker alias and gets a quarter."""
    row = score_episode({"detected": True, "period": 1.0, "t0": 1.0}, TRANSIT)
    assert row["alias"] == 1.0 / 3.0 and row["score"] == 0.25


def test_wrong_period_scores_zero():
    """A detection at an unrelated period is as bad as a miss."""
    row = score_episode({"detected": True, "period": 4.2, "t0": 1.0}, TRANSIT)
    assert row["outcome"] == "wrong_period" and row["score"] == 0.0


def test_miss_scores_zero():
    """Not detecting an injected transit scores zero with outcome ``miss``."""
    row = score_episode({"detected": False}, TRANSIT)
    assert row["outcome"] == "miss" and row["score"] == 0.0


def test_null_correct_scores_one():
    """Saying "none" on a null episode is the right answer and scores 1.0."""
    row = score_episode({"detected": False}, NULL)
    assert row["outcome"] == "true_null" and row["score"] == 1.0


def test_null_false_alarm_scores_zero():
    """A detection on a null episode is a false alarm and scores zero."""
    row = score_episode({"detected": True, "period": 2.0, "t0": 0.0}, NULL)
    assert row["outcome"] == "false_alarm" and row["score"] == 0.0


def test_period_tolerance_boundary():
    """The relative tolerance is a strict upper bound: 0.9 percent passes, 1.1 percent fails at rtol 0.01."""
    assert (
        score_episode({"detected": True, "period": 3.0 * 1.009, "t0": 1.0}, TRANSIT)["period_credit"] == 1.0
    )
    assert (
        score_episode({"detected": True, "period": 3.0 * 1.011, "t0": 1.0}, TRANSIT)["period_credit"] == 0.0
    )


def test_period_tolerance_is_configurable():
    """A looser rtol accepts a period the default rejects, so a curriculum can tighten it later."""
    pred = {"detected": True, "period": 3.0 * 1.03, "t0": 1.0}
    assert score_episode(pred, TRANSIT)["period_credit"] == 0.0
    assert score_episode(pred, TRANSIT, period_rtol=0.05)["period_credit"] == 1.0


def test_epoch_is_folded_on_the_true_period():
    """An epoch several periods after the truth is still correct, because the fold is modulo the period."""
    assert score_episode({"detected": True, "period": 3.0, "t0": 1.0 + 3 * 7}, TRANSIT)["epoch_ok"]


def test_depth_is_flagged_but_not_scored():
    """A depth off by 10x sets ``depth_ok`` False and leaves the score unchanged."""
    row = score_episode({"detected": True, "period": 3.0, "t0": 1.0, "depth": 0.01}, TRANSIT)
    assert row["score"] == 1.0 and row["depth_ok"] is False


def test_injected_episode_with_no_transit_in_data_is_scored_as_null():
    """If every injected transit fell in a gap, "none" is correct and a detection is a false alarm."""
    gone = {**TRANSIT, "n_transits": 0, "snr": 0.0}
    assert score_episode({"detected": False}, gone)["outcome"] == "true_null"
    assert score_episode({"detected": True, "period": 3.0, "t0": 1.0}, gone)["outcome"] == "false_alarm"


def test_summarize_false_alarm_rate_and_completeness():
    """With one hit, one miss, one true null and one false alarm the rates are 0.5 and 0.5."""
    truth = {"a": TRANSIT, "b": {**TRANSIT, "snr": 3.0, "n_transits": 1}, "c": NULL, "d": NULL}
    rows = {
        "a": score_episode({"detected": True, "period": 3.0, "t0": 1.0}, TRANSIT),
        "b": score_episode({"detected": False}, truth["b"]),
        "c": score_episode({"detected": False}, NULL),
        "d": score_episode({"detected": True, "period": 1.0, "t0": 0.0}, NULL),
    }
    summary = summarize(rows, truth)
    assert summary["false_alarm_rate"] == 0.5 and summary["completeness_exact"] == 0.5
    assert summary["mean_score"] == 0.5


def test_summarize_bins_by_snr_and_transit_count():
    """Each injected episode lands in exactly one SNR bin and one transit-count bin."""
    truth = {"a": TRANSIT, "b": {**TRANSIT, "snr": 3.0, "n_transits": 1}}
    rows = {
        "a": score_episode({"detected": True, "period": 3.0, "t0": 1.0}, TRANSIT),
        "b": score_episode({"detected": False}, truth["b"]),
    }
    summary = summarize(rows, truth)
    assert summary["by_snr"]["10-20"] == {"n": 1, "completeness": 1.0}
    assert summary["by_snr"]["0-5"] == {"n": 1, "completeness": 0.0}
    assert summary["by_n_transits"]["1-1"] == {"n": 1, "completeness": 0.0}
    assert sum(bin_["n"] for bin_ in summary["by_snr"].values()) == 2


def test_summarize_counts_zero_transit_injection_as_null():
    """The set-level counts follow the same rule as the episode scorer."""
    gone = {**TRANSIT, "n_transits": 0, "snr": 0.0}
    summary = summarize({"g": score_episode({"detected": False}, gone)}, {"g": gone})
    assert summary["n_null"] == 1 and summary["n_transit"] == 0
