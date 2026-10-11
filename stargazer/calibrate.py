"""Choose a detection threshold from recorded data.

A detector such as the BLS baseline computes a statistic for each light curve and
reports a detection when the statistic passes a threshold. The right threshold
depends on the accepted false-alarm rate. The baseline records the signal detection
efficiency (SDE) of each episode in each prediction. This module takes such a results
file and re-scores the same predictions at other thresholds without a new run. The
choice of threshold is then a function of the results file, which the LOGBOOK cites.
"""

from __future__ import annotations

import numpy as np

from stargazer.evaluate import Results
from stargazer.score import score_set


def rescore_at_threshold(results: Results, truth: dict, threshold: float, stat: str = "sde") -> dict:
    """Re-score a results file with ``detected = stat >= threshold``.

    Parameters
    ----------
    results : Results
        A run whose predictions record ``stat``.
    truth : dict
        The truth of the same episode set.
    threshold : float
    stat : str
        Prediction key that holds the detection statistic.

    Returns
    -------
    dict
        The summary from :func:`stargazer.score.summarize`.
    """
    preds = {}
    for episode_id, pred in results.preds.items():
        rescored = dict(pred)
        rescored["detected"] = bool(rescored.get(stat, -np.inf) >= threshold)
        preds[episode_id] = rescored
    tol = {
        episode_id: results.config[episode_id]
        for episode_id in ("period_rtol", "epoch_frac")
        if episode_id in results.config
    }
    _, summary = score_set(preds, {episode_id: truth[episode_id] for episode_id in preds}, **tol)
    return summary


def threshold_sweep(results: Results, truth: dict, thresholds: list[float], stat: str = "sde") -> list[dict]:
    """Return the summary at each threshold.

    Each row holds ``threshold``, ``false_alarm_rate``, ``completeness_exact`` and
    ``completeness_with_aliases``.
    """
    rows = []
    for threshold in thresholds:
        summary = rescore_at_threshold(results, truth, threshold, stat)
        rows.append(
            {
                "threshold": float(threshold),
                "false_alarm_rate": summary["false_alarm_rate"],
                "completeness_exact": summary["completeness_exact"],
                "completeness_with_aliases": summary["completeness_with_aliases"],
                "mean_score": summary["mean_score"],
            }
        )
    return rows


def choose_threshold(
    results: Results, truth: dict, max_false_alarm: float = 0.05, stat: str = "sde"
) -> float:
    """Return the lowest threshold with a false-alarm rate at or below ``max_false_alarm``.

    The candidate thresholds are the null-episode statistics themselves, so the answer
    is exact for the given set. If no null episode exists, return positive infinity.

    Notes
    -----
    Sort the :math:`n` null statistics in ascending order. For the :math:`i`-th value as
    the threshold, the nulls at or above it number :math:`n - i`, so the false-alarm rate
    is :math:`(n - i) / n`. Return the first value where this rate is at or below
    ``max_false_alarm``. If no value qualifies, return a threshold just above the
    largest null statistic, which gives a false-alarm rate of zero on this set.

    With :math:`n` null episodes the smallest non-zero rate the set can resolve is
    :math:`1 / n`. A target of 0.05 needs at least 20 null episodes to mean anything.
    """
    null_stats = sorted(
        float(pred.get(stat, -np.inf))
        for episode_id, pred in results.preds.items()
        if truth[episode_id]["kind"] == "null"
    )
    if not null_stats:
        return float("inf")
    n_null = len(null_stats)
    for index, threshold in enumerate(null_stats):
        false_alarms = n_null - index  # nulls with stat >= thr
        if false_alarms / n_null <= max_false_alarm:
            return threshold
    return float(null_stats[-1]) + 1e-9
