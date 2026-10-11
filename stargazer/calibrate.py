"""Choose a detection threshold from a results file.

The baseline records the signal detection efficiency (SDE) of each episode. This module
re-scores the same predictions at other thresholds without a new run. The choice of
threshold is then a function of the results file, which the LOGBOOK records.
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
    for k, p in results.preds.items():
        q = dict(p)
        q["detected"] = bool(q.get(stat, -np.inf) >= threshold)
        preds[k] = q
    tol = {k: results.config[k] for k in ("period_rtol", "epoch_frac") if k in results.config}
    _, summary = score_set(preds, {k: truth[k] for k in preds}, **tol)
    return summary


def threshold_sweep(results: Results, truth: dict, thresholds: list[float], stat: str = "sde") -> list[dict]:
    """Return the summary at each threshold.

    Each row holds ``threshold``, ``false_alarm_rate``, ``completeness_exact`` and
    ``completeness_with_aliases``.
    """
    rows = []
    for thr in thresholds:
        s = rescore_at_threshold(results, truth, thr, stat)
        rows.append(
            {
                "threshold": float(thr),
                "false_alarm_rate": s["false_alarm_rate"],
                "completeness_exact": s["completeness_exact"],
                "completeness_with_aliases": s["completeness_with_aliases"],
                "mean_score": s["mean_score"],
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
    Sort the ``n`` null statistics in ascending order. For the ``i``-th value as the
    threshold, the nulls at or above it number ``n - i``, so the false-alarm rate is
    ``(n - i) / n``. Return the first value where this rate is at or below
    ``max_false_alarm``. If no value qualifies, return a threshold just above the
    largest null statistic, which gives a false-alarm rate of zero on this set.

    With ``n`` null episodes the smallest non-zero rate the set can resolve is
    ``1 / n``. A target of 0.05 needs at least 20 null episodes to mean anything.
    """
    null_stats = sorted(
        float(p.get(stat, -np.inf)) for k, p in results.preds.items() if truth[k]["kind"] == "null"
    )
    if not null_stats:
        return float("inf")
    n = len(null_stats)
    for i, thr in enumerate(null_stats):
        false_alarms = n - i  # nulls with stat >= thr
        if false_alarms / n <= max_false_alarm:
            return thr
    return float(null_stats[-1]) + 1e-9
