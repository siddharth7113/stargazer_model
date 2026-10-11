"""Tier 1 detection scorer.

Rules. Each tolerance is an explicit argument, so a run can change it and record it.

* null episode: not detected gives 1.0, detected gives 0.0 (false alarm).
* transit episode, not detected: 0.0 (miss).
* period credit: 1.0 at the true period, 0.5 at 1/2 or 2x, 0.25 at 1/3 or 3x, else 0.0.
  A match means ``|pred / (true * ratio) - 1| < period_rtol``.
* epoch ok: predicted ``t0`` within ``epoch_frac * duration`` of a true transit centre,
  modulo the true period.
* score = ``period_credit * (0.7 + 0.3 * epoch_ok)``, zero when period credit is zero.
* depth_ok is informational only: predicted depth within a factor of two of the truth.
"""

from __future__ import annotations

import numpy as np

from .types import Prediction

ALIASES = [(1.0, 1.0), (0.5, 0.5), (2.0, 0.5), (1.0 / 3.0, 0.25), (3.0, 0.25)]
SNR_BINS = [(0, 5), (5, 7), (7, 10), (10, 20), (20, 50), (50, np.inf)]
NTR_BINS = [(1, 1), (2, 2), (3, 5), (6, np.inf)]


def score_episode(
    pred: Prediction | dict, truth: dict, period_rtol: float = 0.01, epoch_frac: float = 0.5
) -> dict:
    """Score one prediction against one truth record.

    Parameters
    ----------
    pred : Prediction or dict
    truth : dict
        One entry of ``EpisodeSet.truth``.
    period_rtol : float
        Relative period tolerance.
    epoch_frac : float
        Epoch tolerance as a fraction of the transit duration.

    Returns
    -------
    dict
        ``kind, detected, period_credit, alias, epoch_ok, depth_ok, score, outcome``.
        ``outcome`` is one of ``true_null, false_alarm, miss, hit, alias, wrong_period``.
    """
    if isinstance(pred, dict):
        pred = Prediction.from_dict(pred)
    row = {
        "kind": truth["kind"],
        "detected": pred.detected,
        "period_credit": 0.0,
        "alias": None,
        "epoch_ok": False,
        "depth_ok": None,
        "score": 0.0,
        "outcome": None,
    }
    if truth["kind"] == "null":
        row["score"] = 0.0 if pred.detected else 1.0
        row["outcome"] = "false_alarm" if pred.detected else "true_null"
        return row
    if not pred.detected:
        row["outcome"] = "miss"
        return row
    tr = truth["transit"]
    period, t0, dur = tr["period"], tr["t0"], tr["duration"]
    credit, ratio_hit = 0.0, None
    for ratio, c in ALIASES:
        if abs(float(pred.period) / (period * ratio) - 1.0) < period_rtol:
            credit, ratio_hit = c, ratio
            break
    row["period_credit"], row["alias"] = credit, ratio_hit
    if credit > 0:
        phase = ((float(pred.t0) - t0 + 0.5 * period) % period) - 0.5 * period
        row["epoch_ok"] = bool(abs(phase) < epoch_frac * dur)
        row["score"] = credit * (0.7 + 0.3 * row["epoch_ok"])
        row["outcome"] = "hit" if ratio_hit == 1.0 else "alias"
    else:
        row["outcome"] = "wrong_period"
    if pred.depth is not None:
        row["depth_ok"] = bool(0.5 <= float(pred.depth) / tr["depth"] <= 2.0)
    return row


def _label(lo, hi) -> str:
    return f"{lo}-{'inf' if hi == np.inf else hi}"


def _completeness(rows: dict, keys: list[str]) -> float | None:
    return float(np.mean([rows[k]["period_credit"] > 0 for k in keys])) if keys else None


def summarize(rows: dict[str, dict], truth: dict) -> dict:
    """Aggregate per-episode rows into set-level metrics.

    Returns
    -------
    dict
        Mean score, false-alarm rate on nulls, exact and alias-inclusive completeness
        on injections, and completeness binned by expected SNR and by transit count.
    """
    nulls = [r for r in rows.values() if r["kind"] == "null"]
    trans = {k: r for k, r in rows.items() if r["kind"] == "transit"}
    summary = {
        "n_episodes": len(rows),
        "n_null": len(nulls),
        "n_transit": len(trans),
        "mean_score": float(np.mean([r["score"] for r in rows.values()])) if rows else None,
        "false_alarm_rate": float(np.mean([r["detected"] for r in nulls])) if nulls else None,
        "completeness_exact": (
            float(np.mean([r["outcome"] == "hit" for r in trans.values()])) if trans else None
        ),
        "completeness_with_aliases": _completeness(trans, list(trans)),
        "by_snr": {},
        "by_n_transits": {},
    }
    for lo, hi in SNR_BINS:
        ks = [k for k in trans if lo <= truth[k]["snr"] < hi]
        summary["by_snr"][_label(lo, hi)] = {"n": len(ks), "completeness": _completeness(trans, ks)}
    for lo, hi in NTR_BINS:
        ks = [k for k in trans if lo <= truth[k]["n_transits"] <= hi]
        summary["by_n_transits"][_label(lo, hi)] = {"n": len(ks), "completeness": _completeness(trans, ks)}
    return summary


def score_set(preds: dict[str, Prediction | dict], truth: dict, **tol) -> tuple[dict, dict]:
    """Score each episode in ``truth``. A missing prediction counts as not detected."""
    rows = {k: score_episode(preds.get(k, {"detected": False}), truth[k], **tol) for k in truth}
    return rows, summarize(rows, truth)
