"""Compare a prediction with the hidden truth of an episode and give a score.

The scorer is the reward function of the benchmark. It takes what a detection
system claimed about one light curve and what was in fact injected, and returns a
number between 0 and 1 with an outcome label. It has no dependency on files or the
network, so a training loop can call it on each rollout.

Rules. Each tolerance is an explicit argument, so a run can change it and record it.

* null episode: not detected gives 1.0, detected gives 0.0 (false alarm).
* transit episode, not detected: 0.0 (miss).
* period credit: 1.0 at the true period, 0.5 at 1/2 or 2x, 0.25 at 1/3 or 3x, else 0.0.
  A match means ``|pred / (true * ratio) - 1| < period_rtol``.
* epoch ok: predicted ``t0`` within ``epoch_frac * duration`` of a true transit centre,
  modulo the true period.
* score = ``period_credit * (0.7 + 0.3 * epoch_ok)``, zero when period credit is zero.
* depth_ok is informational only: predicted depth within a factor of two of the truth.
* single-event form: ``detected`` with no period and an event time ``t0``. The event is
  credited when ``t0`` passes the epoch check. It scores 1.0 if the episode has one
  transit in the data, else ``single_event_credit``.
"""

from __future__ import annotations

import numpy as np

from stargazer.types import Prediction

ALIASES = [(1.0, 1.0), (0.5, 0.5), (2.0, 0.5), (1.0 / 3.0, 0.25), (3.0, 0.25)]
SNR_BINS = [(0, 5), (5, 7), (7, 10), (10, 20), (20, 50), (50, np.inf)]
NTR_BINS = [(1, 1), (2, 2), (3, 5), (6, np.inf)]


def score_episode(
    pred: Prediction | dict,
    truth: dict,
    period_rtol: float = 0.01,
    epoch_frac: float = 0.5,
    single_event_credit: float = 0.5,
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
    single_event_credit : float
        Score of a credited single-event submission on an episode with two or more
        transits in the data.

    Returns
    -------
    dict
        ``kind, detected, period_credit, alias, epoch_ok, event_ok, depth_ok, score,
        outcome``. ``outcome`` is one of ``true_null, false_alarm, miss, hit, alias,
        wrong_period, event_hit, wrong_event``.

    Notes
    -----
    Period credit: for each ``(ratio, credit)`` in :data:`ALIASES`, in order, the
    predicted period matches if ``|P_pred / (P_true * ratio) - 1| < period_rtol``.
    The first match wins. The ratios 1/2 and 2 cover the common BLS aliases where
    every second transit is missed or two unrelated dips are paired. The ratios 1/3
    and 3 cover the next harmonic.

    Epoch check: the predicted epoch is folded on the true period,

    ``phase = ((t0_pred - t0_true + P/2) mod P) - P/2``

    which lies in ``[-P/2, P/2)``. The check passes if ``|phase| < epoch_frac * T14``.
    The fold uses the true period even for an alias, because an aliased detector
    must still point at a real transit.

    Score: ``period_credit * (0.7 + 0.3 * epoch_ok)``. The weights make a correct
    period with a wrong epoch worth more than an alias with a correct epoch.

    An injected episode whose transits all fall in data gaps, or after the end of
    the data, has ``n_transits == 0``. No detector can find it. The scorer treats it
    as a null episode: "none" scores 1.0 and a detection scores 0.0.

    Single-event form: a prediction with ``detected`` True and no period claims one
    transit-like event at time ``t0`` without a period. A planet with one transit in
    the data has no measurable period, so this is the complete answer for it. The
    event passes the same epoch check as a periodic prediction and sets ``event_ok``.
    A credited event scores 1.0 if ``n_transits == 1`` and ``single_event_credit``
    otherwise, because with two or more transits in the data the period was
    measurable and the event answer is incomplete, like an alias. An event that
    fails the check has outcome ``wrong_event`` and scores 0.0. On a null episode the
    form is a false alarm like any detection.
    """
    if isinstance(pred, dict):
        pred = Prediction.from_dict(pred)
    row = {
        "kind": truth["kind"],
        "detected": pred.detected,
        "period_credit": 0.0,
        "alias": None,
        "epoch_ok": False,
        "event_ok": None,
        "depth_ok": None,
        "score": 0.0,
        "outcome": None,
    }
    if truth["kind"] == "null" or truth.get("n_transits", 1) == 0:
        row["score"] = 0.0 if pred.detected else 1.0
        row["outcome"] = "false_alarm" if pred.detected else "true_null"
        return row
    if not pred.detected:
        row["outcome"] = "miss"
        return row
    tr = truth["transit"]
    period, t0, dur = tr["period"], tr["t0"], tr["duration"]
    if pred.depth is not None:
        row["depth_ok"] = bool(0.5 <= float(pred.depth) / tr["depth"] <= 2.0)
    if pred.period is None:
        row["event_ok"] = pred.t0 is not None and _epoch_ok(float(pred.t0), t0, period, dur, epoch_frac)
        if row["event_ok"]:
            row["score"] = 1.0 if truth.get("n_transits", 1) == 1 else single_event_credit
            row["outcome"] = "event_hit"
        else:
            row["outcome"] = "wrong_event"
        return row
    credit, ratio_hit = 0.0, None
    for ratio, credit_value in ALIASES:
        if abs(float(pred.period) / (period * ratio) - 1.0) < period_rtol:
            credit, ratio_hit = credit_value, ratio
            break
    row["period_credit"], row["alias"] = credit, ratio_hit
    if credit > 0:
        row["epoch_ok"] = _epoch_ok(float(pred.t0), t0, period, dur, epoch_frac)
        row["score"] = credit * (0.7 + 0.3 * row["epoch_ok"])
        row["outcome"] = "hit" if ratio_hit == 1.0 else "alias"
    else:
        row["outcome"] = "wrong_period"
    return row


def _epoch_ok(t0_pred: float, t0_true: float, period: float, duration: float, epoch_frac: float) -> bool:
    """Return True if ``t0_pred``, folded on the true period, is within the tolerance of a transit centre."""
    phase = ((t0_pred - t0_true + 0.5 * period) % period) - 0.5 * period
    return bool(abs(phase) < epoch_frac * duration)


def _label(lo, hi) -> str:
    return f"{lo}-{'inf' if hi == np.inf else hi}"


def _completeness(rows: dict, keys: list[str]) -> float | None:
    """Return the fraction of ``keys`` with a score above zero: a hit, an alias or a credited event."""
    return float(np.mean([rows[key]["score"] > 0 for key in keys])) if keys else None


def _is_exact(row: dict) -> bool:
    """Return True for a hit at the true period, or a credited event on a one-transit episode."""
    return row["outcome"] == "hit" or (row["outcome"] == "event_hit" and row["score"] == 1.0)


def summarize(rows: dict[str, dict], truth: dict) -> dict:
    """Aggregate per-episode rows into set-level metrics.

    Returns
    -------
    dict
        Mean score, false-alarm rate on nulls, exact and alias-inclusive completeness
        on injections, and completeness binned by expected SNR and by transit count.
        Exact completeness counts a hit at the true period and a credited single
        event on a one-transit episode. Alias-inclusive completeness counts every
        episode with a score above zero.
    """

    def _is_null(episode_id: str) -> bool:
        return truth[episode_id]["kind"] == "null" or truth[episode_id].get("n_transits", 1) == 0

    nulls = [rows[episode_id] for episode_id in rows if _is_null(episode_id)]
    trans = {episode_id: rows[episode_id] for episode_id in rows if not _is_null(episode_id)}
    summary = {
        "n_episodes": len(rows),
        "n_null": len(nulls),
        "n_transit": len(trans),
        "mean_score": float(np.mean([row["score"] for row in rows.values()])) if rows else None,
        "false_alarm_rate": float(np.mean([row["detected"] for row in nulls])) if nulls else None,
        "completeness_exact": (float(np.mean([_is_exact(row) for row in trans.values()])) if trans else None),
        "completeness_with_aliases": _completeness(trans, list(trans)),
        "by_snr": {},
        "by_n_transits": {},
    }
    for lo, hi in SNR_BINS:
        keys = [episode_id for episode_id in trans if lo <= truth[episode_id]["snr"] < hi]
        summary["by_snr"][_label(lo, hi)] = {"n": len(keys), "completeness": _completeness(trans, keys)}
    for lo, hi in NTR_BINS:
        keys = [episode_id for episode_id in trans if lo <= truth[episode_id]["n_transits"] <= hi]
        summary["by_n_transits"][_label(lo, hi)] = {
            "n": len(keys),
            "completeness": _completeness(trans, keys),
        }
    return summary


def score_set(preds: dict[str, Prediction | dict], truth: dict, **tol) -> tuple[dict, dict]:
    """Score each episode in ``truth``. A missing prediction counts as not detected."""
    rows = {
        episode_id: score_episode(preds.get(episode_id, {"detected": False}), truth[episode_id], **tol)
        for episode_id in truth
    }
    return rows, summarize(rows, truth)
