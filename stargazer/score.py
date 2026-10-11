"""Tier 1 detection scorer.

Prediction format (dict): {"detected": bool, "period": float, "t0": float,
                           "depth": float | None, "duration": float | None, ...}
Times are in the episode's relative time base.

Rules (all tolerances are explicit arguments so they can be changed and recorded):
  null episode:    "detected" False -> 1.0, True -> 0.0 (false alarm)
  transit episode: not detected -> 0.0 (miss)
                   period credit: 1.0 at the true period, 0.5 at 1/2 or 2x, 0.25 at 1/3 or 3x,
                                  else 0.0; a match is |pred/(true*ratio) - 1| < period_rtol
                   epoch ok:      predicted t0 within epoch_frac * duration of a true transit
                                  centre, modulo the TRUE period
                   score = period_credit * (0.7 + 0.3 * epoch_ok)   (0 if period_credit == 0)
  depth_ok (informational, not scored): predicted depth within a factor 2 of the true depth.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .provenance import git_commit, utc_now

ALIASES = [(1.0, 1.0), (0.5, 0.5), (2.0, 0.5), (1.0 / 3.0, 0.25), (3.0, 0.25)]

SNR_BINS = [(0, 5), (5, 7), (7, 10), (10, 20), (20, 50), (50, np.inf)]
NTR_BINS = [(1, 1), (2, 2), (3, 5), (6, np.inf)]


def score_episode(pred: dict, truth: dict, period_rtol: float = 0.01, epoch_frac: float = 0.5) -> dict:
    detected = bool(pred.get("detected", False))
    row = {"kind": truth["kind"], "detected": detected, "period_credit": 0.0,
           "epoch_ok": False, "depth_ok": None, "score": 0.0, "outcome": None}
    if truth["kind"] == "null":
        row["score"] = 0.0 if detected else 1.0
        row["outcome"] = "false_alarm" if detected else "true_null"
        return row
    if not detected:
        row["outcome"] = "miss"
        return row
    tr = truth["transit"]
    P, t0, dur = tr["period"], tr["t0"], tr["duration"]
    pp = float(pred["period"])
    credit, ratio_hit = 0.0, None
    for ratio, c in ALIASES:
        if abs(pp / (P * ratio) - 1.0) < period_rtol:
            credit, ratio_hit = c, ratio
            break
    row["period_credit"] = credit
    row["alias"] = ratio_hit
    if credit > 0:
        pt0 = float(pred["t0"])
        phase = ((pt0 - t0 + 0.5 * P) % P) - 0.5 * P
        row["epoch_ok"] = bool(abs(phase) < epoch_frac * dur)
        row["score"] = credit * (0.7 + 0.3 * row["epoch_ok"])
        row["outcome"] = "hit" if ratio_hit == 1.0 else "alias"
    else:
        row["outcome"] = "wrong_period"
    if pred.get("depth") is not None:
        row["depth_ok"] = bool(0.5 <= float(pred["depth"]) / tr["depth"] <= 2.0)
    return row


def _bin_label(lo, hi):
    return f"{lo}-{'inf' if hi == np.inf else hi}"


def summarize(rows: dict[str, dict], truth: dict) -> dict:
    nulls = [r for r in rows.values() if r["kind"] == "null"]
    trans = {k: r for k, r in rows.items() if r["kind"] == "transit"}
    summary = {
        "n_episodes": len(rows),
        "n_null": len(nulls),
        "n_transit": len(trans),
        "mean_score": float(np.mean([r["score"] for r in rows.values()])) if rows else None,
        "false_alarm_rate": float(np.mean([r["detected"] for r in nulls])) if nulls else None,
        "completeness_exact": float(np.mean([r["outcome"] == "hit" for r in trans.values()])) if trans else None,
        "completeness_with_aliases": float(np.mean([r["period_credit"] > 0 for r in trans.values()])) if trans else None,
        "by_snr": {},
        "by_n_transits": {},
    }
    for lo, hi in SNR_BINS:
        ks = [k for k in trans if lo <= truth[k]["snr"] < hi]
        summary["by_snr"][_bin_label(lo, hi)] = {
            "n": len(ks),
            "completeness": float(np.mean([trans[k]["period_credit"] > 0 for k in ks])) if ks else None,
        }
    for lo, hi in NTR_BINS:
        ks = [k for k in trans if lo <= truth[k]["n_transits"] <= hi]
        summary["by_n_transits"][_bin_label(lo, hi)] = {
            "n": len(ks),
            "completeness": float(np.mean([trans[k]["period_credit"] > 0 for k in ks])) if ks else None,
        }
    return summary


def score_set(preds: dict[str, dict], truth: dict, **tol) -> tuple[dict, dict]:
    rows = {k: score_episode(preds.get(k, {"detected": False}), truth[k], **tol) for k in truth}
    return rows, summarize(rows, truth)


def write_results(path: Path, *, system: str, config: dict, episode_set: str,
                  episode_set_hash: str, preds: dict, rows: dict, summary: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "created": utc_now(),
        "git_commit": git_commit(),
        "system": system,
        "config": config,
        "episode_set": str(episode_set),
        "episode_set_hash": episode_set_hash,
        "summary": summary,
        "episodes": {k: {"pred": preds.get(k), "score": rows[k]} for k in sorted(rows)},
    }
    path.write_text(json.dumps(doc, indent=1, sort_keys=True, default=_json_default))
    return path


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))
