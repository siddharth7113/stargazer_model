"""Baseline detector: wotan biweight detrend -> bin -> astropy BoxLeastSquares -> SDE threshold.

Deliberately simple. Its job is to be the floor every other system must beat, and to show
where a classical search fails (single transits, shallow signals on active stars).
"""

from __future__ import annotations

import numpy as np

DEFAULT_CONFIG = {
    "detrend_window_days": 0.75,
    "bin_minutes": 10.0,
    "min_period_days": 0.5,
    "max_period_frac_of_span": 0.5,
    "durations_days": [0.03, 0.05, 0.08, 0.12, 0.18, 0.25],
    "frequency_factor": 2.0,
    "sde_threshold": 7.0,
}


def detrend(time, flux, window_days):
    from wotan import flatten

    flat = flatten(time, flux, method="biweight", window_length=window_days, return_trend=False)
    ok = np.isfinite(flat)
    return time[ok], flat[ok]


def bin_lc(time, flux, bin_minutes):
    if bin_minutes <= 0:
        return time, flux, np.full_like(flux, np.std(flux))
    w = bin_minutes / 1440.0
    idx = np.floor((time - time.min()) / w).astype(int)
    uniq, inv, counts = np.unique(idx, return_inverse=True, return_counts=True)
    t = np.bincount(inv, weights=time) / counts
    f = np.bincount(inv, weights=flux) / counts
    # per-bin error from the global point scatter, shrunk by sqrt(n)
    sig = np.std(flux) / np.sqrt(counts)
    return t, f, sig


def detect(time, flux, flux_err=None, config: dict | None = None) -> dict:
    from astropy.timeseries import BoxLeastSquares

    cfg = {**DEFAULT_CONFIG, **(config or {})}
    t, f = detrend(time, flux, cfg["detrend_window_days"])
    t, f, dy = bin_lc(t, f, cfg["bin_minutes"])
    span = t.max() - t.min()
    max_period = max(cfg["min_period_days"] * 1.5, span * cfg["max_period_frac_of_span"])
    bls = BoxLeastSquares(t, f, dy)
    durations = np.array(cfg["durations_days"])
    periods = bls.autoperiod(durations, minimum_period=cfg["min_period_days"],
                             maximum_period=max_period, frequency_factor=cfg["frequency_factor"])
    res = bls.power(periods, durations)
    power = np.asarray(res.power)
    i = int(np.nanargmax(power))
    sde = float((power[i] - np.nanmean(power)) / np.nanstd(power))
    best_period = float(res.period[i])
    best_t0 = float(res.transit_time[i])
    best_dur = float(res.duration[i])
    best_depth = float(res.depth[i])
    detected = bool(sde >= cfg["sde_threshold"])
    return {
        "detected": detected,
        "period": best_period,
        "t0": best_t0,
        "duration": best_dur,
        "depth": best_depth,
        "sde": sde,
        "n_periods": int(len(periods)),
        "n_points_binned": int(len(t)),
    }
