"""Baseline detector.

Steps: wotan biweight detrend, bin, astropy Box Least Squares (BLS), then a signal
detection efficiency (SDE) threshold. The baseline is simple by design. It is the floor
that each other system must beat. It also shows where a classical search fails, for
example on single transits and on shallow signals on active stars.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .types import Prediction

DEFAULT_CONFIG: dict[str, Any] = {
    "detrend_window_days": 0.75,
    "bin_minutes": 10.0,
    "min_period_days": 0.5,
    "max_period_frac_of_span": 0.5,
    "durations_days": [0.03, 0.05, 0.08, 0.12, 0.18, 0.25],
    "frequency_factor": 2.0,
    "sde_threshold": 7.0,
}


def detrend(time: np.ndarray, flux: np.ndarray, window_days: float) -> tuple[np.ndarray, np.ndarray]:
    """Detrend with a biweight filter. Drop non-finite points."""
    from wotan import flatten

    flat = flatten(time, flux, method="biweight", window_length=window_days, return_trend=False)
    ok = np.isfinite(flat)
    return time[ok], flat[ok]


def bin_lc(
    time: np.ndarray, flux: np.ndarray, bin_minutes: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin to a fixed cadence. The error of a bin is the global scatter divided by sqrt(n).

    Returns
    -------
    time, flux, err : ndarray
    """
    if bin_minutes <= 0:
        return time, flux, np.full_like(flux, np.std(flux))
    w = bin_minutes / 1440.0
    idx = np.floor((time - time.min()) / w).astype(int)
    _, inv, counts = np.unique(idx, return_inverse=True, return_counts=True)
    t = np.bincount(inv, weights=time) / counts
    f = np.bincount(inv, weights=flux) / counts
    return t, f, np.std(flux) / np.sqrt(counts)


class BLSDetector:
    """Box Least Squares baseline. It implements the :class:`~stargazer.types.Detector` protocol.

    Parameters
    ----------
    **overrides
        Any key of :data:`DEFAULT_CONFIG`.
    """

    name = "baseline_bls"

    def __init__(self, **overrides: Any):
        unknown = set(overrides) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        self._config = {**DEFAULT_CONFIG, **overrides}

    @property
    def config(self) -> dict[str, Any]:
        """The effective configuration."""
        return dict(self._config)

    def detect(self, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray) -> Prediction:
        """Run detrend, bin and BLS. Report a detection if the SDE is at or above the threshold."""
        from astropy.timeseries import BoxLeastSquares

        cfg = self._config
        t, f = detrend(time, flux, cfg["detrend_window_days"])
        t, f, dy = bin_lc(t, f, cfg["bin_minutes"])
        span = t.max() - t.min()
        max_period = max(cfg["min_period_days"] * 1.5, span * cfg["max_period_frac_of_span"])
        bls = BoxLeastSquares(t, f, dy)
        durations = np.array(cfg["durations_days"])
        periods = bls.autoperiod(
            durations,
            minimum_period=cfg["min_period_days"],
            maximum_period=max_period,
            frequency_factor=cfg["frequency_factor"],
        )
        res = bls.power(periods, durations)
        power = np.asarray(res.power)
        i = int(np.nanargmax(power))
        sde = float((power[i] - np.nanmean(power)) / np.nanstd(power))
        return Prediction(
            detected=bool(sde >= cfg["sde_threshold"]),
            period=float(res.period[i]),
            t0=float(res.transit_time[i]),
            duration=float(res.duration[i]),
            depth=float(res.depth[i]),
            extra={"sde": sde, "n_periods": len(periods), "n_points_binned": len(t)},
        )
