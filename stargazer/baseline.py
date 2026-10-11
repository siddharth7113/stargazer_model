"""Baseline detector.

Steps: Savitzky-Golay detrend [1]_ through lightkurve [2]_, bin, astropy Box Least
Squares (BLS) [3]_ [4]_, then a signal detection efficiency (SDE) threshold [3]_. The
baseline is simple by design. It is the floor that each other system must beat. It also
shows where a classical search fails, for example on single transits and on shallow
signals on active stars.

References
----------
.. [1] Savitzky, A. and Golay, M. J. E. (1964). Smoothing and differentiation of data
   by simplified least squares procedures. Analytical Chemistry, 36, 1627.
   https://doi.org/10.1021/ac60214a047
.. [2] Lightkurve Collaboration, Cardoso, J. V. de M., Hedges, C., et al. (2018).
   Lightkurve: Kepler and TESS time series analysis in Python. Astrophysics Source
   Code Library, ascl:1812.013. https://ascl.net/1812.013
.. [3] Kovács, G., Zucker, S., and Mazeh, T. (2002). A box-fitting algorithm in the
   search for periodic transits. Astronomy and Astrophysics, 391, 369.
   https://doi.org/10.1051/0004-6361:20020802
.. [4] Astropy Collaboration, Price-Whelan, A. M., Lim, P. L., et al. (2022). The
   Astropy Project: sustaining and growing a community-oriented open-source project
   and the latest major release (v5.0) of the core package. The Astrophysical
   Journal, 935, 167. https://doi.org/10.3847/1538-4357/ac7c74
"""

from __future__ import annotations

from typing import Any

import numpy as np

from stargazer.types import Prediction

DEFAULT_CONFIG: dict[str, Any] = {
    "detrend_window_days": 0.75,
    "bin_minutes": 10.0,
    "min_period_days": 0.5,
    "max_period_frac_of_span": 0.5,
    "durations_days": [0.03, 0.05, 0.08, 0.12, 0.18, 0.25],
    "frequency_factor": 2.0,
    "sde_threshold": 7.0,
}


def detrend(
    time: np.ndarray, flux: np.ndarray, window_days: float, polyorder: int = 2, sigma: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """Detrend with a Savitzky-Golay filter. Drop non-finite points.

    Parameters
    ----------
    time : ndarray
        Days.
    flux : ndarray
        Normalised flux.
    window_days : float
        Width of the sliding window, days.
    polyorder : int
        Polynomial order of the filter.
    sigma : float
        Sigma-clipping threshold for outliers during the trend fit.

    Notes
    -----
    The function uses ``lightkurve.LightCurve.flatten`` [2]_, which wraps
    ``scipy.signal.savgol_filter`` [1]_. The filter fits a polynomial of order
    ``polyorder`` in a sliding window and divides the flux by the fit. It:

    - converts ``window_days`` to an odd number of cadences from the median time step,
    - splits the light curve at gaps longer than five cadences, so a gap does not
      distort the fit on either side,
    - clips points more than ``sigma`` from the trend and refits, so deep transits do
      not pull the trend down.

    The window must be about three times the longest transit duration to keep the
    transit. The default 0.75 days keeps transits up to about 6 hours. A longer
    transit loses depth. This is one reason the baseline misses long-period signals.
    """
    import lightkurve as lk

    cadence = float(np.median(np.diff(time)))
    window_length = round(window_days / cadence)
    window_length = max(window_length + (1 - window_length % 2), polyorder + 2)  # odd and valid
    lc = lk.LightCurve(time=time, flux=flux)
    flat = lc.flatten(window_length=window_length, polyorder=polyorder, sigma=sigma, break_tolerance=5)
    f = np.asarray(flat.flux.value, dtype=np.float64)
    ok = np.isfinite(f)
    return time[ok], f[ok]


def bin_lc(
    time: np.ndarray, flux: np.ndarray, bin_minutes: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin to a fixed cadence. The error of a bin is the global scatter divided by sqrt(n).

    Parameters
    ----------
    time : ndarray
        Days.
    flux : ndarray
    bin_minutes : float
        Bin width. Zero or negative returns the input unbinned.

    Returns
    -------
    time, flux, err : ndarray

    Notes
    -----
    Each point gets a bin index ``floor((t - t_min) / w)`` with ``w`` the width in
    days. The binned time and flux are the means in the bin. The error is
    ``std(flux) / sqrt(n)`` with ``n`` the number of points in the bin and
    ``std(flux)`` the standard deviation of the whole detrended light curve. This
    assumes white noise. Binning from 2 minutes to 10 minutes reduces the number of
    points by five and the BLS run time by about the same factor. It does not reduce
    sensitivity to transits longer than about one hour.
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
    """The reference detection system: a classical Box Least Squares (BLS) search.

    BLS [3]_ folds the light curve at each trial period and looks
    for a box-shaped dip. It is the standard first-pass transit search. This class
    is the floor that each other system must beat, and it is the simplest example
    of the :class:`~stargazer.types.Detector` protocol. It also shows where a
    periodic search fails: single transits and shallow signals on active stars.

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
        """Run detrend, bin and BLS. Report a detection if the SDE is at or above the threshold.

        Notes
        -----
        The steps are:

        1. :func:`detrend` with ``detrend_window_days``.
        2. :func:`bin_lc` with ``bin_minutes``.
        3. Build the period grid with ``BoxLeastSquares.autoperiod``. The grid is uniform
           in frequency with a step set by the shortest duration, the baseline and
           ``frequency_factor``. A larger factor gives a coarser grid and a faster run.
           The longest period is ``max_period_frac_of_span`` times the baseline, so that
           at least two transits fit in the data.
        4. ``BoxLeastSquares.power`` with the log-likelihood objective. For each period
           and duration it finds the best epoch and reports the power.
        5. The signal detection efficiency of Kovács, Zucker and Mazeh [3]_ is

           ``SDE = (max(power) - mean(power)) / std(power)``

           over the whole period grid. It measures how far the best peak stands above
           the typical peak. Pure noise gives an SDE of about 5 to 7 on a grid of this
           size. The threshold is chosen from null episodes, see
           :mod:`stargazer.calibrate`.

        The prediction carries the period, epoch, duration and depth of the best peak,
        whether or not it passes the threshold, so a results file can be re-scored at
        another threshold.
        """
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
