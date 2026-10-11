"""Analysis tools that a language-model agent can call on one light curve.

Each function takes plain arrays and returns a small, JSON-serialisable dictionary.
The outputs are short on purpose. An agent reads them as text, so a tool returns
summary numbers and binned curves. A tool does not return raw arrays. The same functions serve tier 1
(detection), tier 2 (vetting) and tier 3 (the reinforcement-learning environment).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from stargazer.baseline import bin_lc, detrend

SEGMENT_BREAK_CADENCES = 5
"""A gap longer than this many cadences splits a light curve into segments. The same
value is the ``break_tolerance`` of the detrending filter in :func:`stargazer.baseline.detrend`."""

MAD_TO_SIGMA = 1.4826
"""Factor from the median absolute deviation to the standard deviation of a normal
distribution, ``1 / Phi^-1(3/4)`` [1]_.

References
----------
.. [1] Rousseeuw, P. J. and Croux, C. (1993). Alternatives to the median absolute
   deviation. Journal of the American Statistical Association, 88, 1273.
   https://doi.org/10.1080/01621459.1993.10476408
"""


def summarize(time: np.ndarray, flux: np.ndarray) -> dict:
    """Describe a light curve in a few numbers.

    Returns
    -------
    dict
        ``n_points``, ``span_days``, ``cadence_minutes``, ``point_scatter_ppm``,
        ``flux_min``, ``flux_max`` and ``gaps_days`` (start and length of each gap
        longer than 0.5 days).

    Notes
    -----
    The point scatter is :func:`stargazer.inject.point_scatter` in parts per million.
    A gap is a step in ``time`` longer than 0.5 days. TESS has one such gap in each
    sector, at the data downlink between the two orbits.
    """
    from stargazer.inject import point_scatter

    dt = np.diff(time)
    gap_idx = np.where(dt > 0.5)[0]
    return {
        "n_points": len(time),
        "span_days": round(float(time.max() - time.min()), 3),
        "cadence_minutes": round(float(np.median(dt)) * 1440.0, 2),
        "point_scatter_ppm": round(point_scatter(flux) * 1e6, 1),
        "flux_min": round(float(flux.min()), 6),
        "flux_max": round(float(flux.max()), 6),
        "gaps_days": [[round(float(time[index]), 3), round(float(dt[index]), 3)] for index in gap_idx],
    }


def bls_peaks(
    time: np.ndarray,
    flux: np.ndarray,
    min_period: float = 0.5,
    max_period: float | None = None,
    detrend_window_days: float = 0.75,
    bin_minutes: float = 10.0,
    n_peaks: int = 5,
    masks: list[dict] | None = None,
) -> dict:
    """Detrend, bin and run a Box Least Squares (BLS) search. Return the top peaks.

    Parameters
    ----------
    time, flux : ndarray
    min_period, max_period : float
        Period search bounds, days. ``max_period`` defaults to half the span.
    detrend_window_days : float
        Savitzky-Golay window, see :func:`stargazer.baseline.detrend`.
    bin_minutes : float
    n_peaks : int
        Number of distinct peaks to return.
    masks : list of dict, optional
        Signals to remove before the search, each ``{"period", "t0", "duration"}`` in
        days. Points within one full duration of each event centre are dropped.

    Returns
    -------
    dict
        ``peaks``: a list of ``period, t0, duration, depth_ppm, sde, n_events_in_data,
        half_period, double_period`` for the strongest peaks, each at least 2 percent
        apart in period. ``n_periods_searched`` and ``n_points_masked``. The alias
        periods and the event count are derived here so the model does not do
        arithmetic on them.

    Notes
    -----
    The signal detection efficiency (SDE) is defined in
    :meth:`stargazer.baseline.BLSDetector.detect`. Peaks are found by sorting the
    power spectrum and skipping any period within 2 percent of a stronger one, so
    the list is not five samples of one wide peak.

    Masking is how pipelines find more than one signal per star: a deep eclipse or
    a single strong dip dominates the power at every window, and a second, shallower
    periodic signal only appears once the first is removed. The mask half-width is
    one full duration, twice the transit, so ingress and egress are removed too.
    """
    from astropy.timeseries import BoxLeastSquares

    work_time, work_flux = detrend(time, flux, detrend_window_days)
    n_masked = 0
    for mask in masks or []:
        phase = ((work_time - mask["t0"] + 0.5 * mask["period"]) % mask["period"]) - 0.5 * mask["period"]
        keep = np.abs(phase) > mask["duration"]
        n_masked += int((~keep).sum())
        work_time, work_flux = work_time[keep], work_flux[keep]
    work_time, work_flux, work_err = bin_lc(work_time, work_flux, bin_minutes)
    span = work_time.max() - work_time.min()
    max_period = max_period or span * 0.5
    bls = BoxLeastSquares(work_time, work_flux, work_err)
    durations = np.array([0.03, 0.05, 0.08, 0.12, 0.18, 0.25])
    periods = bls.autoperiod(
        durations, minimum_period=min_period, maximum_period=max_period, frequency_factor=2.0
    )
    res = bls.power(periods, durations)
    power = np.asarray(res.power)
    mean, std = float(np.nanmean(power)), float(np.nanstd(power))
    order = np.argsort(-power)
    peaks = []
    for index in order:
        period = float(res.period[index])
        if any(abs(period / peak["period"] - 1) < 0.02 for peak in peaks):
            continue
        epoch = float(res.transit_time[index])
        peaks.append(
            {
                "period": round(period, 5),
                "t0": round(epoch, 4),
                "duration": round(float(res.duration[index]), 3),
                "depth_ppm": round(float(res.depth[index]) * 1e6, 1),
                "sde": round((float(power[index]) - mean) / std, 2),
                "n_events_in_data": int(np.floor((work_time.max() - epoch) / period)) + 1,
                "half_period": round(period / 2, 5),
                "double_period": round(period * 2, 5),
            }
        )
        if len(peaks) >= n_peaks:
            break
    return {"peaks": peaks, "n_periods_searched": len(periods), "n_points_masked": n_masked}


def search_signals(
    time: np.ndarray,
    flux: np.ndarray,
    min_period: float = 0.5,
    max_period: float | None = None,
    detrend_window_days: float = 0.75,
    min_sde: float = 10.0,
    max_signals: int = 3,
) -> dict:
    """Find every periodic signal in turn: search, mask the strongest, search again.

    Parameters
    ----------
    time, flux : ndarray
    min_period, max_period : float
        Period search bounds, days. ``max_period`` defaults to half the span.
    detrend_window_days : float
        Savitzky-Golay window, see :func:`stargazer.baseline.detrend`.
    min_sde : float
        The search stops when the strongest remaining peak has a signal detection
        efficiency (SDE) below this value.
    max_signals : int
        The search stops after this many signals.

    Returns
    -------
    dict
        ``signals``: the peaks accepted in order, each with the fields of
        :func:`bls_peaks` plus ``rank``; ``residual_sde``: the SDE of the strongest
        peak left after the last mask, so the caller sees where the search stopped;
        ``stop``: ``"below_min_sde"`` or ``"max_signals"``.

    Notes
    -----
    This is the multiple-planet search of the Kepler pipeline [4]_: after a transit
    signal passes, its events are removed from the light curve and the search runs
    again, until the strongest peak fails the threshold. A deep eclipse or a strong
    dip dominates the periodogram at every period, so a second, shallower signal
    only appears once the first is masked. Each pass calls :func:`bls_peaks` with
    the signals found so far as ``masks``. The SDE of each signal is measured on the
    pass that found it. The default ``min_sde`` of 10 is where noise peaks on real
    TESS stars end, as the LOGBOOK records; a signal above it is still a hypothesis
    to fold and vet.

    References
    ----------
    .. [4] Jenkins, J. M., et al. (2010). Transiting planet search in the Kepler
       pipeline. Proceedings of SPIE, 7740, 77400D. https://doi.org/10.1117/12.856764
    """
    signals: list[dict] = []
    residual_sde = None
    stop = "max_signals"
    while len(signals) < max_signals:
        found = bls_peaks(
            time,
            flux,
            min_period=min_period,
            max_period=max_period,
            detrend_window_days=detrend_window_days,
            masks=[{key: signal[key] for key in ("period", "t0", "duration")} for signal in signals],
        )["peaks"]
        if not found:
            stop = "below_min_sde"
            break
        residual_sde = found[0]["sde"]
        if residual_sde < min_sde:
            stop = "below_min_sde"
            break
        signals.append({"rank": len(signals) + 1, **found[0]})
    if stop == "max_signals":
        residual_sde = None
    return {"signals": signals, "residual_sde": residual_sde, "stop": stop}


def fold(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    n_bins: int = 40,
    detrend_window_days: float = 0.75,
    phase_half_width: float = 0.5,
) -> dict:
    """Fold the detrended light curve at a period and return a binned phase curve.

    Parameters
    ----------
    time, flux : ndarray
    period, t0 : float
        Days.
    n_bins : int
    detrend_window_days : float
    phase_half_width : float
        Phase range to return, in units of the period. 0.5 gives the whole orbit.
        0.05 zooms on the transit.

    Returns
    -------
    dict
        ``phase`` (bin centres, days from the transit centre), ``flux_ppm`` (binned
        flux minus one, integer parts per million, null for an empty bin),
        ``n_points`` (points in the window), ``n_empty_bins`` and ``depth_ppm`` (the
        deepest bin). The output is kept short because the model re-reads it on every
        later turn.

    Notes
    -----
    Phase in days is ``((t - t0 + P/2) mod P) - P/2``. A transit at the right period
    and epoch shows as a single dip at phase zero. A wrong period smears it out.
    """
    work_time, work_flux = detrend(time, flux, detrend_window_days)
    phase = ((work_time - t0 + 0.5 * period) % period) - 0.5 * period
    lim = phase_half_width * period
    mask = np.abs(phase) <= lim
    edges = np.linspace(-lim, lim, n_bins + 1)
    idx = np.clip(np.digitize(phase[mask], edges) - 1, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    sums = np.bincount(idx, weights=work_flux[mask], minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        binned = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    centres = 0.5 * (edges[:-1] + edges[1:])
    flux_ppm = (binned - 1.0) * 1e6
    return {
        "phase": [round(float(centre), 3) for centre in centres],
        "flux_ppm": [None if np.isnan(value) else round(float(value)) for value in flux_ppm],
        "n_points": int(mask.sum()),
        "n_empty_bins": int((counts == 0).sum()),
        "depth_ppm": None if np.all(np.isnan(flux_ppm)) else round(float(-np.nanmin(flux_ppm)), 1),
    }


def single_events(
    time: np.ndarray,
    flux: np.ndarray,
    detrend_window_days: float = 0.75,
    durations_days: tuple[float, ...] = (0.04, 0.08, 0.125, 0.25, 0.5),
    n_events: int = 5,
    min_points: int = 3,
    window_factor: float = 6.0,
    edge_widths: float = 3.0,
) -> dict:
    """Find the strongest individual dips, without assuming they repeat.

    Parameters
    ----------
    time, flux : ndarray
    detrend_window_days : float
        The shortest detrending window, days. Each box width uses the longer of this
        value and ``window_factor`` times the width.
    durations_days : tuple of float
        Box widths to try, days.
    n_events : int
        Number of distinct events to return.
    min_points : int
        A box with fewer points, for example at a gap edge, is skipped.
    window_factor : float
        The detrending window for a box width is at least this many widths.
    edge_widths : float
        A box closer than this many widths to the start or the end of a segment is
        skipped.

    Returns
    -------
    dict
        ``events``: a list of ``time`` (centre, days), ``duration`` (days),
        ``depth_ppm``, ``snr``, ``noise_ppm`` (the noise at the box timescale) and
        ``n_points`` for the strongest non-overlapping dips; ``point_scatter_ppm``.

    Notes
    -----
    A periodic search needs at least two events. A planet with a period longer than
    half the data has at most one, and the mission pipelines report those only as
    single-transit candidates. This tool scores each box of width ``d`` (days)
    centred at each time by the single-event statistic of the Kepler pipeline [2]_,

    ``snr = (median_box_mean - mean_flux_in_box) / sigma_d``

    where ``sigma_d`` is the noise at the box timescale: 1.4826 times the median
    absolute deviation of the box means over every box of width ``d`` that does not
    straddle a gap. On white noise ``sigma_d`` equals ``sigma / sqrt(n)`` with
    ``sigma`` the point scatter and ``n`` the points in the box. On an active star
    it is larger, in the way the combined differential photometric precision of the
    Kepler pipeline [3]_ is larger, so slow variability does not produce an event
    with a high statistic.

    The Savitzky-Golay detrending in :func:`stargazer.baseline.detrend` keeps the
    full depth of a dip only if its window is several times the dip's width. A
    caller that chooses a short window for an active star would erase every long
    event, so each box width is searched on its own detrended series with the window

    ``w = max(detrend_window_days, window_factor * d)``

    in days. A gap longer than five cadences splits the data into segments. A box
    that crosses a segment break is skipped, and so is a box within ``edge_widths``
    widths of the start or the end of its segment. At an edge the filter fits a
    one-sided window and leaves a dip-shaped residual that reaches about three
    widths into the segment; a full run reported it as the same false event on
    every episode of a star. A segment shorter than two detrending windows is all
    edge and is skipped. The strongest boxes are kept greedily with a
    separation of at least two widths, so one dip is not reported five times. An
    event is then a hypothesis: fold at a few trial periods, or check whether a
    second event of the same depth exists.

    References
    ----------
    .. [2] Jenkins, J. M. (2002). The impact of solar-like variability on the
       detectability of transiting terrestrial planets. The Astrophysical Journal,
       575, 493. https://doi.org/10.1086/341136
    .. [3] Christiansen, J. L., et al. (2012). The derivation, properties, and value
       of Kepler's combined differential photometric precision. Publications of the
       Astronomical Society of the Pacific, 124, 1279. https://doi.org/10.1086/668847
    """
    from stargazer.inject import point_scatter

    detrended: dict[float, tuple[np.ndarray, np.ndarray, float]] = {}
    candidates = []
    for duration in durations_days:
        window = max(detrend_window_days, window_factor * duration)
        if window not in detrended:
            work_time, work_flux = detrend(time, flux, window)
            detrended[window] = (work_time, work_flux, point_scatter(work_flux))
        work_time, work_flux, sigma = detrended[window]
        cadence = float(np.median(np.diff(work_time)))
        cumulative = np.concatenate([[0.0], np.cumsum(work_flux)])
        breaks = np.where(np.diff(work_time) > SEGMENT_BREAK_CADENCES * cadence)[0]
        segment_starts = np.concatenate([[work_time[0]], work_time[breaks + 1]])
        segment_ends = np.concatenate([work_time[breaks], [work_time[-1]]])
        n_box = max(min_points, round(duration / cadence))
        if n_box >= len(work_flux):
            continue
        starts = np.arange(len(work_flux) - n_box + 1)
        means = (cumulative[starts + n_box] - cumulative[starts]) / n_box
        box_starts = work_time[starts]
        box_ends = work_time[starts + n_box - 1]
        segment_of_start = np.searchsorted(segment_starts, box_starts, side="right") - 1
        segment_of_end = np.searchsorted(segment_starts, box_ends, side="right") - 1
        start_edge = segment_starts[segment_of_start]
        end_edge = segment_ends[segment_of_end]
        margin = edge_widths * duration
        whole = (
            (segment_of_start == segment_of_end)
            & (end_edge - start_edge >= 2.0 * window)
            & (box_starts - start_edge >= margin)
            & (end_edge - box_ends >= margin)
        )
        if not whole.any():
            continue
        level = float(np.median(means[whole]))
        noise = float(MAD_TO_SIGMA * np.median(np.abs(means[whole] - level)))
        if noise <= 0:
            noise = sigma / np.sqrt(n_box)
        snr = (level - means) / noise
        snr[~whole] = -np.inf
        for index in np.argsort(-snr)[: 4 * n_events]:
            if not np.isfinite(snr[index]) or snr[index] <= 0:
                break
            centre = 0.5 * (work_time[index] + work_time[index + n_box - 1])
            candidates.append(
                (float(snr[index]), float(centre), duration, float(level - means[index]), noise, n_box)
            )
    candidates.sort(reverse=True)
    events = []
    for snr_value, centre, duration, depth, noise, n_box in candidates:
        if any(abs(centre - kept["time"]) < 2 * max(duration, kept["duration"]) for kept in events):
            continue
        events.append(
            {
                "time": round(centre, 4),
                "duration": duration,
                "depth_ppm": round(depth * 1e6, 1),
                "snr": round(snr_value, 2),
                "noise_ppm": round(noise * 1e6, 1),
                "n_points": n_box,
            }
        )
        if len(events) >= n_events:
            break
    shortest = min(detrended) if detrended else None
    scatter = detrended[shortest][2] if shortest is not None else point_scatter(flux)
    return {"events": events, "point_scatter_ppm": round(scatter * 1e6, 1)}


FIT_LIMB_DARKENING_TEFF = 5500.0
"""Effective temperature, kelvin, whose TESS quadratic limb-darkening coefficients the
transit fit uses. The tool sees only a light curve, so it takes a Sun-like star."""

GRAZING_BOUND = 0.9
"""Value of ``b + k`` (impact parameter plus radius ratio) at or above which the fitted
geometry is reported as ``v_shaped``. At 1.0 the planet disc crosses the limb at
mid-transit; the margin below it covers the fit's degeneracy between ``b`` and ``k``
on a near-grazing dip."""

DENSITY_VERDICTS: tuple[tuple[float, str], ...] = (
    (0.1, "too_low_for_a_dwarf"),
    (20.0, "plausible"),
    (np.inf, "too_high"),
)
"""Upper bound of the fitted stellar density, g cm^-3, and the verdict for a value below
it. The first row that exceeds the value wins. A main-sequence star has 0.1 to 20."""


def fit_transit(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    detrend_window_days: float = 0.75,
    bin_minutes: float = 5.0,
) -> dict:
    r"""Fit a physical transit model to the folded light curve at one period and epoch.

    Parameters
    ----------
    time, flux : ndarray
    period : float
        Period, days.
    t0 : float
        Epoch of a transit centre, days.
    duration : float
        First guess of the total transit duration, days, for example from a peak.
    detrend_window_days : float
        Savitzky-Golay window, see :func:`stargazer.baseline.detrend`.
    bin_minutes : float
        Width of the phase bins the fit uses.

    Returns
    -------
    dict
        ``depth_ppm`` and ``depth_err_ppm``, ``duration_hours``, ``t0`` (refined, days),
        ``rp_rs``, ``a_rs``, ``impact_parameter``, ``stellar_density_gcc`` and
        ``density_verdict``, ``shape`` (``flat_bottom`` or ``v_shaped``),
        ``significance_sigma``, ``n_bins`` and ``n_in_transit``.

    Notes
    -----
    The fit is the vetting step a pipeline runs after the box search [5]_. The
    detrended flux is folded on the period, kept within 2.5 durations of the epoch,
    binned, and fitted with the batman model [6]_ of :func:`stargazer.inject.transit_model`
    by nonlinear least squares. The free parameters are the epoch offset, the radius
    ratio :math:`k = R_p / R_\star`, the scaled semi-major axis :math:`a / R_\star` and
    the impact parameter :math:`b`. The quadratic limb-darkening coefficients are fixed
    at the TESS values of Claret [7]_ for a star of :data:`FIT_LIMB_DARKENING_TEFF`
    kelvin, because the tool does not know the host.

    The depth is :math:`k^2` in parts per million. The duration is :math:`T_{14}` from
    :func:`stargazer.inject.transit_duration`. The stellar density follows from the
    period and :math:`a / R_\star` alone, for a circular orbit and a planet of
    negligible mass [8]_,

    .. math::
       \rho_\star = \frac{3 \pi}{G P^2} \left( \frac{a}{R_\star} \right)^3

    with :math:`G` in SI, :math:`P` in seconds and the result converted to
    :math:`\mathrm{g\,cm^{-3}}`. A main-sequence host has a density between 0.1 and
    20, so a value outside that range means the dip is not a transit of a dwarf at
    this period; :data:`DENSITY_VERDICTS` holds the bounds. The shape is
    ``flat_bottom`` if the planet disc lies well inside the stellar disc at
    mid-transit, :math:`b + k < 0.9` (:data:`GRAZING_BOUND`), and ``v_shaped``
    otherwise, a grazing geometry that a blended or grazing eclipsing binary produces
    more often than a planet. The significance is
    :math:`\sqrt{\chi^2_\mathrm{flat} - \chi^2_\mathrm{fit}}`, the improvement of
    the transit model over a flat line, in sigma. The depth error is the square root
    of the diagonal of the covariance :math:`(J^T J)^{-1}` at the solution, scaled by
    the reduced chi-square.

    References
    ----------
    .. [5] Twicken, J. D., et al. (2018). Kepler data validation I: architecture,
       diagnostic tests, and data products for vetting transiting planet candidates.
       Publications of the Astronomical Society of the Pacific, 130, 064502.
       https://doi.org/10.1088/1538-3873/aab694
    .. [6] Kreidberg, L. (2015). batman: BAsic Transit Model cAlculatioN in Python.
       Publications of the Astronomical Society of the Pacific, 127, 1161.
       https://doi.org/10.1086/683602
    .. [7] Claret, A. (2017). Limb and gravity-darkening coefficients for the TESS
       satellite. Astronomy and Astrophysics, 600, A30.
       https://doi.org/10.1051/0004-6361/201629705
    .. [8] Seager, S. and Mallén-Ornelas, G. (2003). A unique solution of planet and
       star parameters from an extrasolar planet transit light curve. The
       Astrophysical Journal, 585, 1038. https://doi.org/10.1086/346105
    """
    from scipy.optimize import least_squares

    from stargazer.inject import (
        DAY,
        G_SI,
        TransitTruth,
        limb_darkening_for_teff,
        transit_duration,
        transit_model,
    )

    work_time, work_flux = detrend(time, flux, detrend_window_days)
    phase = ((work_time - t0 + 0.5 * period) % period) - 0.5 * period
    window = max(2.5 * duration, 0.2)
    keep = np.abs(phase) < window
    phase, work_flux = phase[keep], work_flux[keep]
    bin_phase, bin_flux, bin_err = bin_lc(phase + window, work_flux, bin_minutes)
    bin_phase = bin_phase - window
    bin_err = np.where(
        bin_err > 0, bin_err, np.nanmedian(bin_err[bin_err > 0]) if (bin_err > 0).any() else 1.0
    )
    u1, u2 = limb_darkening_for_teff(FIT_LIMB_DARKENING_TEFF)

    def truth_of(params: np.ndarray) -> TransitTruth:
        dt0, rp_rs, a_rs, b = params
        b = min(b, 0.999 * a_rs)
        inc = float(np.degrees(np.arccos(b / a_rs)))
        return TransitTruth(period, dt0, rp_rs, a_rs, inc, b, u1, u2, 0.0, rp_rs**2)

    def residuals(params: np.ndarray) -> np.ndarray:
        return (bin_flux - transit_model(bin_phase, truth_of(params))) / bin_err

    in_box = np.abs(bin_phase) < 0.5 * duration
    depth_guess = max(float(1.0 - np.mean(bin_flux[in_box])) if in_box.any() else 1e-4, 1e-5)
    start = np.array([0.0, np.sqrt(depth_guess), period / (np.pi * duration), 0.3])
    lower = np.array([-0.5 * duration, 0.002, 1.2, 0.0])
    upper = np.array([0.5 * duration, 0.6, 2000.0, 1.5])
    start = np.clip(start, lower + 1e-9, upper - 1e-9)
    solution = least_squares(residuals, start, bounds=(lower, upper), method="trf")
    dt0, rp_rs, a_rs = solution.x[:3]
    fitted = truth_of(solution.x)
    chi2_fit = float(np.sum(solution.fun**2))
    chi2_flat = float(np.sum(((bin_flux - 1.0) / bin_err) ** 2))
    dof = max(len(bin_flux) - 4, 1)
    jacobian = solution.jac
    try:
        covariance = np.linalg.inv(jacobian.T @ jacobian) * max(chi2_fit / dof, 1.0)
        rp_err = float(np.sqrt(max(covariance[1, 1], 0.0)))
    except np.linalg.LinAlgError:
        rp_err = float("nan")
    fitted_duration = transit_duration(period, a_rs, fitted.inc_deg, rp_rs, fitted.b)
    density_si = 3.0 * np.pi / (G_SI * (period * DAY) ** 2) * a_rs**3  # kg m^-3
    density_gcc = float(density_si / 1000.0)
    verdict = next(name for bound, name in DENSITY_VERDICTS if density_gcc < bound)
    return {
        "depth_ppm": round(float(rp_rs**2) * 1e6, 1),
        "depth_err_ppm": round(2.0 * rp_rs * rp_err * 1e6, 1) if np.isfinite(rp_err) else None,
        "duration_hours": round(fitted_duration * 24.0, 2),
        "t0": round(float(t0 + dt0), 4),
        "rp_rs": round(float(rp_rs), 4),
        "a_rs": round(float(a_rs), 2),
        "impact_parameter": round(float(fitted.b), 3),
        "stellar_density_gcc": round(density_gcc, 3),
        "density_verdict": verdict,
        "shape": "flat_bottom" if fitted.b + rp_rs < GRAZING_BOUND else "v_shaped",
        "significance_sigma": round(float(np.sqrt(max(chi2_flat - chi2_fit, 0.0))), 1),
        "n_bins": len(bin_flux),
        "n_in_transit": int(np.sum(np.abs(bin_phase) < 0.5 * fitted_duration)),
    }


def odd_even(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    detrend_window_days: float = 0.75,
    min_points_per_event: int = 3,
) -> dict:
    """Compare the depth of odd-numbered and even-numbered transits.

    Returns
    -------
    dict
        ``odd_depth_ppm``, ``even_depth_ppm``, ``odd_events``, ``even_events``,
        ``difference_sigma`` and ``verdict``. The verdict is ``"consistent"`` when the
        difference is below 3 sigma, ``"inconsistent"`` when at or above it, and
        ``"insufficient_data"`` when either parity has fewer than two events.

    Notes
    -----
    An eclipsing binary at twice the true period shows two eclipses of different
    depth that a BLS search can mistake for one planet at half the period. A real
    planet has equal odd and even depths.

    The depth of each event is the mean flux deficit of its in-transit points. The
    depth of a parity is the mean over its events, and its standard error is the
    standard deviation of the per-event depths divided by the square root of the
    number of events. The difference in sigma is

    ``|d_odd - d_even| / sqrt(se_odd**2 + se_even**2)``

    Using the scatter between events, rather than the point scatter divided by
    sqrt(n), includes residual stellar variability after detrending. On an active
    star the point-scatter error is many times too small and flags every planet as
    a binary. The verdict field exists so the model reads a word, not a threshold.
    """
    work_time, work_flux = detrend(time, flux, detrend_window_days)
    phase = ((work_time - t0 + 0.5 * period) % period) - 0.5 * period
    epoch = np.round((work_time - t0) / period).astype(int)
    in_transit = np.abs(phase) < 0.5 * duration
    stats = {}
    for name, parity in (("odd", 1), ("even", 0)):
        depths = []
        for event in np.unique(epoch[in_transit & (epoch % 2 == parity)]):
            points = work_flux[in_transit & (epoch == event)]
            if len(points) >= min_points_per_event:
                depths.append(1.0 - points.mean())
        depths = np.array(depths)
        stats[name] = {
            "n_events": len(depths),
            "depth": float(depths.mean()) if len(depths) else None,
            "se": float(depths.std(ddof=1) / np.sqrt(len(depths))) if len(depths) >= 2 else None,
        }
    out = {
        "odd_depth_ppm": None if stats["odd"]["depth"] is None else round(stats["odd"]["depth"] * 1e6, 1),
        "even_depth_ppm": None if stats["even"]["depth"] is None else round(stats["even"]["depth"] * 1e6, 1),
        "odd_events": stats["odd"]["n_events"],
        "even_events": stats["even"]["n_events"],
        "difference_sigma": None,
        "verdict": "insufficient_data",
    }
    if stats["odd"]["se"] is not None and stats["even"]["se"] is not None:
        combined = np.hypot(stats["odd"]["se"], stats["even"]["se"])
        if combined > 0:
            sigma = abs(stats["odd"]["depth"] - stats["even"]["depth"]) / combined
            out["difference_sigma"] = round(float(sigma), 2)
            out["verdict"] = "consistent" if sigma < 3 else "inconsistent"
    return out


def secondary_eclipse(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    detrend_window_days: float = 0.75,
    n_phases: int = 200,
) -> dict:
    """Look for a second dip in the folded light curve, at phase 0.5 and at any other phase.

    Parameters
    ----------
    time, flux : ndarray
    period : float
        Period, days.
    t0 : float
        Epoch of a transit centre, days.
    duration : float
        Transit duration, days; the width of the box the search uses.
    detrend_window_days : float
        Savitzky-Golay window, see :func:`stargazer.baseline.detrend`.
    n_phases : int
        Number of trial phases across the orbit.

    Returns
    -------
    dict
        ``primary_depth_ppm``; ``secondary_depth_ppm`` and ``secondary_sigma`` at
        phase 0.5; ``strongest_other_phase``, ``strongest_other_depth_ppm`` and
        ``strongest_other_sigma`` for the deepest box away from the primary;
        ``noise_ppm``; ``verdict``: ``no_secondary``, ``secondary_at_half_phase`` or
        ``dip_at_other_phase``.

    Notes
    -----
    A self-luminous companion is eclipsed in turn, so an eclipsing binary shows a
    second dip half an orbit after the primary. A planet's secondary eclipse in the
    TESS band is below the noise of one sector for all but the hottest giants, so a
    secondary at 3 sigma or more marks the candidate as a binary. This is the
    secondary-eclipse test of the Kepler Robovetter [9]_.

    The detrended flux is folded on the period. The depth at a phase is the mean
    flux deficit of the points within half a duration of it. The noise is 1.4826
    times the median absolute deviation of the depths at ``n_phases`` trial phases,
    excluding one duration on each side of phase 0, so stellar variability at the
    transit timescale is part of it. The sigma of a depth is the depth over that
    noise. The strongest other dip is the deepest trial phase outside the primary
    window; a strong dip at a phase other than 0.5 means the period is wrong or a
    second signal is present. The verdict is ``secondary_at_half_phase`` if the
    phase-0.5 depth reaches 3 sigma, else ``dip_at_other_phase`` if the strongest
    other dip reaches 5 sigma, else ``no_secondary``.

    References
    ----------
    .. [9] Coughlin, J. L., et al. (2016). Planetary candidates observed by Kepler.
       VII. The first fully uniform catalog based on the entire 48-month data set
       (Q1-Q17 DR24). The Astrophysical Journal Supplement Series, 224, 12.
       https://doi.org/10.3847/0067-0049/224/1/12
    """
    work_time, work_flux = detrend(time, flux, detrend_window_days)
    phase = ((work_time - t0) / period) % 1.0
    half_width = 0.5 * duration / period

    def depth_at(trial: float) -> float | None:
        offset = np.abs(((phase - trial + 0.5) % 1.0) - 0.5)
        points = work_flux[offset < half_width]
        return float(1.0 - points.mean()) if len(points) >= 3 else None

    trials = np.arange(n_phases) / n_phases
    away = np.abs(((trials + 0.5) % 1.0) - 0.5) > 2.0 * half_width
    depths = np.array(
        [depth_at(trial) if ok else np.nan for trial, ok in zip(trials, away, strict=True)], dtype=float
    )
    valid = np.isfinite(depths)
    level = float(np.nanmedian(depths)) if valid.any() else 0.0
    noise = float(MAD_TO_SIGMA * np.nanmedian(np.abs(depths[valid] - level))) if valid.any() else 0.0
    primary = depth_at(0.0)
    secondary = depth_at(0.5)
    strongest = int(np.nanargmax(depths)) if valid.any() else None

    def sigma_of(depth: float | None) -> float | None:
        return None if depth is None or noise <= 0 else round((depth - level) / noise, 2)

    secondary_sigma = sigma_of(secondary)
    other_sigma = sigma_of(float(depths[strongest])) if strongest is not None else None
    if secondary_sigma is not None and secondary_sigma >= 3.0:
        verdict = "secondary_at_half_phase"
    elif other_sigma is not None and other_sigma >= 5.0:
        verdict = "dip_at_other_phase"
    else:
        verdict = "no_secondary"
    return {
        "primary_depth_ppm": None if primary is None else round(primary * 1e6, 1),
        "secondary_depth_ppm": None if secondary is None else round(secondary * 1e6, 1),
        "secondary_sigma": secondary_sigma,
        "strongest_other_phase": None if strongest is None else round(float(trials[strongest]), 3),
        "strongest_other_depth_ppm": None if strongest is None else round(float(depths[strongest]) * 1e6, 1),
        "strongest_other_sigma": other_sigma,
        "noise_ppm": round(noise * 1e6, 1),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Tool registry: one definition of each tool for every backend.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    """One tool an agent can call: its name, description, argument schema and function.

    The schema is JSON Schema, the format that OpenAI-compatible APIs and most chat
    templates use for function calling. The same spec renders to each backend, so a
    tool is defined once.

    Parameters
    ----------
    name : str
    description : str
        What the tool does and when to use it, written for the model.
    parameters : dict
        JSON Schema object with ``properties`` and ``required``.
    function : callable
        ``function(time, flux, **arguments) -> dict``.
    """

    name: str
    description: str
    parameters: dict
    function: Callable[..., dict]

    def definition(self) -> dict:
        """Return the OpenAI function-calling definition."""
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }

    def validate(self, arguments: dict) -> list[str]:
        """Check arguments against the schema. Return a list of problems; empty means valid.

        Notes
        -----
        The check covers what a model gets wrong in practice: an unknown argument, a
        missing required argument, a wrong JSON type, and a non-positive value where
        the schema has ``exclusiveMinimum``. ``number`` accepts an integer. A property
        with ``"type": ["number", "null"]`` accepts null.
        """
        props = self.parameters.get("properties", {})
        problems = [f"unknown argument: {name}" for name in arguments if name not in props]
        problems += [
            f"missing required argument: {name}"
            for name in self.parameters.get("required", [])
            if name not in arguments
        ]
        for name, value in arguments.items():
            if name not in props:
                continue
            allowed = props[name].get("type", "any")
            allowed = allowed if isinstance(allowed, list) else [allowed]
            if not any(_json_type_ok(value, json_type) for json_type in allowed):
                problems.append(f"argument {name} must be {' or '.join(allowed)}, got {type(value).__name__}")
            elif isinstance(value, int | float) and not isinstance(value, bool):
                lo = props[name].get("exclusiveMinimum")
                if lo is not None and value <= lo:
                    problems.append(f"argument {name} must be greater than {lo}, got {value}")
        return problems


def _json_type_ok(value: object, json_type: str) -> bool:
    if json_type == "any":
        return True
    if json_type == "null":
        return value is None
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "array":
        return isinstance(value, list)
    if json_type == "object":
        return isinstance(value, dict)
    return False


def _number(desc: str, default: float | None = None, nullable: bool = False, positive: bool = False) -> dict:
    """Build a JSON Schema for a numeric argument."""
    schema: dict = {"type": ["number", "null"] if nullable else "number", "description": desc}
    if default is not None:
        schema["default"] = default
    if positive:
        schema["exclusiveMinimum"] = 0
    return schema


SUBMISSION_FORMS: dict[str, tuple[str, ...]] = {
    "periodic": ("period", "t0"),
    "single_event": ("t0", "duration", "depth"),
}
"""The arguments a detection needs, by form. A submission with a period is periodic;
one without is a single event. The scorer credits each form by its own rule."""


def _submit(time: np.ndarray, flux: np.ndarray, **kwargs) -> dict:
    """Check that a detection has the arguments of its form. The agent loop reads the arguments.

    Notes
    -----
    ``detected=false`` needs nothing else. A detection with a period is the periodic
    form and needs ``t0``. A detection without a period is the single-event form and
    needs ``t0``, ``duration`` and ``depth``, so the claim names one specific dip. A
    submission that lacks an argument of its form returns an error and does not end
    the episode, so the model can complete it.
    """
    if not kwargs.get("detected"):
        return {"ok": True}
    form = "periodic" if kwargs.get("period") is not None else "single_event"
    missing = [name for name in SUBMISSION_FORMS[form] if kwargs.get(name) is None]
    if missing:
        return {"error": f"a {form} detection needs {', '.join(missing)}", "form": form}
    return {"ok": True, "form": form}


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="summarize",
        description=(
            "Use first, once. Returns n_points, span_days, cadence_minutes, point_scatter_ppm "
            "(the white-noise level; a transit needs depth well above it), flux_min and flux_max "
            "(a range far from 1.0 means an active star), and gaps_days as [start, length] pairs."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        function=lambda time, flux: summarize(time, flux),
    ),
    ToolSpec(
        name="bls_peaks",
        description=(
            "Use after summarize to find candidate periods; rerun with detrend_window_days 0.3 if the star "
            "is active and the peaks look like variability. Returns the five strongest distinct "
            "peaks, each with period (days), t0 (days from start), duration (days), depth_ppm, sde, "
            "n_events_in_data and the alias periods half_period and double_period. Read sde: on real "
            "stars noise alone reaches 10 to 14; treat a peak as a candidate only when it also folds to "
            "a clean dip and passes odd_even. Rerun with a different max_period only to search long "
            "periods."
        ),
        parameters={
            "type": "object",
            "properties": {
                "min_period": _number("Shortest period to search, days.", 0.5, positive=True),
                "max_period": _number(
                    "Longest period to search, days. Null means half the span.",
                    nullable=True,
                    positive=True,
                ),
                "detrend_window_days": _number(
                    "Detrending window, days. Default 0.75. Use 0.3 on an active star (flux range far "
                    "from 1.0) to remove variability; a window shorter than 3 transit durations "
                    "erases the transit.",
                    0.75,
                    positive=True,
                ),
                "masks": {
                    "type": "array",
                    "description": (
                        "Signals to remove before searching, so a strong feature cannot hide a weaker "
                        "periodic one. Each item is {period, t0, duration} in days, usually a rejected "
                        "peak. Use after a peak fails fold or odd_even."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "period": {"type": "number"},
                            "t0": {"type": "number"},
                            "duration": {"type": "number"},
                        },
                        "required": ["period", "t0", "duration"],
                    },
                },
            },
            "required": [],
        },
        function=lambda time, flux, min_period=0.5, max_period=None, detrend_window_days=0.75, masks=None: (
            bls_peaks(
                time,
                flux,
                min_period=min_period,
                max_period=max_period,
                detrend_window_days=detrend_window_days,
                masks=masks,
            )
        ),
    ),
    ToolSpec(
        name="search_signals",
        description=(
            "Run the full periodic search in one call: find the strongest peak, mask it, search again, "
            "and stop when the strongest remaining peak has sde below min_sde or after max_signals. "
            "Returns signals in order (each with period, t0, duration, depth_ppm, sde, "
            "n_events_in_data and the alias periods), residual_sde (the strongest peak left, so a "
            "value near min_sde means a third signal may hide there) and stop. Each signal is a "
            "hypothesis: fold and odd_even it before you submit. Use this in place of repeated "
            "bls_peaks calls with masks."
        ),
        parameters={
            "type": "object",
            "properties": {
                "min_period": _number("Shortest period to search, days.", 0.5, positive=True),
                "max_period": _number(
                    "Longest period to search, days. Default is half the span.",
                    None,
                    nullable=True,
                    positive=True,
                ),
                "detrend_window_days": _number(
                    "Detrending window, days. Default 0.75. Use 0.3 on an active star.", 0.75, positive=True
                ),
                "min_sde": _number(
                    "Stop when the strongest remaining peak is below this sde. Default 10, where noise "
                    "peaks on real stars end.",
                    10.0,
                    positive=True,
                ),
                "max_signals": {
                    "type": "integer",
                    "description": "Stop after this many signals. Default 3.",
                    "default": 3,
                },
            },
            "required": [],
        },
        function=lambda time, flux, **arguments: search_signals(time, flux, **arguments),
    ),
    ToolSpec(
        name="single_events",
        description=(
            "Find the strongest individual dips without assuming they repeat. Use when bls_peaks finds "
            "nothing convincing, or when a planet could have only one or two transits in the data "
            "(long period). Returns up to five events with time (days from start), duration, "
            "depth_ppm, snr, noise_ppm (the noise at the event's timescale, which stellar variability "
            "raises) and n_points. An event with snr above about 8 is a hypothesis, not a "
            "detection: look for a second event of the same depth and fold at the period between "
            "them. A lone event with no partner, snr above about 10, a transit-like duration (hours) "
            "and a flat curve around it can be submitted as a single event: detected=true with t0, "
            "duration and depth and no period."
        ),
        parameters={
            "type": "object",
            "properties": {
                "detrend_window_days": _number(
                    "Shortest detrending window, days. Each box width is searched with a window of at "
                    "least six times the width, so a short window on an active star does not erase "
                    "a long event.",
                    0.75,
                    positive=True,
                ),
            },
            "required": [],
        },
        function=lambda time, flux, detrend_window_days=0.75: single_events(
            time, flux, detrend_window_days=detrend_window_days
        ),
    ),
    ToolSpec(
        name="fold",
        description=(
            "Use to check one candidate. Folds the detrended light curve at period and t0 and returns "
            "phase (days from the transit centre), flux_ppm per bin (null when a bin is empty), "
            "n_points, n_empty_bins and depth_ppm. A planet shows one dip centred at phase 0 with the "
            "rest flat near 0 ppm. Use phase_half_width 0.5 to see the whole orbit (a second dip at "
            "phase 0.5 means the true period is half) and 0.05 to see the transit shape."
        ),
        parameters={
            "type": "object",
            "properties": {
                "period": _number("Period, days.", positive=True),
                "t0": _number("Epoch of a transit centre, days from the start."),
                "phase_half_width": _number(
                    "Fraction of the period to show on each side of zero.", 0.5, positive=True
                ),
                "detrend_window_days": _number(
                    "Detrending window, days. Use the same value as in bls_peaks.", 0.75, positive=True
                ),
            },
            "required": ["period", "t0"],
        },
        function=lambda time, flux, period, t0, phase_half_width=0.5, detrend_window_days=0.75: fold(
            time,
            flux,
            period=period,
            t0=t0,
            phase_half_width=phase_half_width,
            detrend_window_days=detrend_window_days,
        ),
    ),
    ToolSpec(
        name="fit_transit",
        description=(
            "Fit a physical transit model (batman) to one candidate at period, t0 and duration. Returns "
            "depth_ppm with depth_err_ppm, duration_hours, a refined t0, impact_parameter, "
            "stellar_density_gcc with density_verdict (plausible for a dwarf star between 0.1 and 20; "
            "too_low_for_a_dwarf or too_high means the dip does not fit a planet at this period), shape "
            "(flat_bottom is planet-like; v_shaped is grazing, as eclipsing binaries often are) and "
            "significance_sigma (the improvement over a flat line). Use after the fold and before "
            "odd_even on every candidate you consider submitting."
        ),
        parameters={
            "type": "object",
            "properties": {
                "period": _number("Period, days.", positive=True),
                "t0": _number("Epoch of a transit centre, days from the start."),
                "duration": _number(
                    "First guess of the transit duration, days, from the peak.", positive=True
                ),
                "detrend_window_days": _number(
                    "Detrending window, days. Use the same value as in the search.", 0.75, positive=True
                ),
            },
            "required": ["period", "t0", "duration"],
        },
        function=lambda time, flux, period, t0, duration, detrend_window_days=0.75: fit_transit(
            time, flux, period=period, t0=t0, duration=duration, detrend_window_days=detrend_window_days
        ),
    ),
    ToolSpec(
        name="odd_even",
        description=(
            "Compare the depths of odd and even transits at a period, epoch and duration. Read the "
            "verdict field: consistent means planet-like; inconsistent means an eclipsing binary at "
            "twice the period; insufficient_data means fewer than two events of one parity."
        ),
        parameters={
            "type": "object",
            "properties": {
                "period": _number("Period, days.", positive=True),
                "t0": _number("Epoch, days from the start."),
                "duration": _number("Transit duration, days.", positive=True),
            },
            "required": ["period", "t0", "duration"],
        },
        function=lambda time, flux, period, t0, duration: odd_even(
            time, flux, period=period, t0=t0, duration=duration
        ),
    ),
    ToolSpec(
        name="secondary_eclipse",
        description=(
            "Look for a second dip in the fold at a period, epoch and duration. Read the verdict: "
            "secondary_at_half_phase means an eclipsing binary (a self-luminous companion); "
            "dip_at_other_phase means the period is wrong or a second signal is present; "
            "no_secondary is planet-like. Also returns the depths and sigmas behind the verdict."
        ),
        parameters={
            "type": "object",
            "properties": {
                "period": _number("Period, days.", positive=True),
                "t0": _number("Epoch, days from the start."),
                "duration": _number("Transit duration, days.", positive=True),
                "detrend_window_days": _number(
                    "Detrending window, days. Use the same value as in the search.", 0.75, positive=True
                ),
            },
            "required": ["period", "t0", "duration"],
        },
        function=lambda time, flux, period, t0, duration, detrend_window_days=0.75: secondary_eclipse(
            time, flux, period=period, t0=t0, duration=duration, detrend_window_days=detrend_window_days
        ),
    ),
    ToolSpec(
        name="submit_prediction",
        description=(
            "Call exactly once, as the last action, before calls_remaining reaches 0. Two forms. "
            "Periodic: detected=true with period, t0, duration and depth, only for a candidate that "
            "folds to a clean dip and has odd_even verdict consistent. Single event: detected=true "
            "with t0, duration and depth and no period, only for one isolated dip from single_events "
            "that no period explains. Otherwise submit detected=false. A false alarm and a missing "
            "submission both score zero; a correct detected=false scores full marks on a star with "
            "no planet."
        ),
        parameters={
            "type": "object",
            "properties": {
                "detected": {"type": "boolean", "description": "True if a transit signal is present."},
                "period": _number(
                    "Period, days. Give it for a periodic detection; omit it for a single event.",
                    nullable=True,
                    positive=True,
                ),
                "t0": _number(
                    "Epoch of a transit centre, days from the start. Required if detected.", nullable=True
                ),
                "duration": _number(
                    "Transit duration, days. Required for a single event.", nullable=True, positive=True
                ),
                "depth": _number(
                    "Fractional depth, for example 0.0005 for 500 ppm. Required for a single event.",
                    nullable=True,
                    positive=True,
                ),
                "reasoning": {"type": "string", "description": "One or two sentences on the evidence."},
            },
            "required": ["detected"],
        },
        function=_submit,
    ),
)

TOOLS_BY_NAME: dict[str, ToolSpec] = {tool.name: tool for tool in TOOLS}


def tool_definitions() -> list[dict]:
    """Return every tool in OpenAI function-calling format."""
    return [tool.definition() for tool in TOOLS]


def call_tool(name: str, arguments: dict, time: np.ndarray, flux: np.ndarray) -> dict:
    """Validate and run one tool. An invalid call returns ``{"error": ...}`` instead of raising.

    Notes
    -----
    The error goes back to the model as the tool result, so the model can correct the
    call. A crash inside a tool is also returned as an error, so one bad call does not
    end the episode.
    """
    spec = TOOLS_BY_NAME.get(name)
    if spec is None:
        return {"error": f"unknown tool: {name}", "available_tools": sorted(TOOLS_BY_NAME)}
    problems = spec.validate(arguments)
    if problems:
        return {
            "error": "; ".join(problems),
            "expected_arguments": {
                key: schema.get("description", "") for key, schema in spec.parameters["properties"].items()
            },
            "required": spec.parameters.get("required", []),
        }
    try:
        return spec.function(time, flux, **arguments)
    except Exception as error:  # a tool failure is reported to the model, not raised
        return {"error": f"{type(error).__name__}: {error}"[:300]}
