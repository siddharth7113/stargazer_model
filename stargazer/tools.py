"""Analysis tools that a language-model agent can call on one light curve.

Each function takes plain arrays and returns a small, JSON-serialisable dictionary.
The outputs are short on purpose: an agent reads them as text, so a tool returns
summary numbers and binned curves, never raw arrays. The same functions serve tier 1
(detection), tier 2 (vetting) and tier 3 (the reinforcement-learning environment).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from stargazer.baseline import bin_lc, detrend


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
) -> dict:
    """Find the strongest individual dips, without assuming they repeat.

    Parameters
    ----------
    time, flux : ndarray
    detrend_window_days : float
    durations_days : tuple of float
        Box widths to try. The longest must stay below three times the detrend window
        or the detrend step erases the dip.
    n_events : int
        Number of distinct events to return.
    min_points : int
        A box with fewer points, for example at a gap edge, is skipped.

    Returns
    -------
    dict
        ``events``: a list of ``time`` (centre, days), ``duration`` (days),
        ``depth_ppm``, ``snr`` and ``n_points`` for the strongest non-overlapping dips;
        ``point_scatter_ppm``.

    Notes
    -----
    A periodic search needs at least two events. A planet with a period longer than
    half the data has at most one, and the mission pipelines report those only as
    single-transit candidates. This tool scores each box of width ``d`` centred at
    each time by

    ``snr = (1 - mean_flux_in_box) / (sigma / sqrt(n))``

    with ``sigma`` the point-to-point scatter and ``n`` the points in the box. That
    is the single-event statistic of the Kepler pipeline [1]_. Boxes that straddle a
    gap, where the points span more than 1.5 times the box width, are skipped. The
    strongest boxes are kept greedily with a separation of at least two widths, so
    one dip is not reported five times. An event is then a hypothesis: fold at a few
    trial periods, or check whether a second event of the same depth exists.

    References
    ----------
    .. [1] Jenkins, J. M. (2002). The impact of solar-like variability on the
       detectability of transiting terrestrial planets. The Astrophysical Journal,
       575, 493. https://doi.org/10.1086/341136
    """
    from stargazer.inject import point_scatter

    work_time, work_flux = detrend(time, flux, detrend_window_days)
    sigma = point_scatter(work_flux)
    cadence = float(np.median(np.diff(work_time)))
    cumulative = np.concatenate([[0.0], np.cumsum(work_flux)])
    candidates = []
    for duration in durations_days:
        n_box = max(min_points, round(duration / cadence))
        if n_box >= len(work_flux):
            continue
        starts = np.arange(len(work_flux) - n_box + 1)
        spans = work_time[starts + n_box - 1] - work_time[starts]
        means = (cumulative[starts + n_box] - cumulative[starts]) / n_box
        snr = (1.0 - means) / (sigma / np.sqrt(n_box))
        snr[spans > 1.5 * duration] = -np.inf
        for index in np.argsort(-snr)[: 4 * n_events]:
            if not np.isfinite(snr[index]) or snr[index] <= 0:
                break
            centre = 0.5 * (work_time[index] + work_time[index + n_box - 1])
            candidates.append((float(snr[index]), float(centre), duration, float(1.0 - means[index]), n_box))
    candidates.sort(reverse=True)
    events = []
    for snr_value, centre, duration, depth, n_box in candidates:
        if any(abs(centre - kept["time"]) < 2 * max(duration, kept["duration"]) for kept in events):
            continue
        events.append(
            {
                "time": round(centre, 4),
                "duration": duration,
                "depth_ppm": round(depth * 1e6, 1),
                "snr": round(snr_value, 2),
                "n_points": n_box,
            }
        )
        if len(events) >= n_events:
            break
    return {"events": events, "point_scatter_ppm": round(sigma * 1e6, 1)}


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


def _submit(time: np.ndarray, flux: np.ndarray, **kwargs) -> dict:
    """The submit tool has no analysis; the agent loop reads its arguments."""
    return {"ok": True}


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
        name="single_events",
        description=(
            "Find the strongest individual dips without assuming they repeat. Use when bls_peaks finds "
            "nothing convincing, or when a planet could have only one or two transits in the data "
            "(long period). Returns up to five events with time (days from start), duration, "
            "depth_ppm, snr and n_points. An event with snr above about 8 is a hypothesis, not a "
            "detection: look for a second event of the same depth and fold at the period between "
            "them. A submission needs a period, so a lone event with no partner stays detected=false."
        ),
        parameters={
            "type": "object",
            "properties": {
                "detrend_window_days": _number(
                    "Detrending window, days. Use the same value as in bls_peaks.", 0.75, positive=True
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
        name="submit_prediction",
        description=(
            "Call exactly once, as the last action, before calls_remaining reaches 0. Submit "
            "detected=true with period, t0, duration and depth only for a candidate that folds to a "
            "clean dip and has odd_even verdict consistent. Otherwise submit detected=false. A false "
            "alarm and a missing submission both score zero; a correct detected=false scores full marks "
            "on a star with no planet."
        ),
        parameters={
            "type": "object",
            "properties": {
                "detected": {"type": "boolean", "description": "True if a transit signal is present."},
                "period": _number("Period, days. Required if detected.", nullable=True, positive=True),
                "t0": _number(
                    "Epoch of a transit centre, days from the start. Required if detected.", nullable=True
                ),
                "duration": _number("Transit duration, days.", nullable=True, positive=True),
                "depth": _number(
                    "Fractional depth, for example 0.0005 for 500 ppm.", nullable=True, positive=True
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
