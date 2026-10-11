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

    Returns
    -------
    dict
        ``peaks``: a list of ``period, t0, duration, depth_ppm, sde`` for the strongest
        peaks, each at least 2 percent apart in period. ``n_periods_searched``.

    Notes
    -----
    The signal detection efficiency (SDE) is defined in
    :meth:`stargazer.baseline.BLSDetector.detect`. Peaks are found by sorting the
    power spectrum and skipping any period within 2 percent of a stronger one, so
    the list is not five samples of one wide peak.
    """
    from astropy.timeseries import BoxLeastSquares

    work_time, work_flux = detrend(time, flux, detrend_window_days)
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
        peaks.append(
            {
                "period": round(period, 5),
                "t0": round(float(res.transit_time[index]), 4),
                "duration": round(float(res.duration[index]), 3),
                "depth_ppm": round(float(res.depth[index]) * 1e6, 1),
                "sde": round((float(power[index]) - mean) / std, 2),
            }
        )
        if len(peaks) >= n_peaks:
            break
    return {"peaks": peaks, "n_periods_searched": len(periods)}


def fold(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    n_bins: int = 50,
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
        ``phase`` (bin centres, in days from the transit centre), ``flux_ppm``
        (binned flux minus one, in parts per million), ``n`` (points in each bin)
        and ``depth_ppm`` (the deepest bin).

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
        "phase": [round(float(centre), 4) for centre in centres],
        "flux_ppm": [None if np.isnan(value) else round(float(value), 1) for value in flux_ppm],
        "n": [int(centre) for centre in counts],
        "depth_ppm": None if np.all(np.isnan(flux_ppm)) else round(float(-np.nanmin(flux_ppm)), 1),
    }


def odd_even(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    detrend_window_days: float = 0.75,
) -> dict:
    """Compare the depth of odd-numbered and even-numbered transits.

    Returns
    -------
    dict
        ``odd_depth_ppm``, ``even_depth_ppm``, ``odd_n``, ``even_n``, ``difference_sigma``.

    Notes
    -----
    An eclipsing binary at twice the true period shows two eclipses of different
    depth that a BLS search can mistake for one planet at half the period. A real
    planet has equal odd and even depths. The difference in sigma is

    ``|d_odd - d_even| / sqrt(se_odd**2 + se_even**2)``

    with each depth the mean in-transit flux deficit and each standard error the
    point scatter divided by the square root of the number of in-transit points.
    Above 3 sigma is a warning sign.
    """
    from stargazer.inject import point_scatter

    work_time, work_flux = detrend(time, flux, detrend_window_days)
    phase = ((work_time - t0 + 0.5 * period) % period) - 0.5 * period
    epoch = np.round((work_time - t0) / period).astype(int)
    in_tr = np.abs(phase) < 0.5 * duration
    sigma = point_scatter(work_flux)
    out = {}
    for name, parity in (("odd", 1), ("even", 0)):
        mask = in_tr & (epoch % 2 == parity)
        n_points = int(mask.sum())
        depth = float(1.0 - work_flux[mask].mean()) if n_points else float("nan")
        out[f"{name}_depth_ppm"] = None if n_points == 0 else round(depth * 1e6, 1)
        out[f"{name}_n"] = n_points
        out[f"_{name}_se"] = sigma / np.sqrt(n_points) if n_points else float("nan")
        out[f"_{name}_depth"] = depth
    se = np.hypot(out.pop("_odd_se"), out.pop("_even_se"))
    difference = abs(out.pop("_odd_depth") - out.pop("_even_depth"))
    out["difference_sigma"] = None if not np.isfinite(se) or se == 0 else round(float(difference / se), 2)
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
            "Describe the light curve: number of points, span in days, cadence, noise level in ppm, and gaps."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        function=lambda time, flux: summarize(time, flux),
    ),
    ToolSpec(
        name="bls_peaks",
        description=(
            "Detrend and run a Box Least Squares period search. Returns the five strongest peaks with "
            "period (days), t0 (days from start), duration (days), depth_ppm and SDE. SDE above about 7 "
            "is a candidate; below 6 is noise."
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
            },
            "required": [],
        },
        function=lambda time, flux, min_period=0.5, max_period=None: bls_peaks(
            time, flux, min_period=min_period, max_period=max_period
        ),
    ),
    ToolSpec(
        name="fold",
        description=(
            "Fold the detrended light curve at a period and epoch. Returns a 50-bin phase curve in ppm "
            "and the depth of the deepest bin. A real transit is one dip at phase zero. Use "
            "phase_half_width 0.5 for the whole orbit and 0.05 to zoom on the transit."
        ),
        parameters={
            "type": "object",
            "properties": {
                "period": _number("Period, days.", positive=True),
                "t0": _number("Epoch of a transit centre, days from the start."),
                "phase_half_width": _number(
                    "Fraction of the period to show on each side of zero.", 0.5, positive=True
                ),
            },
            "required": ["period", "t0"],
        },
        function=lambda time, flux, period, t0, phase_half_width=0.5: fold(
            time, flux, period=period, t0=t0, phase_half_width=phase_half_width
        ),
    ),
    ToolSpec(
        name="odd_even",
        description=(
            "Compare the depths of odd and even transits at a period, epoch and duration. A difference "
            "above 3 sigma suggests an eclipsing binary at twice the period."
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
            "Submit the final answer. Call it exactly once, as the last action. If no convincing signal "
            "exists, submit detected=false; a false alarm scores zero."
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
        return {"error": f"unknown tool: {name}. Available: {sorted(TOOLS_BY_NAME)}"}
    problems = spec.validate(arguments)
    if problems:
        return {"error": "; ".join(problems)}
    try:
        return spec.function(time, flux, **arguments)
    except Exception as error:  # a tool failure is reported to the model, not raised
        return {"error": f"{type(error).__name__}: {error}"[:300]}
