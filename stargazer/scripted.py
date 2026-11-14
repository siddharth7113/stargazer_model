"""Scripted-rules baseline: the procedure of the rules prompt as fixed code over the same tools.

The rules prompt (:data:`stargazer.agent.RULES_PROMPT`) gives a language model a
procedure of seven steps and a set of decision rules. This module applies the same
procedure and the same rules without a language model. It calls the same tools through
:func:`stargazer.tools.call_tool` and reads the same verdict words. The comparison of
this detector with the agent on one episode set shows what the language model adds
beyond the procedure.

An episode is one light curve that a detector analyses, with or without an injected
transit. A detector is any object that implements :class:`stargazer.types.Detector`.
The truth of an episode is its hidden answer. This module never reads the truth.

Most rules of the prompt read a verdict word, for example ``no_secondary``. Five rules
read a number from a tool result and give no limit for it, so the model decides. This
module sets a limit for each of the five. The limits are in :data:`DEFAULT_CONFIG`:

- A clean dip at phase 0: the transit fit is ``min_significance_sigma`` or more above
  a flat line, and the zoomed fold shows a dip.
- A flat curve away from the dip: no bin of the full-orbit fold away from the transit
  is deeper than ``flat_fraction`` of the dip.
- A second dip at phase 0.5: the bins of the full-orbit fold at phase 0.5 are deeper
  than ``half_phase_fraction`` of the dip.
- A residual peak near ``min_sde``: the peak is less than ``near_min_sde`` below it.
- A second event of the same depth: the two depths and the two durations differ by
  less than ``partner_tolerance``.

Step 4 of the prompt gives the two alias tests in the wrong order. The ``fold`` tool's
description gives the correct order. A second dip at phase 0.5 means that the true
period is half the trial period. A zoomed fold with no clear dip means that half the
trial events fall on empty time, so the true period is double. This module follows the
tool's description.

The threshold of 7.1 standard deviations is the detection threshold of the Kepler
transiting planet search [1]_.

References
----------
.. [1] Jenkins, J. M., Caldwell, D. A., Chandrasekaran, H., et al. (2010). Overview of
   the Kepler science processing pipeline. The Astrophysical Journal Letters, 713, L87.
   https://doi.org/10.1088/2041-8205/713/2/L87
"""

from __future__ import annotations

from typing import Any

import numpy as np

from stargazer.tools import call_tool
from stargazer.types import Prediction

DEFAULT_CONFIG: dict[str, Any] = {
    "active_flux_range": 0.05,
    "active_detrend_window_days": 0.3,
    "detrend_window_days": 0.75,
    "min_sde": 10.0,
    "near_min_sde": 1.0,
    "min_significance_sigma": 7.1,
    "flat_fraction": 0.5,
    "half_phase_fraction": 0.5,
    "single_min_snr": 10.0,
    "single_min_duration_days": 1.0 / 24.0,
    "single_max_duration_days": 1.0,
    "single_max_depth": 0.05,
    "partner_tolerance": 0.5,
}
"""The numbers of the scripted rules.

active_flux_range
    A star whose flux range (``flux_max - flux_min`` from ``summarize``) is above this
    value is active. The rules prompt gives 5 percent.
active_detrend_window_days, detrend_window_days
    The detrending window for an active star and for a quiet star, days. The tool
    descriptions give 0.3 and 0.75.
min_sde, near_min_sde
    The signal detection efficiency (SDE) at which ``search_signals`` stops, and a
    margin below it. If the residual peak is within the margin, ``bls_peaks`` looks
    for the peaks that are left. A peak within the margin becomes a candidate.
min_significance_sigma
    The improvement of the transit fit over a flat line, in standard deviations, that
    makes a "clear dip". The value is the Kepler threshold [1]_.
flat_fraction
    The deepest bin of the full-orbit fold away from the transit, as a fraction of the
    dip, that still counts as flat.
half_phase_fraction
    The depth at phase 0.5, as a fraction of the dip, that counts as a second dip.
single_min_snr, single_min_duration_days, single_max_duration_days, single_max_depth
    The single-event rule of the prompt: signal-to-noise above about 10, a duration of
    hours (days here), a fractional depth below 5 percent.
partner_tolerance
    Two events are "of the same depth" if their depths and their durations differ by
    less than this fraction.
"""


class FoldReading:
    """Read the bins of one full-orbit fold: the depth at phase 0.5 and the flatness of the orbit.

    Parameters
    ----------
    result : dict
        The result of the ``fold`` tool at ``phase_half_width`` 0.5: ``phase`` in days
        from the transit centre and ``flux_ppm`` for each bin, None for an empty bin.
    duration : float
        The transit duration, days.
    """

    def __init__(self, result: dict, duration: float):
        phase = np.asarray(result.get("phase") or [], dtype=float)
        flux = np.asarray([np.nan if value is None else value for value in result.get("flux_ppm") or []])
        keep = np.isfinite(flux)
        self._phase = np.abs(phase[keep])
        self._flux = flux[keep]
        self._bin_width = float(np.median(np.diff(np.sort(phase)))) if phase.size > 1 else 0.0
        self._duration = duration

    def half_phase_depth_ppm(self) -> float:
        """Return the mean depth of the outermost bin on each side, at phase 0.5, ppm (positive for a dip)."""
        if self._flux.size == 0:
            return 0.0
        outer = self._phase > self._phase.max() - 0.5 * self._bin_width
        return float(-np.mean(self._flux[outer]))

    def deepest_away_ppm(self) -> float:
        """Return the depth of the deepest bin away from the transit and from phase 0.5, ppm.

        Notes
        -----
        A bin is away from the transit if its centre is more than one duration and one
        bin width from phase 0. A bin is away from phase 0.5 if it is not one of the
        two outermost bins that :meth:`half_phase_depth_ppm` reads.
        """
        if self._flux.size == 0:
            return 0.0
        away = (self._phase > self._duration + self._bin_width) & (
            self._phase < self._phase.max() - 0.5 * self._bin_width
        )
        if not away.any():
            return 0.0
        return float(-np.min(self._flux[away]))


class ScriptedAnalysis:
    """Run the procedure of the rules prompt on one episode and record every tool call.

    Parameters
    ----------
    time, flux : ndarray
        The light curve: days from the start, and relative flux.
    context : EpisodeContext or None
        The star's catalogue context, passed to the tools that read it.
    config : dict
        The numbers of :data:`DEFAULT_CONFIG`.

    Notes
    -----
    One object analyses one episode, so one detector serves several threads. The
    transcript has the form of :class:`stargazer.agent.ToolAgentDetector`'s transcript
    (tool, arguments, result), so the trajectory viewer reads it.
    """

    def __init__(self, time: np.ndarray, flux: np.ndarray, context, config: dict[str, Any]):
        self._time = time
        self._flux = flux
        self._context = context
        self._config = config
        self._window = config["detrend_window_days"]
        self.transcript: list[dict] = []

    def call(self, name: str, **arguments: Any) -> dict:
        """Call one tool, record the call and return the result."""
        result = call_tool(name, arguments, self._time, self._flux, self._context)
        self.transcript.append({"tool": name, "arguments": arguments, "result": result})
        return result

    def run(self) -> Prediction:
        """Run steps 1 to 7 and return the prediction, with the reason in ``extra``."""
        summary = self.call("summarize")
        flux_range = float(summary.get("flux_max", 1.0)) - float(summary.get("flux_min", 1.0))
        active = flux_range > self._config["active_flux_range"]
        if active:
            self._window = self._config["active_detrend_window_days"]
        for signal in self.periodic_candidates():
            prediction = self.check_periodic(signal["period"], signal["t0"], signal["duration"])
            if prediction is not None:
                return prediction
        return self.single_event(active)

    def periodic_candidates(self) -> list[dict]:
        """Return the periodic signals to check, strongest first (step 2).

        A signal with one event in the data goes to the single-event step. A signal with
        ``edge_verdict`` ``at_segment_edges`` is a systematic. If the residual peak is
        near ``min_sde``, ``bls_peaks`` with the signals masked adds the peaks near it.
        """
        min_sde = self._config["min_sde"]
        search = self.call("search_signals", detrend_window_days=self._window, min_sde=min_sde)
        signals = list(search.get("signals") or [])
        residual = search.get("residual_sde")
        if signals and residual is not None and residual >= min_sde - self._config["near_min_sde"]:
            masks = [{key: signal[key] for key in ("period", "t0", "duration")} for signal in signals]
            peaks = self.call("bls_peaks", detrend_window_days=self._window, masks=masks)
            for peak in peaks.get("peaks") or []:
                known = any(abs(peak["period"] / signal["period"] - 1.0) < 0.01 for signal in signals)
                if not known and peak.get("sde", 0.0) >= min_sde - self._config["near_min_sde"]:
                    signals.append(peak)
        return [
            signal
            for signal in signals
            if signal.get("n_events_in_data", 0) > 1 and signal.get("edge_verdict") != "at_segment_edges"
        ]

    def check_periodic(
        self, period: float, t0: float, duration: float, alias_allowed: bool = True
    ) -> Prediction | None:
        """Apply step 3, step 4 and the decision rules to one periodic signal.

        Parameters
        ----------
        period, t0, duration : float
            The signal, days.
        alias_allowed : bool
            True tests the half or the double period once. An alias test does not test
            a further alias.

        Returns
        -------
        Prediction or None
            The periodic prediction if the signal passes every rule, else None.
        """
        window = self._window
        full = self.call("fold", period=period, t0=t0, phase_half_width=0.5, detrend_window_days=window)
        zoom = self.call("fold", period=period, t0=t0, phase_half_width=0.05, detrend_window_days=window)
        fit = self.call("fit_transit", period=period, t0=t0, duration=duration, detrend_window_days=window)
        dip = float(zoom.get("depth_ppm") or 0.0)
        clear = dip > 0.0 and (fit.get("significance_sigma") or 0.0) >= self._config["min_significance_sigma"]
        if not clear:
            return self.check_periodic(2.0 * period, t0, duration, False) if alias_allowed else None
        reading = FoldReading(full, duration)
        if alias_allowed and reading.half_phase_depth_ppm() >= self._config["half_phase_fraction"] * dip:
            half = self.check_periodic(period / 2.0, t0, duration, False)
            if half is not None:
                return half
        if reading.deepest_away_ppm() > self._config["flat_fraction"] * dip:
            return None
        if fit.get("density_verdict") != "plausible" or fit.get("shape") == "v_shaped":
            return None
        parity = self.call("odd_even", period=period, t0=t0, duration=duration)
        most_of_one_parity = max(parity.get("odd_events") or 0, parity.get("even_events") or 0)
        parity_ok = parity.get("verdict") == "consistent" or (
            parity.get("verdict") == "insufficient_data" and most_of_one_parity >= 2
        )
        if not parity_ok:
            return None
        secondary = self.call(
            "secondary_eclipse", period=period, t0=t0, duration=duration, detrend_window_days=window
        )
        if secondary.get("verdict") != "no_secondary":
            return None
        return Prediction(
            detected=True,
            period=float(period),
            t0=float(fit.get("t0", t0)),
            duration=float(fit.get("duration_hours", duration * 24.0)) / 24.0,
            depth=float(fit.get("depth_ppm", dip)) / 1e6,
            extra={"reason": "periodic signal passes every rule"},
        )

    def single_event(self, active: bool) -> Prediction:
        """Apply step 6 and the single-event rule, and return the prediction.

        The strongest event is the candidate. A partner of the same depth and duration
        gives a period, which must pass the periodic rules. With no partner, an event
        that meets the rule is submitted alone, except on an active star, where a lone
        dip is variability.
        """
        config = self._config
        result = self.call("single_events", detrend_window_days=self._window)
        events = sorted(result.get("events") or [], key=lambda event: event.get("snr", 0.0), reverse=True)
        if not events:
            return Prediction(detected=False, extra={"reason": "no periodic signal and no event"})
        event = events[0]
        meets_rule = (
            event["snr"] > config["single_min_snr"]
            and config["single_min_duration_days"] <= event["duration"] <= config["single_max_duration_days"]
            and event["depth_ppm"] / 1e6 < config["single_max_depth"]
        )
        if not meets_rule:
            return Prediction(detected=False, extra={"reason": "strongest event fails the single-event rule"})
        tolerance = config["partner_tolerance"]
        partners = [
            other
            for other in events[1:]
            if abs(other["depth_ppm"] / event["depth_ppm"] - 1.0) < tolerance
            and abs(other["duration"] / event["duration"] - 1.0) < tolerance
            and abs(other["time"] - event["time"]) > event["duration"]
        ]
        if partners:
            period = abs(partners[0]["time"] - event["time"])
            prediction = self.check_periodic(period, event["time"], event["duration"], False)
            return prediction or Prediction(
                detected=False, extra={"reason": "the partner's period fails the periodic rules"}
            )
        if active:
            return Prediction(detected=False, extra={"reason": "a lone dip on an active star"})
        return Prediction(
            detected=True,
            t0=float(event["time"]),
            duration=float(event["duration"]),
            depth=float(event["depth_ppm"]) / 1e6,
            extra={"reason": "single event with no partner"},
        )


class ScriptedRulesDetector:
    """The rules prompt as fixed code: the agent's procedure and decision rules with no language model.

    The detector calls the same tools as :class:`stargazer.agent.ToolAgentDetector` and
    reads the same verdict words. A results file of this detector against one of the
    agent on the same episode set shows what the model's judgement adds.

    Parameters
    ----------
    **overrides
        Any key of :data:`DEFAULT_CONFIG`.
    """

    name = "scripted_rules"

    def __init__(self, **overrides: Any):
        unknown = set(overrides) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        self._config = {**DEFAULT_CONFIG, **overrides}

    @property
    def config(self) -> dict[str, Any]:
        """The effective configuration."""
        return dict(self._config)

    def detect(self, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray, context=None) -> Prediction:
        """Run the procedure on one light curve and return the prediction with its transcript.

        Parameters
        ----------
        time, flux, flux_err : ndarray
            The light curve: days from the start, relative flux and its error. The
            tools do not read ``flux_err``.
        context : EpisodeContext, optional
            The star's catalogue context.
        """
        analysis = ScriptedAnalysis(time, flux, context, self._config)
        prediction = analysis.run()
        prediction.extra.update({"n_tool_calls": len(analysis.transcript), "transcript": analysis.transcript})
        return prediction
