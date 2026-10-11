"""The interface between a detection system and the benchmark.

A detection system of any kind, from a Box Least Squares search to a language-model
agent, implements :class:`Detector`. The benchmark hands it one episode, which is one
light curve as three arrays, and receives one :class:`Prediction`. The scorer in
:mod:`stargazer.score` compares the prediction with the hidden truth of the episode.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class Prediction:
    """The answer a detection system gives for one light curve.

    A detection system looks at one light curve, called an episode, and reports
    whether it found a transit signal. If it did, it reports the signal's period,
    epoch, duration and depth. A signal with one transit in the data has no period;
    the system then reports the event time as ``t0`` and leaves ``period`` None.
    The benchmark compares this answer with the hidden truth of the episode. A
    prediction is the only output the benchmark needs from a system, so a system in
    any language can take part by writing predictions as JSON.

    Parameters
    ----------
    detected : bool
        True if the detector claims a transit signal is present.
    period : float or None
        Claimed period in days. None with ``detected`` True is the single-event form.
    t0 : float or None
        Claimed epoch of a transit centre, in the episode's relative time base (days).
        Required when ``detected`` is True.
    duration : float or None
        Claimed transit duration in days. Informational.
    depth : float or None
        Claimed fractional depth. Informational, flagged but not scored.
    extra : dict
        Other values the detector records, for example statistics, timings and reasoning.
    """

    detected: bool
    period: float | None = None
    t0: float | None = None
    duration: float | None = None
    depth: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary for JSON. The keys of ``extra`` are at the top level."""
        data = asdict(self)
        extra = data.pop("extra")
        return {**extra, **data}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Prediction:
        """Build a prediction from a dictionary. Unknown keys go into ``extra``.

        Parameters
        ----------
        d : dict
            Must contain ``detected``. Other known fields are optional.
        """
        known = {key: data.get(key) for key in ("period", "t0", "duration", "depth")}
        extra = {key: value for key, value in data.items() if key not in known and key != "detected"}
        return cls(detected=bool(data.get("detected", False)), extra=extra, **known)


@runtime_checkable
class Detector(Protocol):
    """The interface that each detection system implements.

    A detection system receives one light curve as three arrays, analyses it, and
    returns a :class:`Prediction`. The benchmark does not care how the system works.
    The system can be a classical period search, a neural network or a language
    model that calls tools. :class:`stargazer.baseline.BLSDetector` is the reference
    implementation.

    An implementation must be deterministic for a given ``config``. This lets a
    results file reproduce a run. The ``config`` must hold each value that changes
    the output, for example thresholds, window sizes, model names and prompts.

    A detector that also reads the star's catalogue rows declares a keyword
    parameter ``context`` on ``detect``; the benchmark then passes the episode's
    :class:`stargazer.context.EpisodeContext`, or None for a set without one. A
    detector without that parameter receives the three arrays only.

    Examples
    --------
    >>> class AlwaysNone:
    ...     name = "always_none"
    ...     config = {}
    ...
    ...     def detect(self, time, flux, flux_err):
    ...         return Prediction(detected=False)
    >>> isinstance(AlwaysNone(), Detector)
    True
    """

    name: str

    @property
    def config(self) -> dict[str, Any]:
        """Configuration that fully determines the detector's behaviour."""
        ...

    def detect(self, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray) -> Prediction:
        """Analyse one light curve.

        Parameters
        ----------
        time : ndarray
            Relative time in days, starting at zero.
        flux : ndarray
            Median-normalised flux.
        flux_err : ndarray
            Per-point flux uncertainty.

        Returns
        -------
        Prediction
        """
        ...
