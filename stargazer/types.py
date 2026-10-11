"""Core types shared across the package: predictions and the detector protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class Prediction:
    """What a detector says about one episode.

    Parameters
    ----------
    detected : bool
        True if the detector claims a transit signal is present.
    period : float or None
        Claimed period in days. Required when ``detected`` is True.
    t0 : float or None
        Claimed epoch of a transit centre, in the episode's relative time base (days).
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
        d = asdict(self)
        extra = d.pop("extra")
        return {**extra, **d}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Prediction:
        """Build a prediction from a dictionary. Unknown keys go into ``extra``.

        Parameters
        ----------
        d : dict
            Must contain ``detected``. Other known fields are optional.
        """
        known = {k: d.get(k) for k in ("period", "t0", "duration", "depth")}
        extra = {k: v for k, v in d.items() if k not in known and k != "detected"}
        return cls(detected=bool(d.get("detected", False)), extra=extra, **known)


@runtime_checkable
class Detector(Protocol):
    """An object that analyses one light curve and returns a :class:`Prediction`.

    An implementation must be deterministic for a given ``config``. This lets a
    results file reproduce a run.
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
