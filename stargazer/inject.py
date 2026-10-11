"""Transit injection with batman. All parameters are explicit and stored as truth.

Geometry follows the standard relations:
  a/R* from Kepler's third law using stellar density (logg and radius from the TIC),
  inclination from impact parameter b = a/R* cos(i),
  T14 = P/pi * arcsin( sqrt((1+k)^2 - b^2) / (a/R* sin i) ), k = Rp/R*.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

G_SI = 6.67430e-11
R_SUN = 6.957e8
DAY = 86400.0


@dataclass
class TransitTruth:
    period: float        # days
    t0: float            # days, same time base as the light curve
    rp_rs: float         # planet to star radius ratio
    a_rs: float          # semi-major axis in stellar radii
    inc_deg: float
    b: float             # impact parameter
    u1: float
    u2: float
    duration: float      # T14 in days
    depth: float         # rp_rs**2, geometric depth before limb darkening

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TransitTruth":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


def a_rs_from_star(period_days: float, logg_cgs: float, radius_rsun: float) -> float:
    g = 10.0 ** logg_cgs * 1e-2  # cgs cm/s^2 -> m/s^2
    R = radius_rsun * R_SUN
    M = g * R**2 / G_SI
    P = period_days * DAY
    a = (G_SI * M * P**2 / (4 * np.pi**2)) ** (1.0 / 3.0)
    return float(a / R)


def transit_duration(period: float, a_rs: float, inc_deg: float, rp_rs: float, b: float) -> float:
    inc = np.radians(inc_deg)
    arg = np.sqrt(max((1 + rp_rs) ** 2 - b**2, 0.0)) / (a_rs * np.sin(inc))
    return float(period / np.pi * np.arcsin(min(arg, 1.0)))


def limb_darkening_for_teff(teff: float | None) -> tuple[float, float]:
    """Coarse quadratic coefficients in the TESS band. Good enough for injection;
    the exact values are stored in truth so nothing is hidden."""
    if teff is None:
        return 0.40, 0.25
    if teff < 4000:
        return 0.30, 0.35
    if teff < 5000:
        return 0.45, 0.20
    if teff < 6000:
        return 0.40, 0.25
    return 0.33, 0.28


def make_truth(period: float, t0: float, depth: float, b: float, star: dict) -> TransitTruth:
    rp_rs = float(np.sqrt(depth))
    logg = star.get("logg") if star.get("logg") is not None else 4.4
    radius = star.get("radius_rsun") if star.get("radius_rsun") is not None else 1.0
    a_rs = a_rs_from_star(period, logg, radius)
    b = float(min(b, a_rs * 0.999))
    inc = float(np.degrees(np.arccos(b / a_rs)))
    u1, u2 = limb_darkening_for_teff(star.get("teff"))
    dur = transit_duration(period, a_rs, inc, rp_rs, b)
    return TransitTruth(period, t0, rp_rs, a_rs, inc, b, u1, u2, dur, depth)


def transit_model(time: np.ndarray, truth: TransitTruth) -> np.ndarray:
    import batman

    p = batman.TransitParams()
    p.t0 = truth.t0
    p.per = truth.period
    p.rp = truth.rp_rs
    p.a = truth.a_rs
    p.inc = truth.inc_deg
    p.ecc = 0.0
    p.w = 90.0
    p.limb_dark = "quadratic"
    p.u = [truth.u1, truth.u2]
    m = batman.TransitModel(p, np.asarray(time, dtype=np.float64))
    return m.light_curve(p)


def inject(time: np.ndarray, flux: np.ndarray, truth: TransitTruth) -> np.ndarray:
    return flux * transit_model(time, truth)


def sample_truth(
    rng: np.random.Generator,
    time: np.ndarray,
    star: dict,
    period_range: tuple[float, float] = (0.5, 20.0),
    depth_range: tuple[float, float] = (2e-4, 2e-2),
    b_max: float = 0.9,
) -> TransitTruth:
    period = float(np.exp(rng.uniform(np.log(period_range[0]), np.log(period_range[1]))))
    depth = float(np.exp(rng.uniform(np.log(depth_range[0]), np.log(depth_range[1]))))
    b = float(rng.uniform(0.0, b_max))
    t0 = float(time.min() + rng.uniform(0.0, period))
    return make_truth(period, t0, depth, b, star)


def point_scatter(flux: np.ndarray) -> float:
    """Robust point-to-point scatter: 1.4826 * MAD of successive differences / sqrt(2)."""
    d = np.diff(flux)
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0))


def in_transit_mask(time: np.ndarray, truth: TransitTruth) -> np.ndarray:
    phase = ((time - truth.t0 + 0.5 * truth.period) % truth.period) - 0.5 * truth.period
    return np.abs(phase) < 0.5 * truth.duration


def transit_count(time: np.ndarray, truth: TransitTruth, min_points: int = 3) -> int:
    """Number of distinct transit events with at least `min_points` in-transit samples."""
    m = in_transit_mask(time, truth)
    if not m.any():
        return 0
    epochs = np.round((time[m] - truth.t0) / truth.period).astype(int)
    _, counts = np.unique(epochs, return_counts=True)
    return int((counts >= min_points).sum())


def expected_snr(time: np.ndarray, flux: np.ndarray, truth: TransitTruth) -> float:
    """depth / sigma * sqrt(N_in_transit), with sigma the point-to-point scatter of
    the light curve before injection. Standard single-event-statistic style estimate."""
    n_in = int(in_transit_mask(time, truth).sum())
    sigma = point_scatter(flux)
    if n_in == 0 or sigma == 0:
        return 0.0
    return float(truth.depth / sigma * np.sqrt(n_in))
