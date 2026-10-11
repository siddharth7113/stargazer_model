"""Transit injection with batman. Each parameter is explicit. The truth record stores it.

Constants
---------
``G_SI = 6.67430e-11``
    Newtonian constant of gravitation, m^3 kg^-1 s^-2, CODATA 2018 [1]_.
``R_SUN = 6.957e8``
    Nominal solar radius, m, IAU 2015 Resolution B3 [2]_.
``DAY = 86400.0``
    Seconds in one day.

Each function that computes a physical quantity gives its equation in a Notes section
and cites its source by number from the list below.

References
----------
.. [1] Tiesinga, E., Mohr, P. J., Newell, D. B., and Taylor, B. N. (2021). CODATA
   recommended values of the fundamental physical constants: 2018. Reviews of Modern
   Physics, 93, 025010. https://doi.org/10.1103/RevModPhys.93.025010
.. [2] Prša, A., Harmanec, P., Torres, G., et al. (2016). Nominal values for selected
   solar and planetary quantities: IAU 2015 Resolution B3. The Astronomical Journal,
   152, 41. https://doi.org/10.3847/0004-6256/152/2/41
.. [3] Seager, S. and Mallén-Ornelas, G. (2003). A unique solution of planet and star
   parameters from an extrasolar planet transit light curve. The Astrophysical
   Journal, 585, 1038. https://doi.org/10.1086/346105
.. [4] Claret, A. (2017). Limb and gravity-darkening coefficients for the TESS
   satellite. Astronomy and Astrophysics, 600, A30.
   https://doi.org/10.1051/0004-6361/201629705
.. [5] Kreidberg, L. (2015). batman: BAsic Transit Model cAlculatioN in Python.
   Publications of the Astronomical Society of the Pacific, 127, 1161.
   https://doi.org/10.1086/683602
.. [6] Jenkins, J. M. (2002). The impact of solar-like variability on the detectability
   of transiting terrestrial planets. The Astrophysical Journal, 575, 493.
   https://doi.org/10.1086/341136
.. [7] Stassun, K. G., Oelkers, R. J., Paegert, M., et al. (2019). The revised TESS
   Input Catalog and Candidate Target List. The Astronomical Journal, 158, 138.
   https://doi.org/10.3847/1538-3881/ab3467
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

G_SI = 6.67430e-11
R_SUN = 6.957e8
DAY = 86400.0


@dataclass(frozen=True)
class TransitTruth:
    """The hidden answer for one injected transit.

    When the benchmark injects a synthetic transit into a light curve, it stores the
    parameters of that transit in this record. The scorer compares a detector's
    prediction with it. The detector never sees it. The record holds the free
    parameters (period, epoch, depth, impact parameter) and the derived geometry
    (radius ratio, semi-major axis, inclination, duration, limb darkening), so a
    reader can rebuild the exact model without the code that drew it.

    Parameters
    ----------
    period : float
        Orbital period, days.
    t0 : float
        Epoch of a transit centre, days, in the light curve's time base.
    rp_rs : float
        Planet to star radius ratio, dimensionless.
    a_rs : float
        Semi-major axis in stellar radii, dimensionless.
    inc_deg : float
        Orbital inclination, degrees. 90 is edge-on.
    b : float
        Impact parameter, dimensionless. 0 is a central transit.
    u1, u2 : float
        Quadratic limb-darkening coefficients, dimensionless.
    duration : float
        Total transit duration T14, days.
    depth : float
        Geometric depth ``rp_rs**2``, dimensionless. Limb darkening makes the observed
        central depth a little deeper.
    """

    period: float
    t0: float
    rp_rs: float
    a_rs: float
    inc_deg: float
    b: float
    u1: float
    u2: float
    duration: float
    depth: float

    def to_dict(self) -> dict:
        """Return a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TransitTruth:
        """Build from a dictionary. Ignore unknown keys."""
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


def _finite(x: object) -> bool:
    return x is not None and np.isfinite(float(x))


def a_rs_from_star(period_days: float, logg_cgs: float, radius_rsun: float) -> float:
    """Return the semi-major axis in stellar radii. Use Kepler's third law.

    Parameters
    ----------
    period_days : float
        Orbital period, days.
    logg_cgs : float
        Surface gravity as log10 of g in cm s^-2.
    radius_rsun : float
        Stellar radius, solar radii.

    Returns
    -------
    float
        ``a / R*``, dimensionless.

    Notes
    -----
    The steps are:

    1. Surface gravity in SI: ``g = 10**logg * 1e-2`` converts cm s^-2 to m s^-2.
       ``logg`` and the radius come from the TESS Input Catalog [7]_.
    2. Stellar mass from gravity and radius: ``M = g * R**2 / G``, with ``G`` from
       CODATA 2018 [1]_.
    3. Kepler's third law for a planet of negligible mass:
       ``a**3 = G * M * P**2 / (4 * pi**2)``.
    4. Divide by the radius: ``a / R``, with the solar radius from IAU 2015 [2]_.

    Steps 2 and 3 combine to ``a / R = (g * R * P**2 / (4 * pi**2))**(1/3) / R``. The
    code keeps the steps separate for clarity. The planet mass is neglected. This is
    accurate to better than 0.1 percent for a planet below a few Jupiter masses.

    Example: the Sun with ``logg = 4.438`` and ``P = 365.25`` days gives ``a / R = 215``,
    which is one astronomical unit in solar radii.
    """
    g = 10.0**logg_cgs * 1e-2  # m s^-2
    r = radius_rsun * R_SUN  # m
    mass = g * r**2 / G_SI  # kg
    p = period_days * DAY  # s
    a = (G_SI * mass * p**2 / (4 * np.pi**2)) ** (1.0 / 3.0)  # m
    return float(a / r)


def transit_duration(period: float, a_rs: float, inc_deg: float, rp_rs: float, b: float) -> float:
    """Return the total transit duration T14, days. Assume a circular orbit.

    Parameters
    ----------
    period : float
        Orbital period, days.
    a_rs : float
        Semi-major axis in stellar radii.
    inc_deg : float
        Inclination, degrees.
    rp_rs : float
        Radius ratio.
    b : float
        Impact parameter.

    Notes
    -----
    T14 is the time from first to fourth contact. For a circular orbit, equation 3
    of Seager and Mallén-Ornelas [3]_ gives:

    ``T14 = (P / pi) * arcsin( sqrt((1 + k)**2 - b**2) / (a_rs * sin(i)) )``

    where ``k = rp_rs``. The term ``(1 + k)**2 - b**2`` is the chord length squared
    across the stellar disc, in stellar radii. If ``b > 1 + k`` the planet does not
    transit; the code clamps the square root to zero, which gives a duration of zero.
    The arcsine argument is clamped to one for very close orbits.

    Example: Earth and Sun with ``a_rs = 215``, ``b = 0``, ``k = 0.0092`` give
    ``T14 = 0.545`` days, which is 13.1 hours.
    """
    inc = np.radians(inc_deg)
    chord = np.sqrt(max((1 + rp_rs) ** 2 - b**2, 0.0))  # stellar radii
    arg = chord / (a_rs * np.sin(inc))
    return float(period / np.pi * np.arcsin(min(arg, 1.0)))


LIMB_DARKENING_TABLE: tuple[tuple[float, float, float], ...] = (
    # (upper Teff bound in K, u1, u2); the last row has no upper bound
    (4000.0, 0.30, 0.35),
    (5000.0, 0.45, 0.20),
    (6000.0, 0.40, 0.25),
    (np.inf, 0.33, 0.28),
)
"""Quadratic limb-darkening coefficients for the TESS band by effective temperature.

A coarse digest of Claret [4]_ at ``logg = 4.5`` and solar metallicity. Each row is
``(teff_upper_bound_K, u1, u2)``. Rows are in ascending order of the bound. Replace this
table to use another law or a finer grid.
"""

SOLAR_TYPE_LIMB_DARKENING: tuple[float, float] = (0.40, 0.25)
"""Fallback coefficients when the effective temperature is unknown."""


def limb_darkening_for_teff(
    teff: float | None,
    table: tuple[tuple[float, float, float], ...] = LIMB_DARKENING_TABLE,
    default: tuple[float, float] = SOLAR_TYPE_LIMB_DARKENING,
) -> tuple[float, float]:
    """Return quadratic limb-darkening coefficients for an effective temperature.

    Parameters
    ----------
    teff : float or None
        Effective temperature, K. None returns ``default``.
    table : tuple of (float, float, float)
        Rows of ``(teff_upper_bound, u1, u2)`` in ascending order of the bound.
    default : tuple of float
        Coefficients to return when ``teff`` is None.

    Returns
    -------
    u1, u2 : float

    Notes
    -----
    The quadratic law is ``I(mu) / I(1) = 1 - u1 * (1 - mu) - u2 * (1 - mu)**2``, where
    ``mu`` is the cosine of the angle between the line of sight and the surface normal.

    The function finds the first row whose upper bound is above ``teff`` with a binary
    search over the bounds. The coefficients change only the transit shape, not the
    period or the epoch. The truth record stores the values used.
    """
    if teff is None:
        return default
    bounds = [row[0] for row in table]
    index = int(np.searchsorted(bounds, teff, side="right"))
    _, u1, u2 = table[min(index, len(table) - 1)]
    return u1, u2


def make_truth(period: float, t0: float, depth: float, b: float, star: dict) -> TransitTruth:
    """Derive the full geometry from the free parameters and the host star.

    Parameters
    ----------
    period : float
        Orbital period, days.
    t0 : float
        Epoch, days.
    depth : float
        Geometric depth, dimensionless.
    b : float
        Impact parameter.
    star : dict
        Must have finite ``logg`` and ``radius_rsun``. ``teff`` selects the limb
        darkening and may be missing.

    Raises
    ------
    ValueError
        If ``logg`` or ``radius_rsun`` is missing or not finite. The function does not
        substitute a default, because a wrong stellar density gives a wrong duration
        and the truth record would then be silently wrong.

    Notes
    -----
    The derived quantities are:

    - ``rp_rs = sqrt(depth)``, because the geometric depth is the area ratio.
    - ``a_rs`` from :func:`a_rs_from_star`.
    - ``b`` is clamped below ``a_rs``, so that the inclination is defined.
    - ``inc = arccos(b / a_rs)``, from the definition ``b = a_rs * cos(i)`` for a
      circular orbit.
    - ``duration`` from :func:`transit_duration`.
    """
    rp_rs = float(np.sqrt(depth))
    logg, radius = star.get("logg"), star.get("radius_rsun")
    if not _finite(logg) or not _finite(radius):
        raise ValueError(f"host needs finite logg and radius_rsun, got logg={logg} radius={radius}")
    a_rs = a_rs_from_star(period, float(logg), float(radius))
    b = float(min(b, a_rs * 0.999))
    inc = float(np.degrees(np.arccos(b / a_rs)))
    u1, u2 = limb_darkening_for_teff(star.get("teff"))
    dur = transit_duration(period, a_rs, inc, rp_rs, b)
    return TransitTruth(period, t0, rp_rs, a_rs, inc, b, u1, u2, dur, depth)


def transit_model(time: np.ndarray, truth: TransitTruth) -> np.ndarray:
    """Return the batman model flux. The flux is one out of transit.

    Notes
    -----
    batman [5]_ integrates the quadratic limb-darkening law over the part
    of the stellar disc that the planet covers. The orbit is circular: eccentricity 0
    and argument of periastron 90 degrees. The model is evaluated at each time stamp
    without exposure-time integration. At 2-minute cadence the smoothing from a finite
    exposure is below one percent of the depth for durations above one hour.
    """
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
    return batman.TransitModel(p, np.asarray(time, dtype=np.float64)).light_curve(p)


def inject(time: np.ndarray, flux: np.ndarray, truth: TransitTruth) -> np.ndarray:
    """Multiply the flux by the transit model.

    Notes
    -----
    The injection is multiplicative: ``flux_out = flux_in * model``. The model is one
    out of transit, so the noise and the stellar variability of the real light curve
    are unchanged there. In transit the noise scales with the model, which is what a
    real transit does.
    """
    return flux * transit_model(time, truth)


def sample_truth(
    rng: np.random.Generator,
    time: np.ndarray,
    star: dict,
    period_range: tuple[float, float] = (0.5, 20.0),
    depth_range: tuple[float, float] = (1e-4, 1e-2),
    b_max: float = 0.9,
) -> TransitTruth:
    """Draw one injection. Period and depth are log-uniform. Impact parameter is uniform.

    Parameters
    ----------
    rng : numpy.random.Generator
        The one generator for the build.
    time : ndarray
        Light-curve time stamps, days.
    star : dict
        Host parameters, see :func:`make_truth`.
    period_range : tuple of float
        Bounds, days.
    depth_range : tuple of float
        Bounds, dimensionless.
    b_max : float
        Upper bound of the impact parameter.

    Notes
    -----
    The draws, in order, are:

    1. ``period = exp(U(log(p_min), log(p_max)))``. Log-uniform gives equal weight to
       each decade, so short and long periods are both well sampled.
    2. ``depth = exp(U(log(d_min), log(d_max)))``, for the same reason.
    3. ``b = U(0, b_max)``. The cap avoids grazing transits with near-zero duration.
    4. ``t0 = t_start + U(0, period)``.

    The order matters for reproducibility. Do not change it without a new episode set
    version.

    The epoch is uniform in the first period after the start of the light curve. A
    long-period draw can give an episode with one transit or no transit. This is
    intentional: the benchmark must contain the cases a periodic search cannot find.
    """
    period = float(np.exp(rng.uniform(np.log(period_range[0]), np.log(period_range[1]))))
    depth = float(np.exp(rng.uniform(np.log(depth_range[0]), np.log(depth_range[1]))))
    b = float(rng.uniform(0.0, b_max))
    t0 = float(time.min() + rng.uniform(0.0, period))
    return make_truth(period, t0, depth, b, star)


def point_scatter(flux: np.ndarray) -> float:
    """Return the robust point-to-point scatter of the flux.

    Notes
    -----
    The estimator is

    ``sigma = 1.4826 * MAD(diff(flux)) / sqrt(2)``

    - ``diff(flux)`` removes slow stellar variability and systematics, so the estimate
      reflects the white-noise level that limits transit detection.
    - ``MAD`` is the median absolute deviation from the median. It is insensitive to
      outliers and to transits already in the data.
    - ``1.4826`` converts a MAD to a standard deviation for a Gaussian distribution.
    - ``sqrt(2)`` corrects for the fact that a difference of two independent values
      has twice the variance of one value.
    """
    d = np.diff(flux)
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0))


def in_transit_mask(time: np.ndarray, truth: TransitTruth) -> np.ndarray:
    """Return a boolean mask. True marks points within half a duration of a transit centre.

    Notes
    -----
    The phase in days, centred on zero, is

    ``phase = ((t - t0 + P/2) mod P) - P/2``

    which lies in ``[-P/2, P/2)``. A point is in transit if ``|phase| < T14 / 2``.
    """
    phase = ((time - truth.t0 + 0.5 * truth.period) % truth.period) - 0.5 * truth.period
    return np.abs(phase) < 0.5 * truth.duration


def transit_count(time: np.ndarray, truth: TransitTruth, min_points: int = 3) -> int:
    """Return the number of transit events that have at least ``min_points`` samples.

    Notes
    -----
    Each in-transit point gets an integer epoch number ``round((t - t0) / P)``. The
    function counts the distinct epoch numbers with at least ``min_points`` points.
    A transit that falls in a data gap has no points and is not counted. A transit at
    the edge of a gap with fewer than ``min_points`` points is not counted either.
    """
    m = in_transit_mask(time, truth)
    if not m.any():
        return 0
    epochs = np.round((time[m] - truth.t0) / truth.period).astype(int)
    _, counts = np.unique(epochs, return_counts=True)
    return int((counts >= min_points).sum())


def expected_snr(time: np.ndarray, flux: np.ndarray, truth: TransitTruth) -> float:
    """Return the expected detection statistic of the injection.

    Parameters
    ----------
    time : ndarray
        Days.
    flux : ndarray
        Flux before injection.
    truth : TransitTruth

    Notes
    -----
    The statistic is

    ``SNR = depth / sigma * sqrt(N_in)``

    - ``depth`` is the geometric depth.
    - ``sigma`` is :func:`point_scatter` of the flux before injection.
    - ``N_in`` is the number of points inside the transit windows, summed over all
      transits in the data.

    This is the matched-filter signal-to-noise ratio of a box of the right depth and
    duration against white noise. It is the same quantity the Kepler pipeline calls
    the multiple event statistic [6]_, up to the box-versus-limb-darkened shape. It is an
    upper bound for a real detector, which also has to find the period.
    """
    n_in = int(in_transit_mask(time, truth).sum())
    sigma = point_scatter(flux)
    if n_in == 0 or sigma == 0:
        return 0.0
    return float(truth.depth / sigma * np.sqrt(n_in))
