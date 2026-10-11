"""Shared fixtures: synthetic light curves, a host star, an injection and a fake light-curve cache.

The light curves are white noise at TESS cadence with one 1-day gap, so each test knows
the exact noise level and gap position. No test reads the network or the real cache.
"""

import json

import numpy as np
import pytest

from stargazer.data import LightCurveCache
from stargazer.inject import inject, make_truth


def synthetic_lc(seed=0, days=27.0, cadence_min=2.0, sigma=1e-3):
    """Return (time, flux, flux_err).

    Gaussian noise of width sigma, a gap from day 13 to 14, and a BTJD-like time offset.
    """
    rng = np.random.default_rng(seed)
    time = np.arange(0.0, days, cadence_min / 1440.0) + 1000.0
    keep = ~((time - time.min() > 13.0) & (time - time.min() < 14.0))
    time = time[keep]
    flux = 1.0 + rng.normal(0.0, sigma, size=time.size)
    return time, flux, np.full_like(flux, sigma)


@pytest.fixture
def lc():
    """One synthetic light curve with seed 0."""
    return synthetic_lc()


@pytest.fixture
def star():
    """A Sun-like host with every parameter the injector needs."""
    return {"teff": 5500.0, "logg": 4.45, "radius_rsun": 0.95, "tmag": 10.0, "mass_msun": 0.95}


@pytest.fixture
def injected(lc, star):
    """A strong injection in relative time: period 3 days, epoch 1 day, depth 1 percent, with its truth."""
    time, flux, flux_err = lc
    relative = time - time.min()
    truth = make_truth(period=3.0, t0=1.0, depth=0.01, b=0.2, star=star)
    return relative, inject(relative, flux, truth), flux_err, truth


@pytest.fixture
def fake_cache(tmp_path, star):
    """Three synthetic hosts (TIC 1000 to 1002, sector 98) in a cache, plus the targets dict and file."""
    cache = LightCurveCache(tmp_path / "cache")
    cache.root.mkdir()
    targets = {"targets": []}
    for index in range(3):
        tic, sector = 1000 + index, 98
        time, flux, flux_err = synthetic_lc(seed=index)
        np.savez_compressed(
            cache.path(tic, sector),
            time=time,
            flux=flux,
            flux_err=flux_err,
            meta=json.dumps({"tic": tic, "sector": sector}),
        )
        targets["targets"].append({"tic": tic, "sector": sector, **star})
    targets_path = tmp_path / "targets.json"
    targets_path.write_text(json.dumps(targets))
    return cache, targets, targets_path
