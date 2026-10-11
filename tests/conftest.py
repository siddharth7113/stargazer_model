import numpy as np
import pytest


def synthetic_lc(seed=0, days=27.0, cadence_min=2.0, sigma=1e-3):
    rng = np.random.default_rng(seed)
    time = np.arange(0.0, days, cadence_min / 1440.0) + 1000.0  # BTJD-like offset
    # a 1-day gap in the middle, like a TESS orbit gap
    keep = ~((time - time.min() > 13.0) & (time - time.min() < 14.0))
    time = time[keep]
    flux = 1.0 + rng.normal(0.0, sigma, size=time.size)
    return time, flux, np.full_like(flux, sigma)


@pytest.fixture
def lc():
    return synthetic_lc()


@pytest.fixture
def star():
    return {"teff": 5500.0, "logg": 4.45, "radius_rsun": 0.95, "tmag": 10.0, "mass_msun": 0.95}
