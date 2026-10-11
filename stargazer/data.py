"""Download and cache TESS 2-minute SPOC light curves as compact .npz files.

Each cached file holds: time (BTJD days), flux (median-normalised PDCSAP), flux_err,
and a JSON `meta` string with TIC, sector and stellar parameters from the FITS header.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

META_KEYS = {
    "RADIUS": "radius_rsun",
    "TEFF": "teff",
    "LOGG": "logg",
    "TESSMAG": "tmag",
    "RA_OBJ": "ra",
    "DEC_OBJ": "dec",
}


def cache_path(cache_dir: Path, tic: int, sector: int) -> Path:
    return Path(cache_dir) / f"tic{tic}_s{sector:04d}.npz"


def fetch_lightcurve(tic: int, sector: int, cache_dir: Path, overwrite: bool = False) -> Path:
    """Download one SPOC 2-minute light curve and cache it. Returns the cache path."""
    out = cache_path(cache_dir, tic, sector)
    if out.exists() and not overwrite:
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import lightkurve as lk

        sr = lk.search_lightcurve(f"TIC {tic}", sector=sector, author="SPOC", exptime=120)
        if len(sr) == 0:
            raise FileNotFoundError(f"no SPOC 120s light curve for TIC {tic} sector {sector}")
        lc = sr[0].download(flux_column="pdcsap_flux", quality_bitmask="default")
    lc = lc.remove_nans().normalize()
    meta = {"tic": int(tic), "sector": int(sector), "author": "SPOC", "exptime_s": 120}
    for k, name in META_KEYS.items():
        v = lc.meta.get(k)
        meta[name] = None if v is None else float(v)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        time=np.asarray(lc.time.value, dtype=np.float64),
        flux=np.asarray(lc.flux.value, dtype=np.float64),
        flux_err=np.asarray(lc.flux_err.value, dtype=np.float64),
        meta=json.dumps(meta),
    )
    return out


def load_lightcurve(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    with np.load(path) as z:
        meta = json.loads(str(z["meta"])) if "meta" in z else {}
        return z["time"], z["flux"], z["flux_err"], meta
