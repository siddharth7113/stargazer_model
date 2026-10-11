"""Download and cache TESS 2-minute light curves as ``.npz`` files.

The source is the Science Processing Operations Center (SPOC) pipeline. Each cached
file holds ``time`` in BTJD days, ``flux`` as median-normalised PDCSAP flux,
``flux_err``, and a JSON ``meta`` string. The ``meta`` string holds the TIC ID, the
sector and the stellar parameters from the FITS header.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
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


@dataclass
class LightCurve:
    """A light curve and the metadata that injection needs.

    Parameters
    ----------
    time, flux, flux_err : ndarray
        Arrays of equal length. ``flux`` is normalised to a median of one.
    meta : dict
        TIC, sector, and stellar parameters when available.
    """

    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    meta: dict

    @property
    def span(self) -> float:
        """Total baseline in days."""
        return float(self.time.max() - self.time.min())


class LightCurveCache:
    """A directory cache of SPOC 2-minute light curves. The key is (TIC, sector).

    Parameters
    ----------
    root : Path
        Directory holding ``ticNNN_sSSSS.npz`` files.
    """

    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, tic: int, sector: int) -> Path:
        """Return the cache file path for a target. The file does not have to exist."""
        return self.root / f"tic{tic}_s{sector:04d}.npz"

    def has(self, tic: int, sector: int) -> bool:
        """Return True if the target is cached."""
        return self.path(tic, sector).exists()

    def load(self, tic: int, sector: int) -> LightCurve:
        """Load a cached light curve.

        Raises
        ------
        FileNotFoundError
            If the target has not been fetched.
        """
        with np.load(self.path(tic, sector)) as z:
            meta = json.loads(str(z["meta"])) if "meta" in z else {}
            return LightCurve(z["time"], z["flux"], z["flux_err"], meta)

    def fetch(self, tic: int, sector: int, overwrite: bool = False) -> Path:
        """Download one SPOC 120 s light curve from MAST and cache it.

        Parameters
        ----------
        tic, sector : int
            Target and TESS sector.
        overwrite : bool
            Download again if the file exists.

        Returns
        -------
        Path
            The cache file.

        Raises
        ------
        FileNotFoundError
            If MAST has no SPOC 120 s product for the target.
        """
        out = self.path(tic, sector)
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
