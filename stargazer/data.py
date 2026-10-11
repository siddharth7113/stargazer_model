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
    """One star's brightness over time, with the stellar parameters.

    A light curve is a time series of the brightness of one star. TESS measures a
    star every two minutes for about 27 days, one sector. The benchmark injects
    synthetic transits into light curves of stars with no known planet. The stellar
    radius and surface gravity in ``meta`` set the transit geometry.

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
    """Download TESS light curves once and read them from disk after that.

    The cache is a directory of ``.npz`` files. The key of a file is the TESS Input
    Catalog (TIC) identifier of the star and the sector number. ``fetch`` downloads
    a light curve from the Mikulski Archive for Space Telescopes (MAST) if the file
    does not exist. ``load`` reads a cached file. A manifest of file hashes makes the
    cache verifiable.

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
        with np.load(self.path(tic, sector)) as archive:
            meta = json.loads(str(archive["meta"])) if "meta" in archive else {}
            return LightCurve(archive["time"], archive["flux"], archive["flux_err"], meta)

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
        for header_key, name in META_KEYS.items():
            value = lc.meta.get(header_key)
            meta[name] = None if value is None else float(value)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            time=np.asarray(lc.time.value, dtype=np.float64),
            flux=np.asarray(lc.flux.value, dtype=np.float64),
            flux_err=np.asarray(lc.flux_err.value, dtype=np.float64),
            meta=json.dumps(meta),
        )
        return out
