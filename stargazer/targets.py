"""Select host stars with no known transit signal for injection.

Steps:

1. Get the list of TESS [1]_ targets with SPOC [2]_ 120 s data in a sector from MAST.
2. Draw a seeded random subset.
3. Get the TESS Input Catalog (TIC) [3]_ parameters of the subset.
4. Apply brightness and dwarf cuts.
5. Remove each star in the ExoFOP TESS Object of Interest (TOI) [4]_ list.

The output records the TOI CSV hash and download time. This makes the exclusion
auditable.

References
----------
.. [1] Ricker, G. R., Winn, J. N., Vanderspek, R., et al. (2015). Transiting Exoplanet
   Survey Satellite (TESS). Journal of Astronomical Telescopes, Instruments, and
   Systems, 1, 014003. https://doi.org/10.1117/1.JATIS.1.1.014003
.. [2] Jenkins, J. M., Twicken, J. D., McCauliff, S., et al. (2016). The TESS Science
   Processing Operations Center. Proceedings of SPIE, 9913, 99133E.
   https://doi.org/10.1117/12.2233418
.. [3] Stassun, K. G., Oelkers, R. J., Paegert, M., et al. (2019). The revised TESS
   Input Catalog and Candidate Target List. The Astronomical Journal, 158, 138.
   https://doi.org/10.3847/1538-3881/ab3467
.. [4] Guerrero, N. M., Seager, S., Huang, C. X., et al. (2021). The TESS Objects of
   Interest Catalog from the TESS Prime Mission. The Astrophysical Journal
   Supplement Series, 254, 39. https://doi.org/10.3847/1538-4365/abefe1
"""

from __future__ import annotations

import csv
import io
import json
import urllib.request
from pathlib import Path

import numpy as np

from stargazer.provenance import Digest, utc_timestamp

TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
TIC_COLUMNS = [
    ("Tmag", "tmag"),
    ("Teff", "teff"),
    ("rad", "radius_rsun"),
    ("logg", "logg"),
    ("mass", "mass_msun"),
    ("lumclass", "lumclass"),
    ("ra", "ra"),
    ("dec", "dec"),
]


def download_toi_tics(url: str = TOI_URL) -> tuple[set[int], str]:
    """Download the TOI table.

    Returns
    -------
    tics : set of int
        Every TIC ID with a TOI.
    sha256 : str
        Hash of the CSV exactly as downloaded.
    """
    with urllib.request.urlopen(url, timeout=120) as response:
        raw = response.read()
    tics = set()
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8", errors="replace"))):
        value = row.get("TIC ID") or row.get("TIC")
        if value:
            try:
                tics.add(int(float(value)))
            except ValueError:
                continue
    return tics, Digest.of_bytes(raw).hex


def sector_two_minute_tics(sector: int) -> list[int]:
    """Return all TIC IDs that have a SPOC 120 s light curve in a sector. The source is MAST."""
    from astroquery.mast import Observations

    obs = Observations.query_criteria(
        obs_collection="TESS",
        dataproduct_type="timeseries",
        provenance_name="SPOC",
        sequence_number=sector,
        t_exptime=[119, 121],
    )
    tics = set()
    for name in obs["target_name"]:
        try:
            tics.add(int(name))
        except (TypeError, ValueError):
            continue
    return sorted(tics)


def _cell(row, col: str, key: str) -> float | str | None:
    """Read one catalogue cell. A masked, empty or non-finite value becomes None."""
    value = row[col]
    try:
        if value is None or (hasattr(value, "mask") and value.mask):
            return None
        if key == "lumclass":
            return str(value)
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def tic_params(tics: list[int]) -> dict[int, dict]:
    """Fetch TIC catalogue parameters in batches of 500."""
    from astroquery.mast import Catalogs

    out = {}
    for start in range(0, len(tics), 500):
        tab = Catalogs.query_criteria(catalog="Tic", ID=[str(tic) for tic in tics[start : start + 500]])
        for row in tab:
            out[int(row["ID"])] = {key: _cell(row, col, key) for col, key in TIC_COLUMNS}
    return out


def select_targets(
    sector: int,
    n_targets: int,
    seed: int,
    tmag_max: float = 12.0,
    teff_range: tuple[float, float] = (3000.0, 6500.0),
    radius_max: float = 1.5,
    oversample: int = 4,
) -> dict:
    """Pick up to ``n`` quiet dwarf hosts with no TOI in a sector.

    Parameters
    ----------
    sector : int
    n_targets : int
        Number wanted.
    seed : int
        Seed for the subset drawn from the sector list.
    tmag_max : float
        Brightness cut.
    teff_range : tuple of float
        Effective temperature window in K.
    radius_max : float
        Radius cut in solar radii, to keep dwarfs.
    oversample : int
        Number of stars to draw before the cuts, as a multiple of ``n``.

    Returns
    -------
    dict
        The selected targets, and each count and hash needed to audit the selection.
    """
    rng = np.random.default_rng(seed)
    all_tics = sector_two_minute_tics(sector)
    toi_tics, toi_hash = download_toi_tics()
    toi_time = utc_timestamp()
    candidates = [tic for tic in all_tics if tic not in toi_tics]
    pool = [
        int(tic)
        for tic in rng.choice(candidates, size=min(len(candidates), n_targets * oversample), replace=False)
    ]
    params = tic_params(pool)
    chosen = []
    rejected = {"no_params": 0, "tmag": 0, "teff": 0, "radius": 0, "lumclass": 0}
    for tic in pool:
        params_of = params.get(tic)
        if not params_of or any(
            params_of.get(key) is None for key in ("teff", "radius_rsun", "tmag", "logg")
        ):
            rejected["no_params"] += 1
        elif params_of["tmag"] > tmag_max:
            rejected["tmag"] += 1
        elif not (teff_range[0] <= params_of["teff"] <= teff_range[1]):
            rejected["teff"] += 1
        elif params_of["radius_rsun"] > radius_max:
            rejected["radius"] += 1
        elif params_of.get("lumclass") not in (None, "DWARF"):
            rejected["lumclass"] += 1
        else:
            chosen.append({"tic": tic, "sector": sector, **params_of})
            if len(chosen) >= n_targets:
                break
    return {
        "created": utc_timestamp(),
        "sector": sector,
        "seed": seed,
        "n_requested": n_targets,
        "n_selected": len(chosen),
        "cuts": {"tmag_max": tmag_max, "teff_range": list(teff_range), "radius_max": radius_max},
        "n_two_minute_in_sector": len(all_tics),
        "n_excluded_by_toi": len(all_tics) - len(candidates),
        "toi_csv": {"url": TOI_URL, "sha256": toi_hash, "downloaded": toi_time, "n_tics": len(toi_tics)},
        "rejected": rejected,
        "targets": chosen,
    }


def save_targets(targets: dict, path: Path) -> None:
    """Write a targets dictionary as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(targets, indent=2))


def load_targets(path: Path) -> dict:
    """Read a targets JSON."""
    return json.loads(Path(path).read_text())
