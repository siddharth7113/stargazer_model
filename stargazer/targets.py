"""Select host stars with no known transit signal for injection.

Steps:

1. Get the list of SPOC 120 s targets in a sector from MAST.
2. Draw a seeded random subset.
3. Get the TESS Input Catalog (TIC) parameters of the subset.
4. Apply brightness and dwarf cuts.
5. Remove each star in the ExoFOP TESS Object of Interest (TOI) list.

The output records the TOI CSV hash and download time. This makes the exclusion
auditable.
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
    with urllib.request.urlopen(url, timeout=120) as r:
        raw = r.read()
    tics = set()
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8", errors="replace"))):
        v = row.get("TIC ID") or row.get("TIC")
        if v:
            try:
                tics.add(int(float(v)))
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


def _cell(row, col: str, key: str):
    v = row[col]
    try:
        if v is None or (hasattr(v, "mask") and v.mask):
            return None
        return str(v) if key == "lumclass" else float(v)
    except Exception:
        return None


def tic_params(tics: list[int]) -> dict[int, dict]:
    """Fetch TIC catalogue parameters in batches of 500."""
    from astroquery.mast import Catalogs

    out = {}
    for i in range(0, len(tics), 500):
        tab = Catalogs.query_criteria(catalog="Tic", ID=[str(t) for t in tics[i : i + 500]])
        for row in tab:
            out[int(row["ID"])] = {key: _cell(row, col, key) for col, key in TIC_COLUMNS}
    return out


def select_targets(
    sector: int,
    n: int,
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
    n : int
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
    candidates = [t for t in all_tics if t not in toi_tics]
    pool = [int(t) for t in rng.choice(candidates, size=min(len(candidates), n * oversample), replace=False)]
    params = tic_params(pool)
    chosen = []
    rejected = {"no_params": 0, "tmag": 0, "teff": 0, "radius": 0, "lumclass": 0}
    for t in pool:
        p = params.get(t)
        if not p or any(p.get(k) is None for k in ("teff", "radius_rsun", "tmag")):
            rejected["no_params"] += 1
        elif p["tmag"] > tmag_max:
            rejected["tmag"] += 1
        elif not (teff_range[0] <= p["teff"] <= teff_range[1]):
            rejected["teff"] += 1
        elif p["radius_rsun"] > radius_max:
            rejected["radius"] += 1
        elif p.get("lumclass") not in (None, "DWARF"):
            rejected["lumclass"] += 1
        else:
            chosen.append({"tic": t, "sector": sector, **p})
            if len(chosen) >= n:
                break
    return {
        "created": utc_timestamp(),
        "sector": sector,
        "seed": seed,
        "n_requested": n,
        "n_selected": len(chosen),
        "cuts": {"tmag_max": tmag_max, "teff_range": list(teff_range), "radius_max": radius_max},
        "n_two_minute_in_sector": len(all_tics),
        "n_excluded_by_toi": len(all_tics) - len(candidates),
        "toi_csv": {"url": TOI_URL, "sha256": toi_hash, "downloaded": toi_time, "n_tics": len(toi_tics)},
        "rejected": rejected,
        "targets": chosen,
    }


def save_targets(d: dict, path: Path) -> None:
    """Write a targets dictionary as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(d, indent=2))


def load_targets(path: Path) -> dict:
    """Read a targets JSON."""
    return json.loads(Path(path).read_text())
