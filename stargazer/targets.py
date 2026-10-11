"""Select host stars with no known transit signal for injection.

Pipeline: MAST observation list for a sector (SPOC 120 s) -> seeded random subset ->
TIC catalogue parameters -> dwarf and brightness cuts -> exclude anything in the ExoFOP
TOI list. The TOI CSV hash and download time are recorded so the exclusion is auditable.
"""

from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import numpy as np

from .provenance import sha256_bytes, utc_now

TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"


def download_toi_tics(url: str = TOI_URL) -> tuple[set[int], str]:
    """Return the set of TIC IDs with a TOI and the SHA-256 of the CSV as downloaded."""
    with urllib.request.urlopen(url, timeout=120) as r:
        raw = r.read()
    digest = sha256_bytes(raw)
    import csv

    tics = set()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8", errors="replace")))
    for row in reader:
        v = row.get("TIC ID") or row.get("TIC")
        if v:
            try:
                tics.add(int(float(v)))
            except ValueError:
                pass
    return tics, digest


def sector_two_minute_tics(sector: int) -> list[int]:
    """All TIC IDs with a SPOC 120 s light curve in `sector`, from MAST."""
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


def tic_params(tics: list[int]) -> dict[int, dict]:
    from astroquery.mast import Catalogs

    out = {}
    for i in range(0, len(tics), 500):
        chunk = [str(t) for t in tics[i : i + 500]]
        tab = Catalogs.query_criteria(catalog="Tic", ID=chunk)
        for row in tab:
            d = {}
            for col, key in [("Tmag", "tmag"), ("Teff", "teff"), ("rad", "radius_rsun"),
                             ("logg", "logg"), ("mass", "mass_msun"), ("lumclass", "lumclass"),
                             ("ra", "ra"), ("dec", "dec")]:
                v = row[col]
                try:
                    d[key] = None if (v is None or (hasattr(v, "mask") and v.mask)) else (
                        str(v) if key == "lumclass" else float(v))
                except Exception:
                    d[key] = None
            out[int(row["ID"])] = d
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
    rng = np.random.default_rng(seed)
    all_tics = sector_two_minute_tics(sector)
    toi_tics, toi_hash = download_toi_tics()
    toi_time = utc_now()
    candidates = [t for t in all_tics if t not in toi_tics]
    pool = list(rng.choice(candidates, size=min(len(candidates), n * oversample), replace=False))
    pool = [int(t) for t in pool]
    params = tic_params(pool)
    chosen = []
    rejected = {"no_params": 0, "tmag": 0, "teff": 0, "radius": 0, "lumclass": 0}
    for t in pool:
        p = params.get(t)
        if not p or p.get("teff") is None or p.get("radius_rsun") is None or p.get("tmag") is None:
            rejected["no_params"] += 1
            continue
        if p["tmag"] > tmag_max:
            rejected["tmag"] += 1
            continue
        if not (teff_range[0] <= p["teff"] <= teff_range[1]):
            rejected["teff"] += 1
            continue
        if p["radius_rsun"] > radius_max:
            rejected["radius"] += 1
            continue
        if p.get("lumclass") not in (None, "DWARF"):
            rejected["lumclass"] += 1
            continue
        chosen.append({"tic": t, "sector": sector, **p})
        if len(chosen) >= n:
            break
    return {
        "created": utc_now(),
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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(d, indent=2))


def load_targets(path: Path) -> dict:
    return json.loads(Path(path).read_text())
