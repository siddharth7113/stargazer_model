"""Build an episode set.

Agent-facing files (`episodes/ep_XXXX.npz`) contain only relative time, flux, flux_err.
No TIC, no sector, no coordinates, no stellar parameters, no injection parameters.
`truth.json` holds everything else and is read only by the scorer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import __version__
from .data import cache_path, load_lightcurve
from .inject import expected_snr, inject, sample_truth, transit_count
from .provenance import git_commit, sha256_file, utc_now, write_manifest


def build_episode_set(
    targets: dict,
    cache_dir: Path,
    out_dir: Path,
    n: int,
    seed: int,
    null_frac: float = 0.3,
    period_range: tuple[float, float] = (0.5, 20.0),
    depth_range: tuple[float, float] = (2e-4, 2e-2),
    targets_path: Path | None = None,
) -> Path:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    ep_dir = out_dir / "episodes"
    ep_dir.mkdir(parents=True, exist_ok=True)

    available = [t for t in targets["targets"] if cache_path(cache_dir, t["tic"], t["sector"]).exists()]
    if not available:
        raise RuntimeError("no cached light curves for any target; run `stargazer fetch` first")

    truth = {}
    for i in range(n):
        star = available[int(rng.integers(len(available)))]
        time, flux, flux_err, meta = load_lightcurve(cache_path(cache_dir, star["tic"], star["sector"]))
        t_offset = float(time.min())
        is_null = bool(rng.uniform() < null_frac)
        rec = {
            "kind": "null" if is_null else "transit",
            "tic": star["tic"],
            "sector": star["sector"],
            "t_offset": t_offset,
            "n_points": int(len(time)),
            "star": {k: star.get(k) for k in ("teff", "logg", "radius_rsun", "tmag", "mass_msun")},
            "point_scatter": None,
        }
        from .inject import point_scatter

        rec["point_scatter"] = point_scatter(flux)
        if not is_null:
            tr = sample_truth(rng, time, star, period_range=period_range, depth_range=depth_range)
            rec["snr"] = expected_snr(time, flux, tr)
            rec["n_transits"] = transit_count(time, tr)
            flux = inject(time, flux, tr)
            d = tr.to_dict()
            d["t0"] = tr.t0 - t_offset  # truth in the same relative time base as the episode
            rec["transit"] = d
        ep_id = f"ep_{i:04d}"
        np.savez_compressed(
            ep_dir / f"{ep_id}.npz",
            time=(time - t_offset).astype(np.float64),
            flux=flux.astype(np.float64),
            flux_err=flux_err.astype(np.float64),
        )
        truth[ep_id] = rec

    (out_dir / "truth.json").write_text(json.dumps(truth, indent=1, sort_keys=True))
    config = {
        "created": utc_now(),
        "stargazer_version": __version__,
        "git_commit": git_commit(),
        "seed": seed,
        "n": n,
        "null_frac": null_frac,
        "period_range": list(period_range),
        "depth_range": list(depth_range),
        "targets_file": str(targets_path) if targets_path else None,
        "targets_sha256": sha256_file(targets_path) if targets_path else None,
        "n_available_hosts": len(available),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    write_manifest(out_dir)
    return out_dir


def episode_set_hash(out_dir: Path) -> str:
    return sha256_file(Path(out_dir) / "manifest.json")


def load_episode(out_dir: Path, ep_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(Path(out_dir) / "episodes" / f"{ep_id}.npz") as z:
        return z["time"], z["flux"], z["flux_err"]


def load_truth(out_dir: Path) -> dict:
    return json.loads((Path(out_dir) / "truth.json").read_text())


def list_episodes(out_dir: Path) -> list[str]:
    return sorted(p.stem for p in (Path(out_dir) / "episodes").glob("ep_*.npz"))
