"""Episode sets are the unit of evaluation.

An episode file ``episodes/ep_XXXX.npz`` holds only relative time, flux and flux
error. It holds no TIC ID, no sector, no coordinates, no stellar parameters and no
injection parameters. The file ``truth.json`` holds those values. Only the scorer
reads the truth file.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import __version__
from .data import LightCurveCache
from .inject import expected_snr, inject, point_scatter, sample_truth, transit_count
from .provenance import git_commit, sha256_file, utc_now, verify_manifest, write_manifest


@dataclass(frozen=True)
class BuildConfig:
    """The values that determine an episode set for a given cache and targets file.

    Parameters
    ----------
    n : int
        Number of episodes.
    seed : int
        Seed for the one random generator that the build uses.
    null_frac : float
        Fraction of episodes with no injection.
    period_range, depth_range : tuple of float
        Log-uniform sampling bounds, days and fractional depth.
    """

    n: int
    seed: int
    null_frac: float = 0.3
    period_range: tuple[float, float] = (0.5, 20.0)
    depth_range: tuple[float, float] = (2e-4, 2e-2)


class EpisodeSet:
    """A built episode set on disk.

    Parameters
    ----------
    root : Path
        Directory containing ``episodes/``, ``truth.json``, ``config.json`` and
        ``manifest.json``.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        if not (self.root / "manifest.json").exists():
            raise FileNotFoundError(f"{self.root} is not a built episode set")

    @property
    def ids(self) -> list[str]:
        """Sorted episode identifiers."""
        return sorted(p.stem for p in (self.root / "episodes").glob("ep_*.npz"))

    @property
    def truth(self) -> dict:
        """Full truth dictionary keyed by episode id. Not for the agent."""
        return json.loads((self.root / "truth.json").read_text())

    @property
    def config(self) -> dict:
        """The build configuration and provenance."""
        return json.loads((self.root / "config.json").read_text())

    @property
    def hash(self) -> str:
        """SHA-256 of the manifest. This hash identifies the exact set."""
        return sha256_file(self.root / "manifest.json")

    def load(self, ep_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(time, flux, flux_err)`` for one episode."""
        with np.load(self.root / "episodes" / f"{ep_id}.npz") as z:
            return z["time"], z["flux"], z["flux_err"]

    def __iter__(self) -> Iterator[str]:
        return iter(self.ids)

    def __len__(self) -> int:
        return len(self.ids)

    def verify(self) -> list[str]:
        """Recompute each hash. An empty list means the set is intact."""
        return verify_manifest(self.root)

    @classmethod
    def build(
        cls,
        targets: dict,
        cache: LightCurveCache,
        out_dir: Path,
        cfg: BuildConfig,
        targets_path: Path | None = None,
    ) -> EpisodeSet:
        """Build a new episode set by injection into cached light curves.

        Parameters
        ----------
        targets : dict
            Output of :func:`stargazer.targets.select_targets`.
        cache : LightCurveCache
            Must already contain the targets' light curves.
        out_dir : Path
            Destination directory. The function creates it if needed.
        cfg : BuildConfig
        targets_path : Path, optional
            The function records this path and its hash in ``config.json``.

        Raises
        ------
        RuntimeError
            If none of the targets are cached.
        """
        rng = np.random.default_rng(cfg.seed)
        out_dir = Path(out_dir)
        ep_dir = out_dir / "episodes"
        ep_dir.mkdir(parents=True, exist_ok=True)

        hosts = [t for t in targets["targets"] if cache.has(t["tic"], t["sector"])]
        if not hosts:
            raise RuntimeError("no cached light curves for any target; run fetch first")

        truth = {}
        for i in range(cfg.n):
            star = hosts[int(rng.integers(len(hosts)))]
            lc = cache.load(star["tic"], star["sector"])
            t_offset = float(lc.time.min())
            flux = lc.flux
            is_null = bool(rng.uniform() < cfg.null_frac)
            rec = {
                "kind": "null" if is_null else "transit",
                "tic": star["tic"],
                "sector": star["sector"],
                "t_offset": t_offset,
                "n_points": len(lc.time),
                "star": {k: star.get(k) for k in ("teff", "logg", "radius_rsun", "tmag", "mass_msun")},
                "point_scatter": point_scatter(flux),
            }
            if not is_null:
                tr = sample_truth(
                    rng, lc.time, star, period_range=cfg.period_range, depth_range=cfg.depth_range
                )
                rec["snr"] = expected_snr(lc.time, flux, tr)
                rec["n_transits"] = transit_count(lc.time, tr)
                flux = inject(lc.time, flux, tr)
                d = tr.to_dict()
                d["t0"] = tr.t0 - t_offset
                rec["transit"] = d
            ep_id = f"ep_{i:04d}"
            np.savez_compressed(
                ep_dir / f"{ep_id}.npz",
                time=(lc.time - t_offset).astype(np.float64),
                flux=flux.astype(np.float64),
                flux_err=lc.flux_err.astype(np.float64),
            )
            truth[ep_id] = rec

        (out_dir / "truth.json").write_text(json.dumps(truth, indent=1, sort_keys=True))
        config = {
            "created": utc_now(),
            "stargazer_version": __version__,
            "git_commit": git_commit(),
            "build": {
                **cfg.__dict__,
                "period_range": list(cfg.period_range),
                "depth_range": list(cfg.depth_range),
            },
            "targets_file": str(targets_path) if targets_path else None,
            "targets_sha256": sha256_file(targets_path) if targets_path else None,
            "n_available_hosts": len(hosts),
        }
        (out_dir / "config.json").write_text(json.dumps(config, indent=2))
        write_manifest(out_dir)
        return cls(out_dir)
