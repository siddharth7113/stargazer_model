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

from stargazer import __version__
from stargazer.context import BTJD_OFFSET
from stargazer.data import LightCurveCache
from stargazer.inject import expected_snr, inject, point_scatter, sample_truth, transit_count
from stargazer.provenance import Digest, GitState, Manifest, utc_timestamp


@dataclass(frozen=True)
class BuildConfig:
    """The settings that define how an episode set is built.

    Given the same cache of light curves, the same targets file and the same
    ``BuildConfig``, :meth:`EpisodeSet.build` produces identical files. The config is
    stored in the episode set, so a reader can rebuild it.

    Parameters
    ----------
    n_episodes : int
        Number of episodes.
    seed : int
        Seed for the one random generator that the build uses.
    null_frac : float
        Fraction of episodes with no injection.
    period_range, depth_range : tuple of float
        Log-uniform sampling bounds, days and fractional depth.
    """

    n_episodes: int
    seed: int
    null_frac: float = 0.3
    period_range: tuple[float, float] = (0.5, 20.0)
    depth_range: tuple[float, float] = (1e-4, 1e-2)


def _has_finite(record: dict, *keys: str) -> bool:
    return all(record.get(key) is not None and np.isfinite(float(record[key])) for key in keys)


EPHEMERIS_MIN_SNR = 5.0
"""Box-fit depth significance below which a catalogue ephemeris with an expected
signal-to-noise of ``EPHEMERIS_CHECK_SNR`` or more is judged absent: the catalogue says
the transit should be plain and the data at that period show nothing. A TOI ephemeris
from 2019 to 2021 can be hours off by 2025, or its period wrong; on sector 98 two of 37
were (LOGBOOK 2026-11-01)."""

EPHEMERIS_CHECK_SNR = 10.0
"""Expected signal-to-noise above which the absence test applies. A fainter planet is
kept on the catalogue's word, as an injected planet below the findable line is kept."""


def verify_ephemeris(time: np.ndarray, flux: np.ndarray, period: float, t0: float, duration: float) -> dict:
    """Fit a box at a fixed period and return the measured depth, its significance and the epoch shift.

    Parameters
    ----------
    time, flux : ndarray
        The light curve in the episode's time base.
    period, t0, duration : float
        The catalogue ephemeris, days.

    Returns
    -------
    dict
        ``measured_depth_ppm``, ``depth_snr``, ``epoch_shift_days`` (the fitted epoch
        minus ``t0``, wrapped to half a period) and ``t0_fitted``.

    Notes
    -----
    The flux is filtered with the window of the box search at six durations and
    binned to ten minutes, then the Box Least Squares of astropy evaluates the one
    period with the catalogue duration and an epoch grid fifty times finer than the
    duration; the epoch with the most power is the fitted epoch. The depth
    significance is astropy's ``depth_snr``, the depth over its error from the bin
    errors. The filter removes part of a deep transit's depth, so the measured depth
    is a lower bound and the significance, not the depth, decides presence.
    """
    from astropy.timeseries import BoxLeastSquares

    from stargazer.baseline import bin_lc, detrend

    work_time, work_flux = detrend(time, flux, max(0.75, 6.0 * duration))
    bin_time, bin_flux, bin_err = bin_lc(work_time, work_flux, 10.0)
    result = BoxLeastSquares(bin_time, bin_flux, bin_err).power(
        np.array([period]), np.array([duration]), oversample=50
    )
    t0_fitted = float(result.transit_time[0])
    shift = ((t0_fitted - t0 + 0.5 * period) % period) - 0.5 * period
    return {
        "measured_depth_ppm": round(float(result.depth[0]) * 1e6, 1),
        "depth_snr": round(float(result.depth_snr[0]), 1),
        "epoch_shift_days": round(float(shift), 4),
        "t0_fitted": t0_fitted,
    }


def _transits_and_snr(
    time: np.ndarray, flux: np.ndarray, period: float, t0: float, duration: float, depth: float
) -> tuple[int, float]:
    """Return the number of transits with at least three points and the box signal-to-noise at an epoch."""
    phase = ((time - t0 + 0.5 * period) % period) - 0.5 * period
    in_transit = np.abs(phase) < 0.5 * duration
    epochs = np.round((time[in_transit] - t0) / period).astype(int)
    _, counts = np.unique(epochs, return_counts=True)
    scatter = point_scatter(flux)
    snr = float(depth / scatter * np.sqrt(int(in_transit.sum()))) if scatter > 0 else 0.0
    return int(np.sum(counts >= 3)), snr


def catalogue_transit(
    time: np.ndarray, flux: np.ndarray, planet: dict, t_offset: float
) -> tuple[dict, int, float]:
    """Return the truth record, the transit count and the expected signal-to-noise of a catalogue planet.

    Parameters
    ----------
    time : ndarray
        BTJD days of the light curve.
    flux : ndarray
        Normalised flux, as observed.
    planet : dict
        A TOI entry: ``period`` (days), ``epoch_bjd``, ``duration_hours``, ``depth_ppm``.
    t_offset : float
        BTJD of the first point; the record's ``t0`` is in days from it.

    Notes
    -----
    The epoch is moved by whole periods to the first transit centre at or after the
    start of the data, so ``t0`` lies in the episode's window as an injected epoch
    does. The count is the number of transit centres with at least three points
    within half a duration, as :func:`stargazer.inject.transit_count` counts. The
    signal-to-noise is the box statistic of :func:`stargazer.inject.expected_snr`,
    ``depth / scatter * sqrt(n_in)``, with the catalogue depth and duration; on a
    real transit it is an estimate, because the light curve already holds the
    transit and the scatter includes it.

    The ephemeris is then checked against the data with :func:`verify_ephemeris`,
    and the record carries the result under ``ephemeris``: ``no_transit`` when the
    catalogue ephemeris puts no transit in the data, where nothing can be checked;
    ``confirmed`` when the fitted epoch is within one duration of the catalogue's;
    ``shifted`` when it is further and the dip is significant, in which case ``t0``
    becomes the fitted epoch, ``t0_catalogue`` keeps the catalogue's, and the count
    and the signal-to-noise are those of the refined epoch; ``absent`` when the dip
    is below ``EPHEMERIS_MIN_SNR`` and the expected signal-to-noise is at least
    ``EPHEMERIS_CHECK_SNR``. The caller leaves an absent planet out of the set.
    """
    period = float(planet["period"])
    duration = float(planet["duration_hours"]) / 24.0
    depth = float(planet["depth_ppm"]) * 1e-6
    epoch = float(planet["epoch_bjd"]) - BTJD_OFFSET
    first = epoch + period * np.ceil((time.min() - epoch) / period)
    t0_catalogue = first
    n_transits, snr = _transits_and_snr(time, flux, period, first, duration, depth)
    check = {"measured_depth_ppm": None, "depth_snr": None, "epoch_shift_days": None}
    word = "no_transit"
    if n_transits > 0:
        check = verify_ephemeris(time, flux, period, first, duration)
        if check["depth_snr"] < EPHEMERIS_MIN_SNR and snr >= EPHEMERIS_CHECK_SNR:
            word = "absent"
        elif abs(check["epoch_shift_days"]) > duration and check["depth_snr"] >= EPHEMERIS_MIN_SNR:
            word = "shifted"
            first = check["t0_fitted"]
            first = first + period * np.ceil((time.min() - first) / period)
            n_transits, snr = _transits_and_snr(time, flux, period, first, duration, depth)
        else:
            word = "confirmed"
    record = {
        "period": period,
        "t0": float(first - t_offset),
        "duration": duration,
        "depth": depth,
        "rp_rs": float(np.sqrt(depth)),
        "source": "toi",
        "toi": planet.get("toi"),
        "disposition": planet.get("disposition"),
        "alerted": planet.get("alerted"),
        "ephemeris": word,
        "t0_catalogue": float(t0_catalogue - t_offset),
        "measured_depth_ppm": check["measured_depth_ppm"],
        "depth_snr": check["depth_snr"],
        "epoch_shift_days": check["epoch_shift_days"],
    }
    return record, n_transits, snr


class EpisodeSet:
    """A set of light curves with hidden answers, ready for evaluation.

    An episode set is the unit of evaluation. It is a directory with:

    - ``episodes/ep_NNNN.npz``: one file for each episode, holding only ``time``,
      ``flux`` and ``flux_err``. A detector sees these and nothing else.
    - ``truth.json``: for each episode, whether a transit was injected and with what
      parameters, plus the host star and the expected signal-to-noise ratio. Only the
      scorer reads this.
    - ``config.json``: the :class:`BuildConfig`, the git commit and the targets file hash.
    - ``manifest.json``: the SHA-256 of each file. The hash of the manifest identifies
      the set in results files.

    Use :meth:`build` to create a set and the constructor to open an existing one.

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
        return sorted(path.stem for path in (self.root / "episodes").glob("ep_*.npz"))

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
        return Manifest.digest_of(self.root).hex

    def load(self, ep_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(time, flux, flux_err)`` for one episode."""
        with np.load(self.root / "episodes" / f"{ep_id}.npz") as archive:
            return archive["time"], archive["flux"], archive["flux_err"]

    @property
    def contexts(self) -> dict[str, dict]:
        """The ``context.json`` records by episode id, or an empty dict if the set has none."""
        path = self.root / "context.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def context(self, ep_id: str):
        """Return the :class:`stargazer.context.EpisodeContext` of one episode, or None.

        The context holds the star's catalogue rows and nothing about the injection.
        A set built before ``stargazer context`` was run has none.
        """
        from stargazer.context import EpisodeContext

        record = self.contexts.get(ep_id)
        return EpisodeContext.from_dict(record) if record else None

    def __iter__(self) -> Iterator[str]:
        return iter(self.ids)

    def __len__(self) -> int:
        return len(self.ids)

    def verify(self) -> list[str]:
        """Recompute each hash. An empty list means the set is intact."""
        return Manifest.read(self.root).verify(self.root)

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

        cached = [target for target in targets["targets"] if cache.has(target["tic"], target["sector"])]
        hosts = [target for target in cached if _has_finite(target, "logg", "radius_rsun")]
        skipped = [target["tic"] for target in cached if target not in hosts]
        if not hosts:
            raise RuntimeError("no cached light curves with finite stellar parameters; run fetch first")

        truth = {}
        for index in range(cfg.n_episodes):
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
                "star": {key: star.get(key) for key in ("teff", "logg", "radius_rsun", "tmag", "mass_msun")},
                "point_scatter": point_scatter(flux),
            }
            if not is_null:
                tr = sample_truth(
                    rng, lc.time, star, period_range=cfg.period_range, depth_range=cfg.depth_range
                )
                rec["snr"] = expected_snr(lc.time, flux, tr)
                rec["n_transits"] = transit_count(lc.time, tr)
                flux = inject(lc.time, flux, tr)
                transit_record = tr.to_dict()
                transit_record["t0"] = tr.t0 - t_offset
                rec["transit"] = transit_record
            ep_id = f"ep_{index:04d}"
            np.savez_compressed(
                ep_dir / f"{ep_id}.npz",
                time=(lc.time - t_offset).astype(np.float64),
                flux=flux.astype(np.float64),
                flux_err=lc.flux_err.astype(np.float64),
            )
            truth[ep_id] = rec

        (out_dir / "truth.json").write_text(json.dumps(truth, indent=1, sort_keys=True))
        config = {
            "created": utc_timestamp(),
            "git_commit": str(GitState.current()),
            "stargazer_version": __version__,
            "build": {
                **cfg.__dict__,
                "period_range": list(cfg.period_range),
                "depth_range": list(cfg.depth_range),
            },
            "targets_file": str(targets_path) if targets_path else None,
            "targets_sha256": Digest.of_file(targets_path).hex if targets_path else None,
            "n_available_hosts": len(hosts),
            "hosts_skipped_missing_params": skipped,
        }
        (out_dir / "config.json").write_text(json.dumps(config, indent=2))
        Manifest.of_directory(out_dir).write(out_dir)
        return cls(out_dir)

    @classmethod
    def build_real(
        cls,
        targets: dict,
        null_targets: dict,
        cache: LightCurveCache,
        out_dir: Path,
        seed: int,
        n_null: int,
        targets_path: Path | None = None,
        null_targets_path: Path | None = None,
    ) -> EpisodeSet:
        """Build an episode set of real planets: catalogue truth, no injection.

        Parameters
        ----------
        targets : dict
            Output of :func:`stargazer.targets.select_real_targets`; each target
            carries a ``planet`` entry.
        null_targets : dict
            A targets file of hosts with no TOI, from :func:`stargazer.targets.select_targets`,
            for the null episodes.
        cache : LightCurveCache
            Must already contain the light curves of both files.
        out_dir : Path
        seed : int
            Seed of the draw of the null hosts.
        n_null : int
            Number of null episodes.
        targets_path, null_targets_path : Path, optional
            Recorded with their hashes in ``config.json``.

        Notes
        -----
        One transit episode is written for each cached real host, in the order of
        the targets file, with the flux as observed and the truth from the TOI entry
        through :func:`catalogue_transit`. Then ``n_null`` null episodes are drawn
        with replacement from the cached null hosts. A host whose planet has no
        transit in the sector keeps ``n_transits`` 0 and is scored as a null, as an
        injection with no transit is. A host whose catalogue ephemeris the data do
        not show (``absent`` from :func:`catalogue_transit`) is left out and listed
        in ``config.json`` under ``absent_ephemerides``. The context of such a set
        must be written with the TOI entries masked.
        """
        rng = np.random.default_rng(seed)
        out_dir = Path(out_dir)
        ep_dir = out_dir / "episodes"
        ep_dir.mkdir(parents=True, exist_ok=True)
        real_hosts = [target for target in targets["targets"] if cache.has(target["tic"], target["sector"])]
        null_hosts = [
            target for target in null_targets["targets"] if cache.has(target["tic"], target["sector"])
        ]
        if not real_hosts or not null_hosts:
            raise RuntimeError("no cached light curves for the real hosts or the null hosts; run fetch first")
        truth = {}
        index = 0
        absent = []
        for star in real_hosts:
            lc = cache.load(star["tic"], star["sector"])
            t_offset = float(lc.time.min())
            transit, n_transits, snr = catalogue_transit(lc.time, lc.flux, star["planet"], t_offset)
            if transit["ephemeris"] == "absent":
                absent.append({"tic": star["tic"], "toi": transit["toi"], "depth_snr": transit["depth_snr"]})
                continue
            truth[f"ep_{index:04d}"] = {
                "kind": "transit",
                "tic": star["tic"],
                "sector": star["sector"],
                "t_offset": t_offset,
                "n_points": len(lc.time),
                "star": {key: star.get(key) for key in ("teff", "logg", "radius_rsun", "tmag", "mass_msun")},
                "point_scatter": point_scatter(lc.flux),
                "snr": snr,
                "n_transits": n_transits,
                "transit": transit,
            }
            cls._write_episode(ep_dir, f"ep_{index:04d}", lc, t_offset)
            index += 1
        for _ in range(n_null):
            star = null_hosts[int(rng.integers(len(null_hosts)))]
            lc = cache.load(star["tic"], star["sector"])
            t_offset = float(lc.time.min())
            truth[f"ep_{index:04d}"] = {
                "kind": "null",
                "tic": star["tic"],
                "sector": star["sector"],
                "t_offset": t_offset,
                "n_points": len(lc.time),
                "star": {key: star.get(key) for key in ("teff", "logg", "radius_rsun", "tmag", "mass_msun")},
                "point_scatter": point_scatter(lc.flux),
            }
            cls._write_episode(ep_dir, f"ep_{index:04d}", lc, t_offset)
            index += 1
        (out_dir / "truth.json").write_text(json.dumps(truth, indent=1, sort_keys=True))
        config = {
            "created": utc_timestamp(),
            "git_commit": str(GitState.current()),
            "stargazer_version": __version__,
            "build": {"real": True, "seed": seed, "n_null": n_null, "n_real": len(real_hosts) - len(absent)},
            "absent_ephemerides": absent,
            "targets_file": str(targets_path) if targets_path else None,
            "targets_sha256": Digest.of_file(targets_path).hex if targets_path else None,
            "null_targets_file": str(null_targets_path) if null_targets_path else None,
            "null_targets_sha256": Digest.of_file(null_targets_path).hex if null_targets_path else None,
            "toi_csv": targets.get("toi_csv"),
        }
        (out_dir / "config.json").write_text(json.dumps(config, indent=2))
        Manifest.of_directory(out_dir).write(out_dir)
        return cls(out_dir)

    @staticmethod
    def _write_episode(ep_dir: Path, ep_id: str, lc, t_offset: float) -> None:
        """Write one episode file: relative time, the flux as observed, the flux error."""
        np.savez_compressed(
            ep_dir / f"{ep_id}.npz",
            time=(lc.time - t_offset).astype(np.float64),
            flux=lc.flux.astype(np.float64),
            flux_err=lc.flux_err.astype(np.float64),
        )
