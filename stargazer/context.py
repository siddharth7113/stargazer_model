"""What is known about an episode's star beyond its light curve.

An episode is one light curve. A detector that only reads the three arrays cannot
tell a known eclipsing binary from a new candidate, or a dip on the target from a dip
on a neighbour in the same pixels. The mission vetting reports use public catalogues
for those checks. This module fetches the catalogue rows for a host star once, stores
them beside the episode set with the date of the download, and hands them to the
tools as an :class:`EpisodeContext`.

The context carries no truth. It holds the star's identity, position and brightness,
the time offset of the episode, and the catalogue rows. An injected transit has no
catalogue entry, so a match tells the detector about the star's own signals only.

Sources
-------
``TOI_URL``
    The TESS Objects of Interest list from ExoFOP, as a CSV file [1]_.
``TESS_EB_CATALOGUE``
    The TESS eclipsing binary catalogue of Prša et al. (2022) on Vizier [2]_.
``GAIA_CLASS_CATALOGUE`` and ``GAIA_SOURCE_CATALOGUE``
    The Gaia DR3 variability classification and the Gaia DR3 source table on
    Vizier [3]_, [4]_.

References
----------
.. [1] Guerrero, N. M., et al. (2021). The TESS Objects of Interest catalog from the
   TESS prime mission. The Astrophysical Journal Supplement Series, 254, 39.
   https://doi.org/10.3847/1538-4365/abefe1
.. [2] Prša, A., et al. (2022). TESS eclipsing binary stars. I. Short-cadence
   observations of 4584 eclipsing binaries in sectors 1-26. The Astrophysical Journal
   Supplement Series, 258, 16. https://doi.org/10.3847/1538-4365/ac324a
.. [3] Eyer, L., et al. (2023). Gaia Data Release 3. Summary of the variability
   processing and analysis. Astronomy and Astrophysics, 674, A13.
   https://doi.org/10.1051/0004-6361/202244242
.. [4] Gaia Collaboration (2023). Gaia Data Release 3. Summary of the content and
   survey properties. Astronomy and Astrophysics, 674, A1.
   https://doi.org/10.1051/0004-6361/202243940
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from stargazer.provenance import utc_timestamp

TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
TESS_EB_CATALOGUE = "J/ApJS/258/16/tess-ebs"
GAIA_CLASS_CATALOGUE = "I/358/vclassre"
GAIA_SOURCE_CATALOGUE = "I/355/gaiadr3"
BTJD_OFFSET = 2457000.0
"""Days subtracted from a Barycentric Julian Date to give TESS Barycentric TESS Julian
Date (BTJD), the time base of the cached light curves."""

TARGET_MATCH_ARCSEC = 3.0
"""Radius in arcseconds within which a catalogue source is the target itself."""

NEIGHBOUR_RADIUS_ARCSEC = 63.0
"""Radius in arcseconds of the neighbour search: three TESS pixels of 21 arcseconds."""


def tic_id(value: object) -> int | None:
    """Return the value as a TESS Input Catalog identifier, or None if it is not one.

    Notes
    -----
    A TIC identifier is a positive integer. A catalogue cell can be empty, a float
    string or text, so the parser accepts an integer or a string of digits and
    rejects everything else, including zero and negative numbers.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    text = str(value).strip()
    return int(text) if text.isdigit() and int(text) > 0 else None


@dataclass
class EpisodeContext:
    """What a tool may know about an episode's star beyond the light curve.

    Parameters
    ----------
    tic : int
        TESS Input Catalog identifier of the host.
    sector : int
        TESS sector of the light curve.
    ra, dec : float
        Position, degrees, ICRS.
    tmag : float or None
        TESS magnitude.
    t_offset : float
        BTJD of the first point of the episode, so a catalogue epoch converts to
        the episode's time base as ``bjd - BTJD_OFFSET - t_offset``.
    tois : list of dict
        TOI entries for this star: ``toi, period, epoch_bjd, duration_hours,
        depth_ppm, disposition``.
    eclipsing_binaries : list of dict
        TESS eclipsing-binary entries: ``period, epoch_bjd``.
    gaia_classes : list of str
        Gaia DR3 variability classes of the target.
    neighbours : list of dict
        Gaia sources within :data:`NEIGHBOUR_RADIUS_ARCSEC`, excluding the target:
        ``separation_arcsec, gmag, delta_gmag``.
    target_gmag : float or None
        Gaia G magnitude of the target.
    fetched : str
        UTC time of the catalogue download.
    sources : dict
        The URL or catalogue identifier of each source and, for the TOI list, its
        SHA-256.
    """

    tic: int
    sector: int
    ra: float
    dec: float
    tmag: float | None
    t_offset: float
    tois: list[dict] = field(default_factory=list)
    eclipsing_binaries: list[dict] = field(default_factory=list)
    gaia_classes: list[str] = field(default_factory=list)
    neighbours: list[dict] = field(default_factory=list)
    target_gmag: float | None = None
    fetched: str = ""
    sources: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary for JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodeContext:
        """Build a context from a dictionary written by :meth:`to_dict`."""
        return cls(**data)

    def bjd_to_episode_time(self, bjd: float) -> float:
        """Convert a Barycentric Julian Date to days from the start of the episode."""
        return bjd - BTJD_OFFSET - self.t_offset


class ToiList:
    """The TOI list as rows keyed by TIC, parsed from the ExoFOP CSV text.

    Parameters
    ----------
    text : str
        The CSV file content.
    """

    COLUMNS = {
        "toi": "TOI",
        "period": "Period (days)",
        "epoch_bjd": "Epoch (BJD)",
        "duration_hours": "Duration (hours)",
        "depth_ppm": "Depth (ppm)",
        "disposition": "TFOPWG Disposition",
    }

    def __init__(self, text: str):
        self.sha256 = hashlib.sha256(text.encode()).hexdigest()
        self._rows: dict[int, list[dict]] = {}
        for row in csv.DictReader(io.StringIO(text)):
            tic = tic_id(row.get("TIC ID"))
            if tic is None:
                continue
            self._rows.setdefault(tic, []).append(self._entry(row))

    @classmethod
    def download(cls, url: str = TOI_URL) -> ToiList:
        """Download the list."""
        with urllib.request.urlopen(url, timeout=120) as response:
            return cls(response.read().decode("utf-8", errors="replace"))

    def entries(self, tic: int) -> list[dict]:
        """Return the TOI entries of one star, an empty list if it has none."""
        return list(self._rows.get(tic, []))

    def _entry(self, row: dict) -> dict:
        entry: dict[str, Any] = {}
        for key, column in self.COLUMNS.items():
            value = row.get(column, "")
            if key in ("toi", "disposition"):
                entry[key] = value or None
                continue
            try:
                entry[key] = float(value)
            except ValueError:
                entry[key] = None
        return entry


class CatalogueFetcher:
    """Fetch the catalogue rows of host stars and build their contexts.

    Parameters
    ----------
    toi_list : ToiList, optional
        A parsed TOI list. None downloads it on first use.

    Notes
    -----
    The TOI list is one file and is read once. The eclipsing-binary entries and the
    Gaia rows come from Vizier through astroquery, one cone search for each star:
    :data:`TARGET_MATCH_ARCSEC` for the target's own entries and
    :data:`NEIGHBOUR_RADIUS_ARCSEC` for the neighbours. The neighbour list records
    each source's G magnitude difference to the target; the tools derive the flux
    fraction from it.
    """

    def __init__(self, toi_list: ToiList | None = None):
        self._toi_list = toi_list

    @property
    def toi_list(self) -> ToiList:
        """The TOI list, downloaded on first use."""
        if self._toi_list is None:
            self._toi_list = ToiList.download()
        return self._toi_list

    def context_for(self, star: dict, sector: int, t_offset: float) -> EpisodeContext:
        """Build the context of one host star.

        Parameters
        ----------
        star : dict
            A target record with ``tic, ra, dec`` and optionally ``tmag``.
        sector : int
        t_offset : float
            BTJD of the first point of the episode.
        """
        tic = tic_id(star.get("tic"))
        if tic is None:
            raise ValueError(f"target record has no valid TIC identifier: {star.get('tic')!r}")
        eclipsing_binaries = self._eclipsing_binaries(tic)
        gaia_classes, neighbours, target_gmag = self._gaia(float(star["ra"]), float(star["dec"]))
        return EpisodeContext(
            tic=tic,
            sector=int(sector),
            ra=float(star["ra"]),
            dec=float(star["dec"]),
            tmag=star.get("tmag"),
            t_offset=float(t_offset),
            tois=self.toi_list.entries(tic),
            eclipsing_binaries=eclipsing_binaries,
            gaia_classes=gaia_classes,
            neighbours=neighbours,
            target_gmag=target_gmag,
            fetched=utc_timestamp(),
            sources={
                "toi": {"url": TOI_URL, "sha256": self.toi_list.sha256},
                "eclipsing_binaries": TESS_EB_CATALOGUE,
                "gaia_classes": GAIA_CLASS_CATALOGUE,
                "gaia_sources": GAIA_SOURCE_CATALOGUE,
            },
        )

    def _eclipsing_binaries(self, tic: int) -> list[dict]:
        from astroquery.vizier import Vizier

        tables = Vizier(columns=["TIC", "Per", "BJD0"], row_limit=20).query_constraints(
            catalog=TESS_EB_CATALOGUE, TIC=str(tic)
        )
        entries = []
        for table in tables:
            for row in table:
                entries.append(
                    {
                        "period": _float_or_none(row["Per"]),
                        "epoch_bjd": _float_or_none(row["BJD0"]),
                    }
                )
        return entries

    def _gaia(self, ra: float, dec: float) -> tuple[list[str], list[dict], float | None]:
        import astropy.units as units
        from astropy.coordinates import SkyCoord
        from astroquery.vizier import Vizier

        coord = SkyCoord(ra, dec, unit="deg")
        classes = []
        for table in Vizier(columns=["Class"], row_limit=5).query_region(
            coord, radius=TARGET_MATCH_ARCSEC * units.arcsec, catalog=GAIA_CLASS_CATALOGUE
        ):
            classes += [str(row["Class"]) for row in table]
        sources = Vizier(columns=["Source", "RAJ2000", "DEJ2000", "Gmag"], row_limit=200).query_region(
            coord, radius=NEIGHBOUR_RADIUS_ARCSEC * units.arcsec, catalog=GAIA_SOURCE_CATALOGUE
        )
        rows = []
        for table in sources:
            for row in table:
                position = SkyCoord(float(row["RAJ2000"]), float(row["DEJ2000"]), unit="deg")
                rows.append((float(coord.separation(position).arcsec), _float_or_none(row["Gmag"])))
        rows.sort()
        target_gmag = next(
            (gmag for sep, gmag in rows if sep <= TARGET_MATCH_ARCSEC and gmag is not None), None
        )
        neighbours = [
            {
                "separation_arcsec": round(sep, 1),
                "gmag": gmag,
                "delta_gmag": None if gmag is None or target_gmag is None else round(gmag - target_gmag, 2),
            }
            for sep, gmag in rows
            if sep > TARGET_MATCH_ARCSEC
        ]
        return classes, neighbours, target_gmag


def _float_or_none(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if number != number else number


class ContextWriter:
    """Write ``context.json`` for an episode set from its truth and targets files.

    Parameters
    ----------
    fetcher : CatalogueFetcher

    Notes
    -----
    The writer reads the truth file for the TIC, the sector and the time offset of
    each episode, which identify the star and the time base and carry nothing about
    the injection. One context is fetched for each distinct star and sector, and
    every episode of that star receives it with its own time offset.
    """

    def __init__(self, fetcher: CatalogueFetcher):
        self._fetcher = fetcher

    def write(self, episode_dir: Path, targets: dict) -> Path:
        """Fetch the contexts, write ``context.json`` and return its path."""
        truth = json.loads((episode_dir / "truth.json").read_text())
        stars = {tic_id(target["tic"]): target for target in targets["targets"]}
        cache: dict[tuple[int, int], EpisodeContext] = {}
        contexts = {}
        for episode_id, record in truth.items():
            tic = tic_id(record.get("tic"))
            if tic is None:
                raise ValueError(f"{episode_id} has no valid TIC identifier in the truth file")
            key = (tic, int(record["sector"]))
            if key not in cache:
                cache[key] = self._fetcher.context_for(stars[key[0]], key[1], float(record["t_offset"]))
            context = EpisodeContext.from_dict(cache[key].to_dict())
            context.t_offset = float(record["t_offset"])
            contexts[episode_id] = context.to_dict()
        path = episode_dir / "context.json"
        path.write_text(json.dumps(contexts, indent=1, sort_keys=True))
        return path
