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
``TESS_EB_CATALOGUE`` and ``TESS_EB_URL``
    The TESS eclipsing binary catalogue of Prša et al. (2022) on Vizier [2]_, and the
    same catalogue as one CSV file at MAST [5]_, which :class:`TessEbCatalogue` reads
    from disk so that a context never waits on Vizier. Both give the epoch of the
    first primary eclipse as BTJD (``BJD - 2457000``); the context stores it as BJD.
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
.. [5] TESS Eclipsing Binary Working Group (2021). TESS-EBS: the TESS eclipsing binary
   catalogue, version 1.0. MAST High Level Science Product.
   https://doi.org/10.17909/t9-9gm4-fx30
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
TESS_EB_URL = (
    "https://archive.stsci.edu/hlsps/tess-ebs/hlsp_tess-ebs_tess_lcf-ffi_s0001-s0026_tess_v1.0_cat.csv"
)
"""The TESS eclipsing-binary catalogue as one CSV file at MAST, the disk copy's origin."""
GAIA_CLASS_CATALOGUE = "I/358/vclassre"
GAIA_SOURCE_CATALOGUE = "I/355/gaiadr3"

GAIA_SOURCES: dict[str, str] = {
    "vizier": GAIA_SOURCE_CATALOGUE,
    "mast": "MAST Catalogs Gaia DR3 (catalog=Gaia, version=3)",
}
"""Where the Gaia sources of a context can come from, by name. Vizier is the default;
MAST holds a copy of Gaia DR3 and answered on 2026-11-01 while every Vizier mirror had
been down for days (LOGBOOK 2026-11-01)."""
BTJD_OFFSET = 2457000.0
"""Days subtracted from a Barycentric Julian Date to give TESS Barycentric TESS Julian
Date (BTJD), the time base of the cached light curves."""

TARGET_MATCH_ARCSEC = 3.0
"""Radius in arcseconds within which a catalogue source is the target itself."""

NEIGHBOUR_RADIUS_ARCSEC = 63.0
"""Radius in arcseconds of the neighbour search: three TESS pixels of 21 arcseconds."""

GAIA_ECLIPSING_CLASS = "ECL"
"""The Gaia DR3 variability class of an eclipsing binary [3]_."""


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
        "tmag": "TESS Mag",
        "alerted": "Date TOI Alerted (UTC)",
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
            if key in ("toi", "disposition", "alerted"):
                entry[key] = value or None
                continue
            try:
                entry[key] = float(value)
            except ValueError:
                entry[key] = None
        return entry


class TessEbCatalogue:
    """The TESS eclipsing-binary catalogue as rows keyed by TIC, from the MAST CSV text.

    Parameters
    ----------
    text : str
        The CSV file content, as served at :data:`TESS_EB_URL`.

    Notes
    -----
    The file's ``bjd0`` column is the time of the first primary eclipse as BTJD
    (``BJD - 2457000``), the same number as the ``BJD0`` column of the Vizier copy.
    An entry stores it as ``epoch_bjd`` in full BJD, the convention of the TOI list,
    so that :meth:`EpisodeContext.bjd_to_episode_time` converts both the same way.
    """

    def __init__(self, text: str):
        self.sha256 = hashlib.sha256(text.encode()).hexdigest()
        self._rows: dict[int, list[dict]] = {}
        for row in csv.DictReader(io.StringIO(text)):
            tic = tic_id(row.get("tess_id"))
            if tic is None:
                continue
            self._rows.setdefault(tic, []).append(
                eclipsing_binary_entry(_float_or_none(row.get("period")), _float_or_none(row.get("bjd0")))
            )

    @classmethod
    def download(cls, url: str = TESS_EB_URL) -> TessEbCatalogue:
        """Download the catalogue."""
        with urllib.request.urlopen(url, timeout=120) as response:
            return cls(response.read().decode("utf-8", errors="replace"))

    @classmethod
    def read(cls, path: Path) -> TessEbCatalogue:
        """Read the catalogue from a CSV file on disk."""
        return cls(Path(path).read_text())

    def entries(self, tic: int) -> list[dict]:
        """Return the eclipsing-binary entries of one star, an empty list if it has none."""
        return [dict(entry) for entry in self._rows.get(tic, [])]


def eclipsing_binary_entry(period: float | None, epoch_btjd: float | None) -> dict:
    """Build a context entry from a catalogue period and an epoch given as BTJD.

    Notes
    -----
    Both copies of the TESS eclipsing-binary catalogue give the epoch as
    ``BJD - 2457000``. The entry stores ``epoch_bjd = epoch_btjd + BTJD_OFFSET`` so
    the context converts it as it converts a TOI epoch.
    """
    return {
        "period": period,
        "epoch_bjd": None if epoch_btjd is None else epoch_btjd + BTJD_OFFSET,
    }


class CatalogueFetcher:
    """Fetch the catalogue rows of host stars and build their contexts.

    Parameters
    ----------
    toi_list : ToiList, optional
        A parsed TOI list. None downloads it on first use.

    gaia_source : str
        ``"vizier"`` (the default) or ``"mast"``, see ``GAIA_SOURCES``.
    eclipsing_binaries : TessEbCatalogue, optional
        A copy of the TESS eclipsing-binary catalogue on disk. When given, the
        eclipsing-binary entries come from it and no Vizier query is made for them.
    vizier_server : str, optional
        The Vizier mirror for every Vizier query, for example
        ``"vizier.cfa.harvard.edu"``. None uses astroquery's default (Strasbourg).
    fetch_tois : bool
        False leaves the TOI entries out and never downloads the list; the source
        record says ``"not fetched"``. A writer that masks the entries uses this, so
        a masked set does not wait on ExoFOP.

    Notes
    -----
    The TOI list is one file and is read once. The eclipsing-binary entries and the
    Gaia rows come from Vizier through astroquery, one cone search for each star:
    :data:`TARGET_MATCH_ARCSEC` for the target's own entries and
    :data:`NEIGHBOUR_RADIUS_ARCSEC` for the neighbours. The neighbour list records
    each source's G magnitude difference to the target; the tools derive the flux
    fraction from it. With ``gaia_source`` ``"mast"`` the Gaia sources come from the
    copy of Gaia DR3 at MAST instead, and the two catalogues that only Vizier serves,
    the variability classes and the TESS eclipsing binaries, are left empty with
    their source recorded as unavailable; the context then says nothing about them
    rather than something wrong. Two ways fill those fields without Strasbourg: a
    disk copy of the eclipsing-binary catalogue in ``eclipsing_binaries``, and a
    mirror in ``vizier_server``, from which the Gaia classes are then fetched even
    with ``gaia_source`` ``"mast"`` (the Gaia source cone search was too slow on the
    Harvard mirror on 2026-11-01; the small class catalogue answered in a second).
    """

    def __init__(
        self,
        toi_list: ToiList | None = None,
        gaia_source: str = "vizier",
        eclipsing_binaries: TessEbCatalogue | None = None,
        vizier_server: str | None = None,
        fetch_tois: bool = True,
    ):
        if gaia_source not in GAIA_SOURCES:
            raise ValueError(f"gaia_source must be one of {sorted(GAIA_SOURCES)}, not {gaia_source!r}")
        self._toi_list = toi_list
        self._gaia_source = gaia_source
        self._eclipsing_binaries_on_disk = eclipsing_binaries
        self._vizier_server = vizier_server
        self._fetch_tois = fetch_tois

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
        ra, dec = float(star["ra"]), float(star["dec"])
        if self._gaia_source == "vizier":
            gaia_classes, neighbours, target_gmag = self._gaia(ra, dec)
            sources = {"gaia_classes": self._vizier_name(GAIA_CLASS_CATALOGUE)}
            sources["gaia_sources"] = self._vizier_name(GAIA_SOURCE_CATALOGUE)
        elif self._uses_vizier:
            gaia_classes = self._gaia_classes(ra, dec)
            neighbours, target_gmag = self._gaia_mast(ra, dec)
            sources = {"gaia_classes": self._vizier_name(GAIA_CLASS_CATALOGUE)}
            sources["gaia_sources"] = GAIA_SOURCES["mast"]
        else:
            gaia_classes = []
            neighbours, target_gmag = self._gaia_mast(ra, dec)
            sources = {"gaia_classes": "unavailable (Vizier not used)"}
            sources["gaia_sources"] = GAIA_SOURCES["mast"]
        if self._eclipsing_binaries_on_disk is not None:
            eclipsing_binaries = self._eclipsing_binaries_on_disk.entries(tic)
            sources["eclipsing_binaries"] = {
                "url": TESS_EB_URL,
                "sha256": self._eclipsing_binaries_on_disk.sha256,
            }
        elif self._uses_vizier:
            eclipsing_binaries = self._eclipsing_binaries(tic)
            sources["eclipsing_binaries"] = self._vizier_name(TESS_EB_CATALOGUE)
        else:
            eclipsing_binaries = []
            sources["eclipsing_binaries"] = "unavailable (Vizier not used)"
        if self._fetch_tois:
            tois = self.toi_list.entries(tic)
            sources["toi"] = {"url": TOI_URL, "sha256": self.toi_list.sha256}
        else:
            tois = []
            sources["toi"] = "not fetched"
        return EpisodeContext(
            tic=tic,
            sector=int(sector),
            ra=float(star["ra"]),
            dec=float(star["dec"]),
            tmag=star.get("tmag"),
            t_offset=float(t_offset),
            tois=tois,
            eclipsing_binaries=eclipsing_binaries,
            gaia_classes=gaia_classes,
            neighbours=neighbours,
            target_gmag=target_gmag,
            fetched=utc_timestamp(),
            sources=sources,
        )

    def binary_evidence(self, star: dict) -> list[str]:
        """Return the catalogue reasons to call a star a known eclipsing binary.

        Parameters
        ----------
        star : dict
            A target record with ``tic, ra, dec``.

        Returns
        -------
        list of str
            Empty for a star in neither catalogue. Otherwise ``"tess_eb"`` when the
            TIC has an entry in the TESS eclipsing-binary catalogue, and
            ``"gaia_ECL"`` when the Gaia DR3 variability class of the source within
            :data:`TARGET_MATCH_ARCSEC` is ECL (eclipsing binary).

        Notes
        -----
        Host selection uses this list as a cut. A star with its own eclipses is a
        false alarm for every detector, and an injection on such a star is unfindable
        by design, so the benchmark excludes these stars as it excludes TOI hosts.
        Each catalogue is read only where it is reachable: the binaries from the disk
        copy or Vizier, the classes from Vizier when the fetcher uses it.
        """
        tic = tic_id(star.get("tic"))
        if tic is None:
            raise ValueError(f"target record has no valid TIC identifier: {star.get('tic')!r}")
        evidence = []
        if self._eclipsing_binaries_on_disk is not None:
            has_entry = bool(self._eclipsing_binaries_on_disk.entries(tic))
        elif self._uses_vizier:
            has_entry = bool(self._eclipsing_binaries(tic))
        else:
            has_entry = False
        if has_entry:
            evidence.append("tess_eb")
        if self._uses_vizier and GAIA_ECLIPSING_CLASS in self._gaia_classes(
            float(star["ra"]), float(star["dec"])
        ):
            evidence.append("gaia_" + GAIA_ECLIPSING_CLASS)
        return evidence

    @property
    def _uses_vizier(self) -> bool:
        """True when a Vizier query can run: the default source, or a mirror is named."""
        return self._gaia_source == "vizier" or self._vizier_server is not None

    def _vizier_name(self, catalogue: str) -> str:
        """The source record of a Vizier catalogue: its identifier, with the mirror when one is set."""
        if self._vizier_server is None:
            return catalogue
        return f"{catalogue} at {self._vizier_server}"

    def _vizier(self, columns: list[str], row_limit: int):
        """An astroquery Vizier query object on the configured mirror."""
        from astroquery.vizier import Vizier

        if self._vizier_server is None:
            return Vizier(columns=columns, row_limit=row_limit)
        return Vizier(columns=columns, row_limit=row_limit, vizier_server=self._vizier_server)

    def _eclipsing_binaries(self, tic: int) -> list[dict]:
        tables = self._vizier(["TIC", "Per", "BJD0"], 20).query_constraints(
            catalog=TESS_EB_CATALOGUE, TIC=str(tic)
        )
        entries = []
        for table in tables:
            for row in table:
                entries.append(
                    eclipsing_binary_entry(_float_or_none(row["Per"]), _float_or_none(row["BJD0"]))
                )
        return entries

    def _gaia_classes(self, ra: float, dec: float) -> list[str]:
        import astropy.units as units
        from astropy.coordinates import SkyCoord

        classes = []
        for table in self._vizier(["Class"], 5).query_region(
            SkyCoord(ra, dec, unit="deg"),
            radius=TARGET_MATCH_ARCSEC * units.arcsec,
            catalog=GAIA_CLASS_CATALOGUE,
        ):
            classes += [str(row["Class"]) for row in table]
        return classes

    def _gaia(self, ra: float, dec: float) -> tuple[list[str], list[dict], float | None]:
        import astropy.units as units
        from astropy.coordinates import SkyCoord

        coord = SkyCoord(ra, dec, unit="deg")
        classes = self._gaia_classes(ra, dec)
        sources = self._vizier(["Source", "RAJ2000", "DEJ2000", "Gmag"], 200).query_region(
            coord, radius=NEIGHBOUR_RADIUS_ARCSEC * units.arcsec, catalog=GAIA_SOURCE_CATALOGUE
        )
        rows = []
        for table in sources:
            for row in table:
                position = SkyCoord(float(row["RAJ2000"]), float(row["DEJ2000"]), unit="deg")
                rows.append((float(coord.separation(position).arcsec), _float_or_none(row["Gmag"])))
        neighbours, target_gmag = self._neighbours_from_rows(rows)
        return classes, neighbours, target_gmag

    def _gaia_mast(self, ra: float, dec: float) -> tuple[list[dict], float | None]:
        """Return the neighbours and the target's G magnitude from the Gaia DR3 copy at MAST."""
        import astropy.units as units
        from astropy.coordinates import SkyCoord
        from astroquery.mast import Catalogs

        coord = SkyCoord(ra, dec, unit="deg")
        table = Catalogs.query_region(
            coord, radius=NEIGHBOUR_RADIUS_ARCSEC * units.arcsec, catalog="Gaia", version=3
        )
        rows = []
        for row in table:
            position = SkyCoord(float(row["ra"]), float(row["dec"]), unit="deg")
            rows.append((float(coord.separation(position).arcsec), _float_or_none(row["phot_g_mean_mag"])))
        return self._neighbours_from_rows(rows)

    @staticmethod
    def _neighbours_from_rows(rows: list[tuple[float, float | None]]) -> tuple[list[dict], float | None]:
        """Split (separation, G) rows into the target's magnitude and the neighbour list."""
        rows = sorted(rows)
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
        return neighbours, target_gmag


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

    def __init__(self, fetcher: CatalogueFetcher, mask_tois: bool = False, reuse: Path | None = None):
        self._fetcher = fetcher
        self._mask_tois = mask_tois
        self._reuse = Path(reuse) if reuse else None

    def write(self, episode_dir: Path, targets: dict) -> Path:
        """Fetch the contexts, write ``context.json`` and return its path.

        Notes
        -----
        With ``mask_tois`` the TOI entries of every host are left out of the context
        and the source record says ``"masked"``. A real-planet episode set needs this:
        its hidden answer is the host's own TOI, which ``ephemeris_match`` would
        otherwise hand to the model. The eclipsing-binary entries, the Gaia classes
        and the neighbours stay. With ``reuse``, a context file of another episode
        set, every star and sector found there is taken from it with its source
        record extended by ``reused_from``, and only the other stars are fetched.
        """
        truth = json.loads((episode_dir / "truth.json").read_text())
        stars = {tic_id(target["tic"]): target for target in targets["targets"]}
        cache: dict[tuple[int, int], EpisodeContext] = {}
        if self._reuse is not None:
            for record in json.loads(self._reuse.read_text()).values():
                context = EpisodeContext.from_dict(record)
                context.sources = {**context.sources, "reused_from": str(self._reuse)}
                cache.setdefault((context.tic, context.sector), context)
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
            if self._mask_tois:
                context.tois = []
                context.sources = {**context.sources, "toi": "masked"}
            contexts[episode_id] = context.to_dict()
        path = episode_dir / "context.json"
        path.write_text(json.dumps(contexts, indent=1, sort_keys=True))
        return path
