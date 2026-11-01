"""Unit tests for stargazer.context: the TOI list parser, the context record and the writer."""

import json

from stargazer.context import (
    BTJD_OFFSET,
    CatalogueFetcher,
    ContextWriter,
    EpisodeContext,
    TessEbCatalogue,
    ToiList,
    eclipsing_binary_entry,
    tic_id,
)

TOI_CSV = """TIC ID,TOI,Period (days),Epoch (BJD),Duration (hours),Depth (ppm),TFOPWG Disposition
123,100.01,3.5,2459000.5,2.1,800,PC
123,100.02,,,1.0,,FP
456,200.01,10.0,2459010.0,3.0,1500,CP
"""


def test_toi_list_groups_entries_by_tic_and_parses_numbers():
    """Each star gets its own TOI entries, with numbers as floats and an empty cell as None."""
    toi_list = ToiList(TOI_CSV)
    entries = toi_list.entries(123)
    assert [entry["toi"] for entry in entries] == ["100.01", "100.02"]
    assert (
        entries[0]["period"] == 3.5
        and entries[0]["epoch_bjd"] == 2459000.5
        and entries[0]["depth_ppm"] == 800
    )
    assert entries[1]["period"] is None and entries[1]["disposition"] == "FP"
    assert toi_list.entries(999) == []


def test_toi_list_records_the_hash_of_the_text():
    """The hash identifies the downloaded file, so a results file can name the exact list."""
    assert len(ToiList(TOI_CSV).sha256) == 64 and ToiList(TOI_CSV).sha256 == ToiList(TOI_CSV).sha256


def test_context_converts_a_catalogue_epoch_to_episode_time():
    """A BJD epoch minus the BTJD offset and the episode's time offset is days from the episode start."""
    context = EpisodeContext(tic=1, sector=98, ra=0.0, dec=0.0, tmag=None, t_offset=3800.0)
    assert context.bjd_to_episode_time(BTJD_OFFSET + 3812.5) == 12.5


def test_context_round_trips_through_a_dictionary():
    """``to_dict`` and ``from_dict`` are inverses, so context.json holds every field."""
    context = EpisodeContext(
        tic=1, sector=98, ra=1.0, dec=2.0, tmag=9.5, t_offset=10.0, tois=[{"toi": "1.01", "period": 2.0}]
    )
    assert EpisodeContext.from_dict(context.to_dict()) == context


class FakeFetcher(CatalogueFetcher):
    """Returns a fixed context for any star and counts the calls, so no test touches the network."""

    def __init__(self):
        super().__init__(toi_list=ToiList(TOI_CSV))
        self.calls = 0

    def context_for(self, star, sector, t_offset):
        self.calls += 1
        return EpisodeContext(
            tic=int(star["tic"]), sector=sector, ra=star["ra"], dec=star["dec"], tmag=None, t_offset=t_offset
        )


def test_writer_fetches_each_star_once_and_keeps_each_episode_offset(tmp_path):
    """Two episodes of one star share one fetch; each context carries its own time offset and no truth."""
    truth = {
        "ep_0000": {
            "kind": "transit",
            "tic": 123,
            "sector": 98,
            "t_offset": 100.0,
            "transit": {"period": 3.0},
        },
        "ep_0001": {"kind": "null", "tic": 123, "sector": 98, "t_offset": 100.5},
    }
    (tmp_path / "truth.json").write_text(json.dumps(truth))
    targets = {"targets": [{"tic": 123, "ra": 1.0, "dec": 2.0}]}
    fetcher = FakeFetcher()
    path = ContextWriter(fetcher).write(tmp_path, targets)
    written = json.loads(path.read_text())
    assert fetcher.calls == 1
    assert written["ep_0000"]["t_offset"] == 100.0 and written["ep_0001"]["t_offset"] == 100.5
    assert "transit" not in written["ep_0000"] and "kind" not in written["ep_0000"]


def test_tic_id_accepts_a_positive_integer_and_nothing_else():
    """A TIC identifier is a positive integer; an empty cell, text, zero, a float or a bool is None."""
    assert tic_id(123) == 123 and tic_id("123") == 123 and tic_id(" 7 ") == 7
    assert all(tic_id(value) is None for value in ("", "abc", 0, -5, "12.5", True, None))


def test_toi_list_skips_rows_without_a_valid_tic():
    """A row whose TIC cell is empty or text does not land under any star."""
    toi_list = ToiList("TIC ID,TOI,Period (days)\n,1.01,2.0\nabc,1.02,3.0\n9,1.03,4.0\n")
    assert toi_list.entries(9)[0]["period"] == 4.0 and sum(len(toi_list.entries(t)) for t in (9,)) == 1


class _QueryCountingFetcher(CatalogueFetcher):
    """Answers the two catalogue queries from fixed tables, so the evidence rule is tested without Vizier."""

    def __init__(self, eclipsing_binaries, gaia_classes):
        super().__init__(toi_list=ToiList(TOI_CSV))
        self._fixed_binaries = eclipsing_binaries
        self._fixed_classes = gaia_classes

    def _eclipsing_binaries(self, tic):
        return list(self._fixed_binaries)

    def _gaia_classes(self, ra, dec):
        return list(self._fixed_classes)


def test_binary_evidence_names_each_catalogue_that_flags_the_star():
    """A TESS catalogue entry gives tess_eb, a Gaia ECL class gives gaia_ECL, both give both in that order."""
    star = {"tic": 5, "ra": 1.0, "dec": 2.0}
    assert _QueryCountingFetcher([{"period": 2.0}], ["ECL"]).binary_evidence(star) == ["tess_eb", "gaia_ECL"]
    assert _QueryCountingFetcher([], ["ECL"]).binary_evidence(star) == ["gaia_ECL"]
    assert _QueryCountingFetcher([{"period": 2.0}], []).binary_evidence(star) == ["tess_eb"]


def test_binary_evidence_is_empty_for_a_rotational_variable():
    """A Gaia class other than ECL, such as a spotted rotator, is not a reason to exclude a host."""
    assert (
        _QueryCountingFetcher([], ["RS", "SOLAR_LIKE"]).binary_evidence({"tic": 5, "ra": 1.0, "dec": 2.0})
        == []
    )


def test_writer_masks_the_toi_entries_when_asked(tmp_path):
    """With ``mask_tois`` the context of a real-planet set carries no TOI entry and names the source as
    masked, so ``ephemeris_match`` cannot hand the hidden answer to the model; the other fields stay.
    """
    truth = {"ep_0000": {"kind": "transit", "tic": 123, "sector": 98, "t_offset": 100.0}}
    (tmp_path / "truth.json").write_text(json.dumps(truth))
    targets = {"targets": [{"tic": 123, "ra": 1.0, "dec": 2.0}]}

    class FetcherWithToi(FakeFetcher):
        def context_for(self, star, sector, t_offset):
            context = super().context_for(star, sector, t_offset)
            context.tois = [{"toi": "1.01", "period": 3.0}]
            context.sources = {"toi": {"url": "x"}, "gaia_classes": "y"}
            return context

    written = json.loads(ContextWriter(FetcherWithToi(), mask_tois=True).write(tmp_path, targets).read_text())
    assert written["ep_0000"]["tois"] == [] and written["ep_0000"]["sources"]["toi"] == "masked"
    assert written["ep_0000"]["sources"]["gaia_classes"] == "y" and written["ep_0000"]["tic"] == 123


def test_fetcher_with_mast_gaia_uses_no_vizier_and_records_what_is_unavailable(monkeypatch):
    """With ``gaia_source`` mast the neighbours come from the MAST query, no Vizier query runs, and the
    two Vizier-only catalogues are empty with their source marked unavailable, so a reader knows they
    were not checked.
    """
    fetcher = CatalogueFetcher(toi_list=ToiList(TOI_CSV), gaia_source="mast")

    def no_vizier(*args, **kwargs):
        raise AssertionError("Vizier was queried")

    monkeypatch.setattr(fetcher, "_eclipsing_binaries", no_vizier)
    monkeypatch.setattr(fetcher, "_gaia", no_vizier)
    monkeypatch.setattr(fetcher, "_gaia_mast", lambda ra, dec: ([{"separation_arcsec": 5.0}], 10.0))
    context = fetcher.context_for({"tic": 456, "ra": 1.0, "dec": 2.0}, 98, 100.0)
    assert context.neighbours == [{"separation_arcsec": 5.0}] and context.target_gmag == 10.0
    assert context.eclipsing_binaries == [] and context.gaia_classes == []
    assert context.sources["gaia_sources"].startswith("MAST")
    assert "unavailable" in context.sources["gaia_classes"]
    assert context.tois[0]["toi"] == "200.01"


def test_neighbours_from_rows_splits_the_target_from_the_neighbours():
    """The source within the match radius is the target and sets the magnitude differences of the rest."""
    rows = [(20.0, 14.0), (1.0, 10.0), (40.0, None)]
    neighbours, target_gmag = CatalogueFetcher._neighbours_from_rows(rows)
    assert target_gmag == 10.0
    assert neighbours == [
        {"separation_arcsec": 20.0, "gmag": 14.0, "delta_gmag": 4.0},
        {"separation_arcsec": 40.0, "gmag": None, "delta_gmag": None},
    ]


def test_writer_reuses_the_contexts_of_another_set_and_fetches_only_the_rest(tmp_path):
    """A star found in the reused file is copied with its own time offset and a ``reused_from`` source,
    and the fetcher is asked only for the star that is not there.
    """
    old = EpisodeContext(
        tic=123, sector=98, ra=1.0, dec=2.0, tmag=None, t_offset=50.0, gaia_classes=["RS"], sources={"x": "y"}
    )
    (tmp_path / "old_context.json").write_text(json.dumps({"ep_0009": old.to_dict()}))
    truth = {
        "ep_0000": {"kind": "null", "tic": 123, "sector": 98, "t_offset": 100.0},
        "ep_0001": {"kind": "null", "tic": 999, "sector": 98, "t_offset": 101.0},
    }
    (tmp_path / "truth.json").write_text(json.dumps(truth))
    targets = {"targets": [{"tic": 123, "ra": 1.0, "dec": 2.0}, {"tic": 999, "ra": 3.0, "dec": 4.0}]}
    fetcher = FakeFetcher()
    writer = ContextWriter(fetcher, reuse=tmp_path / "old_context.json")
    written = json.loads(writer.write(tmp_path, targets).read_text())
    assert fetcher.calls == 1
    assert written["ep_0000"]["gaia_classes"] == ["RS"] and written["ep_0000"]["t_offset"] == 100.0
    assert written["ep_0000"]["sources"]["reused_from"].endswith("old_context.json")
    assert written["ep_0001"]["tic"] == 999 and "reused_from" not in written["ep_0001"]["sources"]


EB_CSV = """tess_id,signal_id,ra,dec,bjd0,period,sectors
260128333,1,92.13,-59.54,1585.1666739843085,14.608582815613214,"4,5"
260128333,2,92.13,-59.54,1586.0,7.3,"4,5"
not_a_tic,1,0.0,0.0,1.0,2.0,"1"
"""


def test_tess_eb_catalogue_groups_entries_by_tic_and_stores_the_epoch_as_bjd():
    """The MAST CSV gives the eclipse epoch as BTJD; an entry stores it as BJD so the context converts
    it as it converts a TOI epoch. Two rows of one star give two entries and a row without a TIC is skipped.
    """
    catalogue = TessEbCatalogue(EB_CSV)
    entries = catalogue.entries(260128333)
    assert [entry["period"] for entry in entries] == [14.608582815613214, 7.3]
    assert entries[0]["epoch_bjd"] == 1585.1666739843085 + BTJD_OFFSET
    assert catalogue.entries(1) == []


def test_eclipsing_binary_entry_leaves_a_missing_epoch_as_none():
    """A catalogue row without an epoch gives ``epoch_bjd`` None, not an offset added to nothing."""
    assert eclipsing_binary_entry(2.0, None) == {"period": 2.0, "epoch_bjd": None}


def test_fetcher_takes_the_binaries_from_the_disk_copy_and_queries_vizier_for_nothing_else(monkeypatch):
    """With a disk copy of the eclipsing-binary catalogue and the MAST Gaia source, no Vizier query runs:
    the binaries come from the copy with its hash recorded, and the evidence rule reads the copy too.
    """
    fetcher = CatalogueFetcher(
        toi_list=ToiList(TOI_CSV), gaia_source="mast", eclipsing_binaries=TessEbCatalogue(EB_CSV)
    )

    def no_vizier(*args, **kwargs):
        raise AssertionError("Vizier was queried")

    monkeypatch.setattr(fetcher, "_vizier", no_vizier)
    monkeypatch.setattr(fetcher, "_gaia_mast", lambda ra, dec: ([], 10.0))
    star = {"tic": 260128333, "ra": 92.13, "dec": -59.54}
    context = fetcher.context_for(star, 98, 100.0)
    assert len(context.eclipsing_binaries) == 2
    assert context.sources["eclipsing_binaries"]["sha256"] == TessEbCatalogue(EB_CSV).sha256
    assert "unavailable" in context.sources["gaia_classes"]
    assert fetcher.binary_evidence(star) == ["tess_eb"]


def test_fetcher_with_mast_gaia_takes_the_classes_from_a_named_vizier_mirror(monkeypatch):
    """A Vizier mirror in ``vizier_server`` fills the Gaia classes even with the MAST Gaia source, and the
    source record names the mirror, so a context says where each field came from.
    """
    fetcher = CatalogueFetcher(
        toi_list=ToiList(TOI_CSV),
        gaia_source="mast",
        eclipsing_binaries=TessEbCatalogue(EB_CSV),
        vizier_server="vizier.cfa.harvard.edu",
    )
    monkeypatch.setattr(fetcher, "_gaia_classes", lambda ra, dec: ["ECL"])
    monkeypatch.setattr(fetcher, "_gaia_mast", lambda ra, dec: ([], 10.0))
    context = fetcher.context_for({"tic": 456, "ra": 1.0, "dec": 2.0}, 98, 100.0)
    assert context.gaia_classes == ["ECL"]
    assert context.sources["gaia_classes"] == "I/358/vclassre at vizier.cfa.harvard.edu"
    assert context.sources["gaia_sources"].startswith("MAST")


def test_fetcher_without_tois_downloads_no_list_and_says_so(monkeypatch):
    """With ``fetch_tois`` False the TOI list is never downloaded and the source record says not fetched,
    so a masked real-planet set can be built while ExoFOP is down.
    """
    fetcher = CatalogueFetcher(
        gaia_source="mast", eclipsing_binaries=TessEbCatalogue(EB_CSV), fetch_tois=False
    )

    def no_download(cls, url=None):
        raise AssertionError("the TOI list was downloaded")

    monkeypatch.setattr(ToiList, "download", classmethod(no_download))
    monkeypatch.setattr(fetcher, "_gaia_mast", lambda ra, dec: ([], 10.0))
    context = fetcher.context_for({"tic": 456, "ra": 1.0, "dec": 2.0}, 98, 100.0)
    assert context.tois == [] and context.sources["toi"] == "not fetched"
