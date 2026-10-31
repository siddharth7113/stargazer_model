"""Unit tests for stargazer.targets: catalogue cell parsing and the selection cuts, network stubbed."""

import stargazer.targets as targets_module
from stargazer.context import CatalogueFetcher, ToiList
from stargazer.targets import _cell, remove_binaries, select_real_targets, select_targets


class _Masked:
    """Mimics a masked astropy table cell."""

    mask = True


def test_cell_masked_is_none():
    """A masked catalogue value means "unknown"."""
    assert _cell({"rad": _Masked()}, "rad", "radius_rsun") is None


def test_cell_nan_is_none():
    """The TIC returns NaN for an unknown radius; it must become None so the cuts can reject it."""
    assert _cell({"rad": float("nan")}, "rad", "radius_rsun") is None


def test_cell_number_and_string():
    """Numeric strings become floats; the luminosity class stays a string."""
    assert _cell({"rad": "1.5"}, "rad", "radius_rsun") == 1.5
    assert _cell({"lumclass": "DWARF"}, "lumclass", "lumclass") == "DWARF"


GOOD = {
    "tmag": 10.0,
    "teff": 5000.0,
    "radius_rsun": 1.0,
    "logg": 4.4,
    "mass_msun": 1.0,
    "lumclass": "DWARF",
    "ra": 0.0,
    "dec": 0.0,
}


class FakeBinaryCatalogue(CatalogueFetcher):
    """Names a fixed set of TICs as binaries and records each star asked about, so no test touches Vizier."""

    def __init__(self, binaries=None):
        super().__init__()
        self.binaries = dict(binaries or {})
        self.asked = []

    def binary_evidence(self, star):
        self.asked.append(star["tic"])
        return list(self.binaries.get(star["tic"], []))


def _stub_network(monkeypatch, params, toi_tics=frozenset({3}), binaries=None):
    """Replace the MAST, ExoFOP and Vizier calls with in-memory data."""
    monkeypatch.setattr(targets_module, "sector_two_minute_tics", lambda sector: sorted(params))
    monkeypatch.setattr(targets_module, "download_toi_tics", lambda: (set(toi_tics), "deadbeef"))
    monkeypatch.setattr(targets_module, "tic_params", lambda tics: {tic: params[tic] for tic in tics})
    monkeypatch.setattr(targets_module, "CatalogueFetcher", lambda: FakeBinaryCatalogue(binaries))


def test_select_targets_excludes_toi_and_applies_each_cut(monkeypatch):
    """Eight stars, one good: the TOI host and each failing cut are counted in the right bucket."""
    params = {
        1: GOOD,
        2: {**GOOD, "tmag": 13.0},
        3: GOOD,
        4: {**GOOD, "teff": 7000.0},
        5: {**GOOD, "radius_rsun": 3.0},
        6: {**GOOD, "logg": None},
        7: {**GOOD, "lumclass": "GIANT"},
        8: GOOD,
    }
    _stub_network(monkeypatch, params, binaries={8: ["tess_eb"]})
    selection = select_targets(sector=98, n_targets=10, seed=1)
    assert [target["tic"] for target in selection["targets"]] == [1]
    assert selection["n_excluded_by_toi"] == 1
    assert selection["rejected"] == {
        "no_params": 1,
        "tmag": 1,
        "teff": 1,
        "radius": 1,
        "lumclass": 1,
        "binary": 1,
    }
    assert selection["removed_binaries"] == [{"tic": 8, "evidence": ["tess_eb"]}]


def test_select_targets_asks_the_catalogues_only_for_stars_that_pass_the_cuts(monkeypatch):
    """The binary check is one Vizier query for each star, so it runs last, only on the dwarfs that remain."""
    params = {1: GOOD, 2: {**GOOD, "tmag": 13.0}, 3: {**GOOD, "radius_rsun": 3.0}}
    fetcher = FakeBinaryCatalogue()
    _stub_network(monkeypatch, params, toi_tics=frozenset())
    select_targets(sector=98, n_targets=10, seed=1, fetcher=fetcher)
    assert fetcher.asked == [1]


def test_select_targets_records_the_binary_catalogues(monkeypatch):
    """The targets file names the two catalogues behind the cut, as it names the TOI list, for audit."""
    _stub_network(monkeypatch, {1: GOOD})
    selection = select_targets(sector=98, n_targets=1, seed=1)
    assert selection["binary_catalogues"] == {
        "tess_eb": "J/ApJS/258/16/tess-ebs",
        "gaia_classes": "I/358/vclassre",
    }


def test_remove_binaries_keeps_the_other_stars_and_records_the_removed_ones():
    """Two of three stars stay in order; the removed star is listed with evidence; the input time is kept."""
    targets = {
        "created": "2026-10-11T03:55:46+00:00",
        "seed": 1,
        "n_selected": 3,
        "targets": [{"tic": 1, **GOOD}, {"tic": 2, **GOOD}, {"tic": 3, **GOOD}],
    }
    filtered = remove_binaries(targets, FakeBinaryCatalogue({2: ["tess_eb", "gaia_ECL"]}))
    assert [star["tic"] for star in filtered["targets"]] == [1, 3]
    assert filtered["n_selected"] == 2 and filtered["seed"] == 1
    assert filtered["removed_binaries"] == [{"tic": 2, "evidence": ["tess_eb", "gaia_ECL"]}]
    assert (
        filtered["derived_from"] == "2026-10-11T03:55:46+00:00"
        and filtered["created"] != filtered["derived_from"]
    )
    assert [star["tic"] for star in targets["targets"]] == [1, 2, 3]


def test_select_targets_records_toi_hash(monkeypatch):
    """The TOI CSV hash from the download is written into the selection for audit."""
    _stub_network(monkeypatch, {1: GOOD})
    assert select_targets(sector=98, n_targets=1, seed=1)["toi_csv"]["sha256"] == "deadbeef"


def test_select_targets_stops_at_n(monkeypatch):
    """With 20 good stars and n_targets 4, exactly 4 are selected."""
    _stub_network(monkeypatch, {tic: GOOD for tic in range(10, 30)}, toi_tics=frozenset())
    assert select_targets(sector=98, n_targets=4, seed=1)["n_selected"] == 4


def test_select_targets_is_seeded(monkeypatch):
    """The same seed picks the same stars; a different seed picks a different subset."""
    _stub_network(monkeypatch, {tic: GOOD for tic in range(100)}, toi_tics=frozenset())
    first = [target["tic"] for target in select_targets(98, 5, seed=1)["targets"]]
    assert first == [target["tic"] for target in select_targets(98, 5, seed=1)["targets"]]
    assert first != [target["tic"] for target in select_targets(98, 5, seed=2)["targets"]]


REAL_TOI_CSV = (
    "TIC ID,TOI,Period (days),Epoch (BJD),Duration (hours),Depth (ppm),"
    "TFOPWG Disposition,TESS Mag,Date TOI Alerted (UTC)\n"
    """1,100.01,3.5,2459000.5,2.1,800,CP,10.0,2019-06-01
2,200.01,5.0,2459001.0,3.0,1500,PC,10.0,2020-01-01
3,300.01,2.0,2459002.0,1.5,900,KP,13.0,2019-06-01
4,400.01,4.0,2459003.0,2.0,700,CP,9.0,2019-06-01
4,400.02,9.0,2459004.0,2.5,300,PC,9.0,2021-06-01
5,500.01,,2459005.0,2.0,600,KP,9.0,2019-06-01
"""
)


def test_select_real_targets_keeps_single_confirmed_bright_hosts_with_a_full_ephemeris(monkeypatch):
    """Of five TOI hosts in the sector only TIC 1 passes: TIC 2 is a candidate, TIC 3 is too faint, TIC 4
    has a second TOI and TIC 5 has no period. The target carries the TOI entry as the planet and the file
    is marked real.
    """
    monkeypatch.setattr(targets_module, "sector_two_minute_tics", lambda sector: [1, 2, 3, 4, 5, 6])
    monkeypatch.setattr(targets_module, "tic_params", lambda tics: {tic: dict(GOOD) for tic in tics})
    selection = select_real_targets(98, tmag_max=12.0, toi_list=ToiList(REAL_TOI_CSV))
    assert selection["real"] is True and [target["tic"] for target in selection["targets"]] == [1]
    planet = selection["targets"][0]["planet"]
    assert planet["toi"] == "100.01" and planet["period"] == 3.5 and planet["alerted"] == "2019-06-01"
    assert (
        selection["n_single_confirmed"] == 1
        and selection["toi_csv"]["sha256"] == ToiList(REAL_TOI_CSV).sha256
    )
