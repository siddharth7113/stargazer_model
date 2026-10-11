"""Unit tests for stargazer.context: the TOI list parser, the context record and the writer."""

import json

from stargazer.context import BTJD_OFFSET, CatalogueFetcher, ContextWriter, EpisodeContext, ToiList, tic_id

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
