"""Unit tests for stargazer.data: the cache layout and loading. Downloading is not tested; it needs MAST."""

import json
import sys

import numpy as np
import pytest

from stargazer.data import LightCurve, LightCurveCache


def test_path_layout(tmp_path):
    """The file name encodes the TIC id and the zero-padded sector."""
    assert LightCurveCache(tmp_path).path(123, 7).name == "tic123_s0007.npz"


def test_has_is_false_before_fetch(tmp_path):
    """An empty cache has nothing."""
    assert not LightCurveCache(tmp_path).has(1, 1)


def test_load_returns_arrays_and_meta(tmp_path):
    """A cached file loads into a LightCurve with its metadata and a span in days."""
    cache = LightCurveCache(tmp_path)
    np.savez_compressed(
        cache.path(1, 1),
        time=np.array([1.0, 2.0]),
        flux=np.array([1.0, 1.0]),
        flux_err=np.array([0.1, 0.1]),
        meta=json.dumps({"tic": 1, "teff": 5000.0}),
    )
    lc = cache.load(1, 1)
    assert isinstance(lc, LightCurve) and lc.meta["teff"] == 5000.0 and lc.span == 1.0


def test_load_missing_raises(tmp_path):
    """Loading a target that was never fetched is an error, not an empty light curve."""
    with pytest.raises(FileNotFoundError):
        LightCurveCache(tmp_path).load(5, 5)


def test_fetch_returns_cached_path_without_download(tmp_path, monkeypatch):
    """If the file exists, ``fetch`` must not import lightkurve; any download attempt here fails loudly."""
    cache = LightCurveCache(tmp_path)
    cache.path(1, 1).write_bytes(b"x")
    monkeypatch.setitem(sys.modules, "lightkurve", None)
    assert cache.fetch(1, 1) == cache.path(1, 1)
