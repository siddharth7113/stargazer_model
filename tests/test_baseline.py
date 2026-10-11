import numpy as np

from stargazer.baseline import detect
from stargazer.inject import inject, make_truth
from stargazer.score import score_episode


def test_baseline_recovers_strong_injection(lc, star):
    time, flux, fe = lc
    truth = make_truth(period=3.0, t0=time.min() + 1.0, depth=0.01, b=0.2, star=star)
    f = inject(time, flux, truth)
    t_rel = time - time.min()
    pred = detect(t_rel, f, fe)
    assert pred["detected"], pred
    row = score_episode(pred, {"kind": "transit", "transit": {**truth.to_dict(), "t0": truth.t0 - time.min()}})
    assert row["period_credit"] == 1.0, (pred, row)
    assert row["epoch_ok"], (pred, row)


def test_baseline_quiet_on_pure_noise(lc):
    time, flux, fe = lc
    pred = detect(time - time.min(), flux, fe)
    assert not pred["detected"], pred


def test_baseline_misses_single_transit_by_design(lc, star):
    # BLS needs two or more events; a single 1 percent transit should not score an exact hit.
    time, flux, fe = lc
    truth = make_truth(period=40.0, t0=time.min() + 20.0, depth=0.01, b=0.0, star=star)
    f = inject(time, flux, truth)
    pred = detect(time - time.min(), f, fe)
    row = score_episode(pred, {"kind": "transit", "transit": {**truth.to_dict(), "t0": 20.0}})
    assert row["outcome"] != "hit"
