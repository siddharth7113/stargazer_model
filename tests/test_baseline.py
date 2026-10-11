from stargazer.baseline import BLSDetector
from stargazer.inject import inject, make_truth
from stargazer.score import score_episode


def test_baseline_recovers_strong_injection(lc, star):
    time, flux, fe = lc
    truth = make_truth(period=3.0, t0=time.min() + 1.0, depth=0.01, b=0.2, star=star)
    f = inject(time, flux, truth)
    pred = BLSDetector().detect(time - time.min(), f, fe)
    assert pred.detected, pred
    row = score_episode(
        pred, {"kind": "transit", "transit": {**truth.to_dict(), "t0": truth.t0 - time.min()}}
    )
    assert row["period_credit"] == 1.0, (pred, row)
    assert row["epoch_ok"], (pred, row)


def test_baseline_quiet_on_pure_noise(lc):
    time, flux, fe = lc
    pred = BLSDetector().detect(time - time.min(), flux, fe)
    assert not pred.detected, pred


def test_baseline_misses_single_transit_by_design(lc, star):
    time, flux, fe = lc
    truth = make_truth(period=40.0, t0=time.min() + 20.0, depth=0.01, b=0.0, star=star)
    f = inject(time, flux, truth)
    pred = BLSDetector().detect(time - time.min(), f, fe)
    row = score_episode(pred, {"kind": "transit", "transit": {**truth.to_dict(), "t0": 20.0}})
    assert row["outcome"] != "hit"


def test_config_is_recorded_and_validated():
    det = BLSDetector(sde_threshold=9.0)
    assert det.config["sde_threshold"] == 9.0
    try:
        BLSDetector(nope=1)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown key accepted")
