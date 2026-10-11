import numpy as np

from stargazer.inject import (a_rs_from_star, expected_snr, in_transit_mask, inject, make_truth,
                              sample_truth, transit_count, transit_duration, transit_model)


def test_a_rs_sun_like_matches_known_value():
    # Sun-like star, 365.25 d -> 1 AU / R_sun = 215.03
    a = a_rs_from_star(365.25, 4.438, 1.0)
    assert abs(a / 215.03 - 1) < 0.01


def test_depth_at_mid_transit_matches_rp_rs_squared(lc, star):
    time, flux, _ = lc
    truth = make_truth(period=3.0, t0=time.min() + 1.0, depth=0.01, b=0.0, star=star)
    model = transit_model(time, truth)
    dip = 1.0 - model.min()
    # limb darkening makes the central depth a bit deeper than rp_rs**2
    assert 0.01 <= dip <= 0.014, dip


def test_injection_is_multiplicative_and_outside_transit_unchanged(lc, star):
    time, flux, _ = lc
    truth = make_truth(period=3.0, t0=time.min() + 1.0, depth=0.005, b=0.3, star=star)
    out = inject(time, flux, truth)
    m = in_transit_mask(time, truth)
    assert np.allclose(out[~m], flux[~m])
    assert out[m].mean() < flux[m].mean()


def test_duration_formula_matches_model_width(lc, star):
    time, _, _ = lc
    truth = make_truth(period=3.0, t0=time.min() + 1.0, depth=0.01, b=0.0, star=star)
    model = transit_model(time, truth)
    in_tr = time[model < 1.0 - 1e-9]
    first = in_tr[in_tr < time.min() + 2.5]
    width = first.max() - first.min()
    assert abs(width / truth.duration - 1) < 0.1, (width, truth.duration)


def test_sample_truth_is_deterministic(lc, star):
    time, _, _ = lc
    a = sample_truth(np.random.default_rng(42), time, star)
    b = sample_truth(np.random.default_rng(42), time, star)
    assert a == b
    c = sample_truth(np.random.default_rng(43), time, star)
    assert a != c


def test_transit_count_and_snr(lc, star):
    time, flux, _ = lc
    truth = make_truth(period=5.0, t0=time.min() + 0.5, depth=0.003, b=0.0, star=star)
    n = transit_count(time, truth)
    assert n == 6, n  # 27-day span, 5-day period, first at +0.5 -> 0.5,5.5,10.5,15.5,20.5,25.5
    snr = expected_snr(time, flux, truth)
    n_in = in_transit_mask(time, truth).sum()
    assert abs(snr - 0.003 / 1e-3 * np.sqrt(n_in)) / snr < 0.15


def test_single_transit_counted_once(lc, star):
    time, _, _ = lc
    truth = make_truth(period=40.0, t0=time.min() + 20.0, depth=0.01, b=0.0, star=star)
    assert transit_count(time, truth) == 1
