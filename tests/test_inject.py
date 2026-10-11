"""Unit tests for stargazer.inject: orbital geometry, the transit model, sampling and derived statistics."""

import numpy as np
import pytest

from stargazer.inject import (
    TransitTruth,
    a_rs_from_star,
    expected_snr,
    in_transit_mask,
    inject,
    limb_darkening_for_teff,
    make_truth,
    point_scatter,
    sample_truth,
    transit_count,
    transit_duration,
    transit_model,
)


def test_a_rs_from_star_sun_at_one_year_is_one_au():
    """Kepler's third law for the Sun must give 1 au, which is 215.03 solar radii, within 1 percent."""
    assert abs(a_rs_from_star(365.25, 4.438, 1.0) / 215.03 - 1) < 0.01


def test_a_rs_scales_with_period_to_the_two_thirds():
    """Doubling the period multiplies a/R* by 2**(2/3); this pins the exponent, not just one value."""
    ratio = a_rs_from_star(2.0, 4.4, 1.0) / a_rs_from_star(1.0, 4.4, 1.0)
    assert ratio == pytest.approx(2 ** (2 / 3))


def test_transit_duration_earth_sun_is_thirteen_hours():
    """A central Earth transit lasts about 13.1 hours; a wrong formula is off by hours, not minutes."""
    t14 = transit_duration(period=365.25, a_rs=215.03, inc_deg=90.0, rp_rs=0.0092, b=0.0)
    assert abs(t14 * 24 - 13.1) < 0.2


def test_transit_duration_shrinks_with_impact_parameter():
    """A grazing chord is shorter than a central one."""
    central = transit_duration(3.0, 10.0, 90.0, 0.1, 0.0)
    grazing = transit_duration(3.0, 10.0, 86.0, 0.1, 0.7)
    assert grazing < central


def test_transit_duration_is_zero_when_planet_misses_the_disc():
    """With b above 1 + k there is no chord; the function returns zero instead of a NaN."""
    assert transit_duration(period=3.0, a_rs=10.0, inc_deg=80.0, rp_rs=0.01, b=1.5) == 0.0


def test_limb_darkening_default_when_teff_unknown():
    """A host with no Teff gets the solar-type row, so injection never fails on a missing Teff."""
    assert limb_darkening_for_teff(None) == (0.40, 0.25)


def test_limb_darkening_picks_row_by_upper_bound():
    """Bounds are exclusive on the low side: exactly 4000 K belongs to the 4000 to 5000 row."""
    assert limb_darkening_for_teff(3500.0) == (0.30, 0.35)
    assert limb_darkening_for_teff(4000.0) == (0.45, 0.20)
    assert limb_darkening_for_teff(9000.0) == (0.33, 0.28)


def test_limb_darkening_accepts_custom_table():
    """The table is an argument, so a finer grid can replace it without a code change."""
    table = ((5000.0, 0.1, 0.1), (np.inf, 0.2, 0.2))
    assert limb_darkening_for_teff(7000.0, table=table) == (0.2, 0.2)


def test_make_truth_derives_geometry(star):
    """Depth 1 percent gives k = 0.1; b = 0 gives 90 degrees; the chord gives a positive duration."""
    truth = make_truth(period=3.0, t0=0.0, depth=0.01, b=0.0, star=star)
    assert truth.rp_rs == pytest.approx(0.1)
    assert truth.inc_deg == pytest.approx(90.0)
    assert 0 < truth.duration < 0.5


def test_make_truth_rejects_missing_or_nan_star_parameters():
    """A NaN radius from the catalogue must raise, not silently become a solar value in the truth."""
    for bad in ({"logg": None, "radius_rsun": 1.0}, {"logg": 4.4, "radius_rsun": float("nan")}, {}):
        with pytest.raises(ValueError):
            make_truth(period=3.0, t0=0.0, depth=1e-3, b=0.0, star=bad)


def test_truth_round_trips_through_dict_and_ignores_extra_keys(star):
    """truth.json may gain keys in a later version; ``from_dict`` must read older and newer files."""
    truth = make_truth(period=3.0, t0=0.5, depth=0.001, b=0.3, star=star)
    assert TransitTruth.from_dict({**truth.to_dict(), "extra": 1}) == truth


def test_transit_model_depth_matches_rp_rs_squared(lc, star):
    """The central dip is the area ratio, deepened by limb darkening by at most 40 percent."""
    time, _, _ = lc
    truth = make_truth(period=3.0, t0=time.min() + 1.0, depth=0.01, b=0.0, star=star)
    dip = 1.0 - transit_model(time, truth).min()
    assert 0.01 <= dip <= 0.014


def test_transit_model_is_one_out_of_transit(lc, star):
    """Multiplicative injection relies on the model being exactly one away from the transit."""
    time, _, _ = lc
    truth = make_truth(period=3.0, t0=time.min() + 1.0, depth=0.01, b=0.0, star=star)
    model = transit_model(time, truth)
    assert np.all(model[~in_transit_mask(time, truth)] == 1.0)


def test_transit_model_width_matches_duration(lc, star):
    """The time the model spends below one must equal the T14 the truth record claims, within 10 percent."""
    time, _, _ = lc
    truth = make_truth(period=3.0, t0=time.min() + 1.0, depth=0.01, b=0.0, star=star)
    model = transit_model(time, truth)
    in_first = time[(model < 1.0 - 1e-9) & (time < time.min() + 2.5)]
    assert abs((in_first.max() - in_first.min()) / truth.duration - 1) < 0.1


def test_inject_leaves_out_of_transit_flux_unchanged(lc, star):
    """The real noise must survive injection untouched outside the transit windows."""
    time, flux, _ = lc
    truth = make_truth(period=3.0, t0=time.min() + 1.0, depth=0.005, b=0.3, star=star)
    out = inject(time, flux, truth)
    mask = in_transit_mask(time, truth)
    assert np.array_equal(out[~mask], flux[~mask])
    assert out[mask].mean() < flux[mask].mean() - 0.003


def test_sample_truth_is_deterministic_for_a_seed(lc, star):
    """Reproducibility of an episode set rests on the sampler being a pure function of the seed."""
    time, _, _ = lc
    first = sample_truth(np.random.default_rng(42), time, star)
    assert first == sample_truth(np.random.default_rng(42), time, star)
    assert first != sample_truth(np.random.default_rng(43), time, star)


def test_sample_truth_respects_ranges(lc, star):
    """Every draw stays inside the period, depth and impact-parameter bounds over 20 seeds."""
    time, _, _ = lc
    for seed in range(20):
        truth = sample_truth(
            np.random.default_rng(seed), time, star, period_range=(1, 2), depth_range=(1e-3, 2e-3)
        )
        assert 1 <= truth.period <= 2 and 1e-3 <= truth.depth <= 2e-3 and 0 <= truth.b <= 0.9
        assert time.min() <= truth.t0 < time.min() + truth.period


def test_point_scatter_recovers_gaussian_sigma():
    """On pure Gaussian noise the MAD-based estimator must return the true sigma within 3 percent."""
    flux = 1.0 + np.random.default_rng(0).normal(0, 2e-3, 50_000)
    assert abs(point_scatter(flux) / 2e-3 - 1) < 0.03


def test_point_scatter_ignores_slow_trend():
    """A 5 percent sinusoid is 50 times the noise; differencing must remove it from the estimate."""
    time = np.linspace(0, 27, 20_000)
    flux = 1.0 + 0.05 * np.sin(time) + np.random.default_rng(1).normal(0, 1e-3, time.size)
    assert abs(point_scatter(flux) / 1e-3 - 1) < 0.05


def test_in_transit_mask_selects_half_duration_window(lc, star):
    """The mask is the set of points within T14/2 of a transit centre, folded on the period."""
    time, _, _ = lc
    truth = make_truth(period=5.0, t0=time.min() + 0.5, depth=0.003, b=0.0, star=star)
    phase = ((time - truth.t0 + 2.5) % 5.0) - 2.5
    expected = np.abs(phase) < 0.5 * truth.duration
    assert np.array_equal(in_transit_mask(time, truth), expected) and expected.sum() > 0


def test_transit_count_counts_events_in_data(lc, star):
    """A 5-day period starting at 0.5 days gives six events in 27 days: 0.5, 5.5, 10.5, 15.5, 20.5, 25.5."""
    time, _, _ = lc
    truth = make_truth(period=5.0, t0=time.min() + 0.5, depth=0.003, b=0.0, star=star)
    assert transit_count(time, truth) == 6


def test_transit_count_single_event():
    """A period longer than the data with the epoch inside it gives exactly one event."""
    time = np.arange(0, 27, 2 / 1440) + 1000.0
    truth = make_truth(period=40.0, t0=1020.0, depth=0.01, b=0.0, star={"logg": 4.4, "radius_rsun": 1.0})
    assert transit_count(time, truth) == 1


def test_transit_count_zero_when_event_is_in_gap(lc, star):
    """An event centred in the 13 to 14 day gap has no points and is not counted."""
    time, _, _ = lc
    truth = make_truth(period=100.0, t0=time.min() + 13.5, depth=0.01, b=0.0, star=star)
    assert transit_count(time, truth) == 0


def test_expected_snr_formula(lc, star):
    """SNR is depth over sigma times the square root of the in-transit point count."""
    time, flux, _ = lc
    truth = make_truth(period=5.0, t0=time.min() + 0.5, depth=0.003, b=0.0, star=star)
    n_in = in_transit_mask(time, truth).sum()
    assert expected_snr(time, flux, truth) == pytest.approx(0.003 / point_scatter(flux) * np.sqrt(n_in))


def test_expected_snr_zero_without_in_transit_points(lc, star):
    """An injection that lands in a gap has no signal in the data, so its SNR is zero, not NaN."""
    time, flux, _ = lc
    truth = make_truth(period=100.0, t0=time.min() + 13.5, depth=0.01, b=0.0, star=star)
    assert expected_snr(time, flux, truth) == 0.0
