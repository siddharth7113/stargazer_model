"""Unit tests for stargazer.baseline: the detrend step, the binning step and the BLS detector."""

import numpy as np
import pytest

from stargazer.baseline import BLSDetector, bin_lc, detrend


def test_detrend_removes_slow_trend_in_the_interior():
    """A 2 percent, 5-day sinusoid must be flattened to below 0.1 percent away from the edges.

    The Savitzky-Golay fit is less constrained in the last half window at each end, so the
    check excludes the outer 0.4 days.
    """
    time = np.arange(0, 10, 2 / 1440)
    flux = 1.0 + 0.02 * np.sin(2 * np.pi * time / 5.0)
    out_time, flat = detrend(time, flux, window_days=0.75)
    interior = (out_time > 0.4) & (out_time < 9.6)
    assert np.abs(flat[interior] - 1.0).max() < 1e-3


def test_detrend_keeps_short_transit(injected):
    """A 2-hour transit is far shorter than the 0.75-day window, so 80 percent of its depth survives."""
    time, flux, _, truth = injected
    _, flat = detrend(time, flux, window_days=0.75)
    assert 1.0 - flat.min() > 0.8 * truth.depth


def test_detrend_drops_non_finite_points():
    """A NaN in the input must not propagate; the output is finite and no longer than the input."""
    time = np.arange(0, 5, 2 / 1440)
    flux = np.ones_like(time)
    flux[100] = np.nan
    out_time, out_flux = detrend(time, flux, window_days=0.75)
    assert np.isfinite(out_flux).all() and len(out_time) == len(out_flux) < len(time)


def test_bin_lc_means_and_counts():
    """One day at 1-minute cadence binned to 10 minutes gives 144 bins; the first is the mean of 0 to 9."""
    time = np.arange(0, 1, 1 / 1440)
    flux = np.arange(time.size, dtype=float)
    binned_time, binned_flux, _ = bin_lc(time, flux, bin_minutes=10)
    assert len(binned_time) == 144
    assert binned_flux[0] == pytest.approx(4.5)
    assert binned_time[0] == pytest.approx(np.mean(time[:10]))


def test_bin_lc_error_is_scatter_over_sqrt_n():
    """The documented error model is the global standard deviation divided by sqrt(points in bin)."""
    time = np.arange(0, 1, 1 / 1440)
    flux = np.random.default_rng(0).normal(1.0, 1e-3, time.size)
    _, _, err = bin_lc(time, flux, bin_minutes=10)
    assert np.allclose(err, flux.std() / np.sqrt(10))


def test_bin_lc_zero_width_returns_input():
    """A non-positive width disables binning and returns the same arrays."""
    time = np.arange(0, 1, 1 / 1440)
    flux = np.ones_like(time)
    out_time, out_flux, _ = bin_lc(time, flux, bin_minutes=0)
    assert out_time is time and out_flux is flux


def test_config_defaults_are_the_documented_values():
    """The results file reproduces a run from ``config``; these are the values the LOGBOOK cites."""
    config = BLSDetector().config
    assert (
        config["sde_threshold"] == 7.0
        and config["detrend_window_days"] == 0.75
        and config["bin_minutes"] == 10.0
    )


def test_config_override_is_recorded():
    """A changed threshold must appear in ``config``, or the results file would misreport the run."""
    assert BLSDetector(sde_threshold=9.0).config["sde_threshold"] == 9.0


def test_config_is_a_copy():
    """Mutating the returned dict must not change the detector."""
    detector = BLSDetector()
    detector.config["sde_threshold"] = 99
    assert detector.config["sde_threshold"] == 7.0


def test_unknown_config_key_is_rejected():
    """A typo in a config key must fail loudly, not be ignored."""
    with pytest.raises(ValueError):
        BLSDetector(nope=1)


def test_detect_recovers_strong_injection(injected):
    """A 1 percent, 3-day signal at SNR above 100 must be found at the right period and epoch."""
    time, flux, flux_err, truth = injected
    pred = BLSDetector().detect(time, flux, flux_err)
    assert pred.detected
    assert abs(pred.period / truth.period - 1) < 0.01
    assert abs(((pred.t0 - truth.t0 + 1.5) % 3.0) - 1.5) < 0.5 * truth.duration
    assert pred.extra["sde"] > 15


def test_detect_reports_statistic_even_when_not_detected(lc):
    """Calibration re-scores a run at other thresholds, so the SDE and best period must always be present."""
    time, flux, flux_err = lc
    pred = BLSDetector(sde_threshold=1e9).detect(time - time.min(), flux, flux_err)
    assert not pred.detected and pred.extra["sde"] > 0 and pred.period > 0


def test_detect_period_grid_is_bounded_by_span(lc):
    """The longest trial period is half the baseline, so at least two transits fit in the data."""
    time, flux, flux_err = lc
    pred = BLSDetector(max_period_frac_of_span=0.5).detect(time - time.min(), flux, flux_err)
    assert 0.5 <= pred.period <= 0.5 * (time.max() - time.min())
