"""Unit tests for stargazer.scripted: the rules prompt applied as fixed code over the tools.

The light curves are the white-noise fixtures of ``conftest.py``: 27 days at 2-minute cadence,
noise of 1,000 ppm and a gap from day 13 to day 14. Each test adds one signal with a known answer.
"""

import numpy as np
import pytest

from stargazer.inject import inject, make_truth
from stargazer.scripted import FoldReading, ScriptedRulesDetector


def test_fold_reading_measures_the_dip_at_half_phase_and_the_deepest_bin_away_from_the_transit():
    """The reading must take the depth at phase 0.5 from the outermost bins and the flatness from the
    bins between, so that a second dip and a bump in the orbit are told apart from the transit itself.
    """
    phase = np.linspace(-0.5, 0.5, 21).tolist()
    flux = [0.0] * 21
    flux[10] = -1000.0
    flux[0] = flux[20] = -600.0
    flux[5] = -300.0
    flux[15] = None
    reading = FoldReading({"phase": phase, "flux_ppm": flux}, duration=0.05)
    assert reading.half_phase_depth_ppm() == pytest.approx(600.0)
    assert reading.deepest_away_ppm() == pytest.approx(300.0)


def test_scripted_detector_finds_a_deep_periodic_transit_at_its_period(injected):
    """A 1 percent transit every 3 days passes every rule, so the detector submits the period within
    1 percent, the tolerance of the scorer.
    """
    time, flux, flux_err, truth = injected
    prediction = ScriptedRulesDetector().detect(time, flux, flux_err)
    assert prediction.detected and prediction.period == pytest.approx(truth.period, rel=0.01)


def test_scripted_detector_declines_white_noise(lc):
    """A light curve of noise only holds no signal that meets the rules, so the detector submits
    detected=false, which scores full marks on a star with no planet.
    """
    time, flux, flux_err = lc
    prediction = ScriptedRulesDetector().detect(time - time.min(), flux, flux_err)
    assert not prediction.detected


def test_scripted_detector_submits_a_lone_deep_event_as_a_single_event(lc):
    """One 5,000 ppm dip of 5 hours with no partner meets the single-event rule, so the detector
    submits it with its time and no period.
    """
    time, flux, flux_err = lc
    relative = time - time.min()
    flux = flux.copy()
    flux[np.abs(relative - 20.0) < 0.1] -= 0.005
    prediction = ScriptedRulesDetector().detect(relative, flux, flux_err)
    assert prediction.detected and prediction.period is None
    assert prediction.t0 == pytest.approx(20.0, abs=0.1)


def test_scripted_detector_declines_an_eclipsing_binary_with_a_secondary_eclipse(lc, star):
    """A 1 percent primary and a 0.5 percent secondary at phase 0.5 is a binary: odd_even fails at half
    the period and secondary_eclipse fails at the period, so the detector submits detected=false.
    """
    time, flux, flux_err = lc
    relative = time - time.min()
    primary = make_truth(period=3.0, t0=1.0, depth=0.01, b=0.2, star=star)
    secondary = make_truth(period=3.0, t0=2.5, depth=0.005, b=0.2, star=star)
    flux = inject(relative, inject(relative, flux, primary), secondary)
    prediction = ScriptedRulesDetector().detect(relative, flux, flux_err)
    assert not prediction.detected


def test_scripted_detector_records_every_tool_call_in_the_agents_transcript_form(lc):
    """The transcript holds one entry for each tool call with the tool, the arguments and the result,
    the form the trajectory viewer reads for the agent, and the count matches it.
    """
    time, flux, flux_err = lc
    prediction = ScriptedRulesDetector().detect(time - time.min(), flux, flux_err)
    transcript = prediction.extra["transcript"]
    assert transcript[0]["tool"] == "summarize" and set(transcript[0]) == {"tool", "arguments", "result"}
    assert prediction.extra["n_tool_calls"] == len(transcript)


def test_scripted_detector_refuses_an_unknown_config_key():
    """A misspelt override must fail at construction, so a results file never records a number that
    the detector did not use.
    """
    with pytest.raises(ValueError, match="unknown config keys"):
        ScriptedRulesDetector(min_significance=5.0)
