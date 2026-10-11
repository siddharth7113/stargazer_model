"""Unit tests for stargazer.tools: each analysis tool, the schema validator and the dispatcher."""

import numpy as np
import pytest

from stargazer import tools
from stargazer.inject import inject, make_truth
from stargazer.tools import TOOLS, TOOLS_BY_NAME, ToolSpec, call_tool, tool_definitions


def test_summarize_cadence_and_scatter(lc):
    """The summary reports the 2-minute cadence and the 1000 ppm noise the fixture was built with."""
    time, flux, _ = lc
    summary = tools.summarize(time - time.min(), flux)
    assert summary["n_points"] == len(time)
    assert abs(summary["cadence_minutes"] - 2.0) < 0.01
    assert abs(summary["point_scatter_ppm"] / 1000.0 - 1) < 0.05


def test_summarize_reports_gap(lc):
    """The fixture has one 1-day gap at day 13; the summary must list exactly that."""
    time, flux, _ = lc
    gaps = tools.summarize(time - time.min(), flux)["gaps_days"]
    assert len(gaps) == 1 and abs(gaps[0][0] - 13.0) < 0.01 and abs(gaps[0][1] - 1.0) < 0.01


def test_bls_peaks_top_peak_is_injection(injected):
    """The strongest peak must be the injected 3-day period with an SDE well above the noise level."""
    time, flux, _, truth = injected
    peaks = tools.bls_peaks(time, flux)["peaks"]
    assert abs(peaks[0]["period"] / truth.period - 1) < 0.01 and peaks[0]["sde"] > 7


def test_bls_peaks_are_distinct_periods(injected):
    """Five peaks must be five different periods, not five samples of one wide peak."""
    time, flux, _, _ = injected
    periods = [peak["period"] for peak in tools.bls_peaks(time, flux)["peaks"]]
    assert len(periods) == 5
    assert all(
        abs(first / second - 1) >= 0.02 for index, first in enumerate(periods) for second in periods[:index]
    )


def test_fold_dip_is_at_phase_zero(injected):
    """Folding at the true period and epoch puts the deepest bin at phase zero with the injected depth."""
    time, flux, _, truth = injected
    curve = tools.fold(time, flux, period=truth.period, t0=truth.t0, phase_half_width=0.05)
    deepest = int(np.nanargmin([np.nan if value is None else value for value in curve["flux_ppm"]]))
    assert abs(curve["phase"][deepest]) < 0.02
    assert 8000 < curve["depth_ppm"] < 14000


def test_fold_at_wrong_period_has_no_dip(injected):
    """Folding at an unrelated period smears the transit; no bin is deeper than a few hundred ppm."""
    time, flux, _, _ = injected
    curve = tools.fold(time, flux, period=4.37, t0=0.0, phase_half_width=0.5)
    assert curve["depth_ppm"] < 2000


def test_fold_empty_bins_are_null(lc):
    """With 2 days of data folded at 10 days, most phase bins have no points and must be null, not NaN."""
    time = np.arange(0, 2, 2 / 1440)
    curve = tools.fold(time, np.ones_like(time), period=10.0, t0=0.0, phase_half_width=0.5)
    assert None in curve["flux_ppm"] and curve["n_empty_bins"] > 0 and curve["n_points"] == time.size


def test_odd_even_equal_for_planet(injected):
    """Odd and even transits of one planet have the same depth: below 3 sigma and verdict consistent."""
    time, flux, _, truth = injected
    result = tools.odd_even(time, flux, period=truth.period, t0=truth.t0, duration=truth.duration)
    assert result["odd_events"] >= 2 and result["even_events"] >= 2
    assert result["difference_sigma"] < 3 and result["verdict"] == "consistent"


def test_odd_even_flags_binary(injected):
    """Making the odd events 0.3 percent shallower, like a secondary eclipse, gives verdict inconsistent."""
    time, flux, _, truth = injected
    epoch = np.round((time - truth.t0) / truth.period).astype(int)
    phase = ((time - truth.t0 + 1.5) % 3.0) - 1.5
    shallow = (np.abs(phase) < 0.5 * truth.duration) & (epoch % 2 == 1)
    binary = flux.copy()
    binary[shallow] = 1.0 - 0.003
    result = tools.odd_even(time, binary, period=truth.period, t0=truth.t0, duration=truth.duration)
    assert result["difference_sigma"] > 3 and result["verdict"] == "inconsistent"


def test_odd_even_tolerates_slow_variability(injected):
    """A 5 percent, 7-day sinusoid left after injection must not turn a planet into a binary verdict."""
    time, flux, _, truth = injected
    active = flux * (1.0 + 0.05 * np.sin(2 * np.pi * time / 7.0))
    result = tools.odd_even(time, active, period=truth.period, t0=truth.t0, duration=truth.duration)
    assert result["verdict"] == "consistent", result


def test_odd_even_with_one_event_per_parity_is_insufficient():
    """Two events in total give one per parity; no standard error exists, so the verdict says so."""
    time = np.arange(0, 4, 2 / 1440)  # events at day 1 (even) and day 3 (odd) only
    flux = np.ones_like(time)
    result = tools.odd_even(time, flux, period=2.0, t0=1.0, duration=0.1)
    assert result["odd_events"] == 1 and result["even_events"] == 1
    assert result["difference_sigma"] is None and result["verdict"] == "insufficient_data"


def test_odd_even_without_points_has_null_depth():
    """One day of data folded at 10 days with the epoch at day 5 has no in-transit points of either parity."""
    time = np.arange(0, 1, 2 / 1440)
    result = tools.odd_even(time, np.ones_like(time), period=10.0, t0=5.0, duration=0.1)
    assert result["odd_depth_ppm"] is None and result["verdict"] == "insufficient_data"


def test_every_tool_definition_is_well_formed():
    """Each registry entry renders to the OpenAI function format and its required keys exist in properties."""
    for spec in TOOLS:
        definition = spec.definition()
        assert definition["type"] == "function" and definition["function"]["name"] == spec.name
        assert spec.parameters["type"] == "object"
        assert set(spec.parameters.get("required", [])) <= set(spec.parameters["properties"])
    assert [tool["function"]["name"] for tool in tool_definitions()] == [spec.name for spec in TOOLS]


def _spec(**properties):
    """A throwaway ToolSpec with one required argument ``a_value`` and the given property schemas."""
    schema = {"type": "object", "properties": properties, "required": ["a_value"]}
    return ToolSpec("x", "x", schema, lambda *args, **kwargs: {})


def test_validate_accepts_integer_as_number():
    """JSON has one numeric type; a model that writes ``3`` for a number must not be rejected."""
    assert _spec(a_value={"type": "number"}).validate({"a_value": 3}) == []


def test_validate_missing_required():
    """A required argument that is absent is reported by name."""
    assert _spec(a_value={"type": "number"}).validate({}) == ["missing required argument: a_value"]


def test_validate_unknown_argument():
    """An argument the schema does not list is reported, since the tool would otherwise ignore it silently."""
    problems = _spec(a_value={"type": "number"}).validate({"a_value": 1, "zz": 2})
    assert problems == ["unknown argument: zz"]


def test_validate_wrong_type():
    """A string where a number is expected is reported with the expected and actual types."""
    problems = _spec(a_value={"type": "number"}).validate({"a_value": "3"})
    assert problems == ["argument a_value must be number, got str"]


def test_validate_nullable_type():
    """A property typed ``["number", "null"]`` accepts None."""
    assert _spec(a_value={"type": ["number", "null"]}).validate({"a_value": None}) == []


def test_validate_boolean_rejects_string():
    """``"yes"`` is not a boolean; a model that writes it must be corrected."""
    problems = _spec(a_value={"type": "boolean"}).validate({"a_value": "yes"})
    assert problems == ["argument a_value must be boolean, got str"]


def test_validate_exclusive_minimum():
    """A period of zero would divide by zero downstream; ``exclusiveMinimum`` rejects it."""
    spec = _spec(a_value={"type": "number", "exclusiveMinimum": 0})
    assert spec.validate({"a_value": 0}) == ["argument a_value must be greater than 0, got 0"]
    assert spec.validate({"a_value": 0.1}) == []


def test_registry_periods_durations_and_depth_are_positive():
    """Every period, duration, width and depth argument in the registry carries the positive constraint."""
    constrained = {
        (name, key)
        for name, spec in TOOLS_BY_NAME.items()
        for key, schema in spec.parameters["properties"].items()
        if "exclusiveMinimum" in schema
    }
    assert constrained == {
        ("bls_peaks", "min_period"),
        ("bls_peaks", "max_period"),
        ("bls_peaks", "detrend_window_days"),
        ("fold", "period"),
        ("fold", "phase_half_width"),
        ("fold", "detrend_window_days"),
        ("odd_even", "period"),
        ("odd_even", "duration"),
        ("single_events", "detrend_window_days"),
        ("submit_prediction", "period"),
        ("submit_prediction", "duration"),
        ("submit_prediction", "depth"),
    }


def test_submit_periodic_detection_needs_t0(lc):
    """A detection with a period but no epoch returns an error, so the episode does not end half answered."""
    time, flux, _ = lc
    result = call_tool("submit_prediction", {"detected": True, "period": 3.0}, time, flux)
    assert result == {"error": "a periodic detection needs t0", "form": "periodic"}


def test_submit_single_event_needs_t0_duration_and_depth(lc):
    """A detection without a period is a single event and must name the dip by time, duration and depth."""
    time, flux, _ = lc
    result = call_tool("submit_prediction", {"detected": True, "t0": 5.0}, time, flux)
    assert result["error"] == "a single_event detection needs duration, depth"
    complete = {"detected": True, "t0": 5.0, "duration": 0.1, "depth": 0.001}
    assert call_tool("submit_prediction", complete, time, flux) == {"ok": True, "form": "single_event"}


def test_submit_no_detection_needs_nothing_else(lc):
    """``detected=false`` is a complete answer on its own."""
    time, flux, _ = lc
    assert call_tool("submit_prediction", {"detected": False}, time, flux) == {"ok": True}


def test_call_tool_unknown_name(lc):
    """An unknown tool name returns an error plus the list of available tools, so the model can recover."""
    time, flux, _ = lc
    result = call_tool("nope", {}, time, flux)
    assert result["error"] == "unknown tool: nope" and "bls_peaks" in result["available_tools"]


def test_call_tool_invalid_arguments_return_error(lc):
    """Validation problems come back with the expected arguments, so the model can fix the call."""
    time, flux, _ = lc
    result = call_tool("fold", {"period": 3.0}, time, flux)
    assert result["error"] == "missing required argument: t0"
    assert result["required"] == ["period", "t0"] and "t0" in result["expected_arguments"]


def test_call_tool_exception_becomes_error(lc, monkeypatch):
    """A crash inside a tool is reported to the model as an error, not raised into the agent loop."""
    time, flux, _ = lc
    schema = {"type": "object", "properties": {}, "required": []}
    monkeypatch.setitem(TOOLS_BY_NAME, "boom", ToolSpec("boom", "x", schema, lambda *args, **kwargs: 1 / 0))
    assert call_tool("boom", {}, time, flux)["error"].startswith("ZeroDivisionError")


def test_bls_peaks_detrend_window_is_applied(injected):
    """The window argument reaches the detrend step: a different window changes the peak statistics."""
    time, flux, _, _ = injected
    wide = tools.bls_peaks(time, flux, detrend_window_days=0.75)["peaks"][0]
    narrow = tools.bls_peaks(time, flux, detrend_window_days=0.2)["peaks"][0]
    assert wide["sde"] != narrow["sde"]


def test_bls_peaks_derived_fields(injected):
    """Each peak carries the alias periods and the event count, so the model does no arithmetic."""
    time, flux, _, _ = injected
    top = tools.bls_peaks(time, flux)["peaks"][0]
    assert top["half_period"] == pytest.approx(top["period"] / 2, rel=1e-4)
    assert top["double_period"] == pytest.approx(top["period"] * 2, rel=1e-4)
    assert top["n_events_in_data"] == 9  # 27 days at 3 days from t0 = 1: events at 1, 4, ..., 25


def test_bls_peaks_mask_reveals_weaker_signal(lc, star):
    """With a 1 percent, 3-day signal masked, a 0.2 percent, 7-day signal becomes the top peak."""
    time, flux, _ = lc
    relative = time - time.min()
    strong = make_truth(period=3.0, t0=1.0, depth=0.01, b=0.0, star=star)
    weak = make_truth(period=7.0, t0=2.0, depth=0.002, b=0.0, star=star)
    both = inject(relative, inject(relative, flux, strong), weak)
    top_before = tools.bls_peaks(relative, both)["peaks"][0]["period"]
    masked = tools.bls_peaks(relative, both, masks=[{"period": 3.0, "t0": 1.0, "duration": strong.duration}])
    assert abs(top_before / 3.0 - 1) < 0.01
    assert abs(masked["peaks"][0]["period"] / 7.0 - 1) < 0.01 and masked["n_points_masked"] > 0


def test_bls_peaks_masks_argument_is_validated():
    """A mask item missing a key is rejected by the schema, not by a crash inside the tool."""
    spec = TOOLS_BY_NAME["bls_peaks"]
    assert spec.validate({"masks": [{"period": 3.0, "t0": 1.0, "duration": 0.1}]}) == []
    assert any("must be array" in problem for problem in spec.validate({"masks": "3.0"}))


def test_single_events_finds_a_lone_transit(lc, star):
    """One 1 percent event at day 20 with no repeat is found at the right time with a high SNR."""
    time, flux, _ = lc
    relative = time - time.min()
    truth = make_truth(period=40.0, t0=20.0, depth=0.01, b=0.0, star=star)
    events = tools.single_events(relative, inject(relative, flux, truth))["events"]
    assert abs(events[0]["time"] - 20.0) < 0.05 and events[0]["snr"] > 20
    assert all(abs(event["time"] - 20.0) > 0.2 for event in events[1:])  # no duplicate of the same dip


def test_single_events_on_noise_are_weak(lc):
    """Pure noise gives only low-SNR events, well below the level a real dip reaches."""
    time, flux, _ = lc
    events = tools.single_events(time - time.min(), flux)["events"]
    assert events and events[0]["snr"] < 8


def test_single_events_skips_boxes_across_the_gap(lc, star):
    """A box spanning the 1-day gap would mix two separated segments; no event is reported there."""
    time, flux, _ = lc
    relative = time - time.min()
    events = tools.single_events(relative, flux)["events"]
    assert all(not (12.9 < event["time"] < 14.1) for event in events)
