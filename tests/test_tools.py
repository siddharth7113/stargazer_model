"""Unit tests for stargazer.tools: each analysis tool, the schema validator and the dispatcher."""

import numpy as np

from stargazer import tools
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
    assert None in curve["flux_ppm"] and curve["n"][0] == 0


def test_odd_even_equal_for_planet(injected):
    """Odd and even transits of one planet have the same depth, so the difference is below 3 sigma."""
    time, flux, _, truth = injected
    result = tools.odd_even(time, flux, period=truth.period, t0=truth.t0, duration=truth.duration)
    assert result["odd_n"] > 0 and result["even_n"] > 0 and result["difference_sigma"] < 3


def test_odd_even_flags_binary(injected):
    """Making the odd events 0.3 percent shallower, like a secondary eclipse, must exceed 3 sigma."""
    time, flux, _, truth = injected
    epoch = np.round((time - truth.t0) / truth.period).astype(int)
    phase = ((time - truth.t0 + 1.5) % 3.0) - 1.5
    shallow = (np.abs(phase) < 0.5 * truth.duration) & (epoch % 2 == 1)
    binary = flux.copy()
    binary[shallow] = 1.0 - 0.003
    result = tools.odd_even(time, binary, period=truth.period, t0=truth.t0, duration=truth.duration)
    assert result["difference_sigma"] > 3


def test_odd_even_without_points_has_null_depth():
    """One day of data folded at 10 days with the epoch at day 5 has no in-transit points of either parity."""
    time = np.arange(0, 1, 2 / 1440)
    result = tools.odd_even(time, np.ones_like(time), period=10.0, t0=5.0, duration=0.1)
    assert result["odd_depth_ppm"] is None and result["difference_sigma"] is None


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
        ("fold", "period"),
        ("fold", "phase_half_width"),
        ("odd_even", "period"),
        ("odd_even", "duration"),
        ("submit_prediction", "period"),
        ("submit_prediction", "duration"),
        ("submit_prediction", "depth"),
    }


def test_call_tool_unknown_name(lc):
    """An unknown tool name returns an error naming the available tools, so the model can recover."""
    time, flux, _ = lc
    error = call_tool("nope", {}, time, flux)["error"]
    assert error.startswith("unknown tool: nope") and "bls_peaks" in error


def test_call_tool_invalid_arguments_return_error(lc):
    """Validation problems come back as the tool result, joined by semicolons."""
    time, flux, _ = lc
    assert call_tool("fold", {"period": 3.0}, time, flux) == {"error": "missing required argument: t0"}


def test_call_tool_exception_becomes_error(lc, monkeypatch):
    """A crash inside a tool is reported to the model as an error, not raised into the agent loop."""
    time, flux, _ = lc
    schema = {"type": "object", "properties": {}, "required": []}
    monkeypatch.setitem(TOOLS_BY_NAME, "boom", ToolSpec("boom", "x", schema, lambda *args, **kwargs: 1 / 0))
    assert call_tool("boom", {}, time, flux)["error"].startswith("ZeroDivisionError")
