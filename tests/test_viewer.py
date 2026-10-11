"""Unit tests for stargazer.viewer: turn grouping, formatting helpers and rendering, without the UI."""

import asyncio
import io
import json

import pytest
from rich.console import Console
from rich.text import Text

from stargazer.evaluate import Results
from stargazer.viewer import (
    PEAK_COLUMNS,
    EpisodeRow,
    LeftBar,
    ToolCall,
    Turn,
    adjacent_index,
    comparison_rows,
    episode_label,
    episode_rows,
    filter_rows,
    format_arguments,
    format_result,
    group_turns,
    key_value_rows,
    matching_turns,
    next_filter,
    record_table,
    render_trajectory,
    result_detail,
    result_error,
    result_sparkline,
    run_facts,
    sparkline,
    summarize_numbers,
    summary_line,
    thinking_title,
    turn_header,
)

TRANSIT = {"period": 3.0, "t0": 1.0, "duration": 0.1, "depth": 1e-3}
TRUTH = {
    "ep_0000": {"kind": "transit", "snr": 12.0, "n_transits": 4, "transit": TRANSIT},
    "ep_0001": {"kind": "null"},
}


def _call(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


PRED = {
    "detected": True,
    "period": 3.0,
    "t0": 1.0,
    "depth": 9e-4,
    "duration": 0.125,
    "stop": "submitted",
    "n_tool_calls": 2,
    "n_nudges": 1,
    "usage": {"input_tokens": 10, "output_tokens": 5},
    "reasoning": "clean dip",
    "thinking": ["let me look", ""],
    "conversation": [
        {"role": "user", "content": "Analyse."},
        {"role": "assistant", "content": "", "tool_calls": [_call("a", "fold", {"period": 3.0})]},
        {
            "role": "tool",
            "tool_call_id": "a",
            "content": json.dumps({"phase": list(range(40)), "depth_ppm": 900}),
        },
        {"role": "user", "content": "Your last reply contained no tool call"},
        {"role": "assistant", "content": "", "tool_calls": [_call("b", "submit_prediction", {})]},
        {"role": "tool", "tool_call_id": "b", "content": json.dumps({"ok": True})},
    ],
}
ROWS = {"ep_0000": {"outcome": "hit", "score": 1.0}, "ep_0001": {"outcome": "false_alarm", "score": 0.0}}


def _results():
    preds = {"ep_0000": PRED, "ep_0001": {"detected": True, "error": "RuntimeError: boom"}}
    return Results("tool_agent", {}, "set", "hash", preds, ROWS, {})


def _row(outcome, error=False):
    return EpisodeRow("ep", "transit", 10.0, outcome, 0.0, 1, "submitted", error)


PEAKS = {
    "peaks": [
        {"period": 14.40244, "t0": 9.6553, "depth_ppm": 14496.0, "sde": 11.37, "n_events_in_data": 3},
        {"period": 21.52326, "t0": 2.4553, "depth_ppm": 11457.0, "sde": 8.95, "n_events_in_data": 3},
    ],
    "calls_remaining": 38,
}


def _result(name, result, arguments=None):
    return ToolCall(name, json.dumps(arguments or {}), json.dumps(result))


def _plain(renderable, width=100):
    console = Console(file=io.StringIO(), width=width, record=True, color_system=None)
    console.print(renderable)
    return console.export_text()


def test_episode_rows_in_order_with_truth_and_score():
    """Each row joins the truth kind and SNR with the score row and the agent's stop reason."""
    rows = episode_rows(_results(), TRUTH)
    assert rows[0] == EpisodeRow("ep_0000", "transit", 12.0, "hit", 1.0, 2, "submitted", False)
    assert rows[1].kind == "null" and rows[1].snr == 0.0 and rows[1].stop.startswith("RuntimeError")


def test_episode_rows_flag_error_and_unsubmitted_run_but_not_missing_stop():
    """An error or a stop other than ``submitted`` is an error; a baseline has no stop and is not one."""
    preds = {"ep_0000": {"stop": "max_turns"}, "ep_0001": {"detected": False}}
    results = Results("bls", {}, "set", "hash", preds, ROWS, {})
    assert [row.error for row in episode_rows(results, TRUTH)] == [True, False]
    assert episode_rows(_results(), TRUTH)[1].error


def test_filter_rows_keeps_only_the_named_outcome():
    """The ``miss`` filter drops hits, so the list shows only the misses."""
    rows = [_row("hit"), _row("miss"), _row("miss")]
    assert [row.outcome for row in filter_rows(rows, "miss")] == ["miss", "miss"]


def test_filter_rows_errors_uses_the_error_flag_not_the_outcome():
    """A run that failed can have any outcome, so the ``errors`` filter reads the flag."""
    rows = [_row("hit"), _row("true_null", error=True)]
    assert filter_rows(rows, "errors") == [rows[1]]


def test_next_filter_returns_to_all_after_the_last():
    """Pressing ``f`` on the last filter goes back to the full list."""
    assert next_filter("all") == "hit"
    assert next_filter("errors") == "all"


def test_group_turns_gives_each_turn_the_thinking_at_its_position():
    """Thinking entry n belongs to assistant message n; an empty entry gives no thinking."""
    turns = group_turns(PRED)
    assert [turn.thinking for turn in turns] == ["let me look", ""]


def test_group_turns_matches_results_to_calls_by_id():
    """Results that come back in another order still go to the call with the same id."""
    prediction = {
        "conversation": [
            {"role": "assistant", "tool_calls": [_call("a", "fold", {}), _call("b", "odd_even", {})]},
            {"role": "tool", "tool_call_id": "b", "content": "odd"},
            {"role": "tool", "tool_call_id": "a", "content": "folded"},
        ]
    }
    calls = group_turns(prediction)[0].calls
    assert [(call.name, call.result) for call in calls] == [("fold", "folded"), ("odd_even", "odd")]


def test_group_turns_puts_a_nudge_on_the_turn_before_it():
    """A user message after a turn is the loop's nudge to that turn; the task prompt is not a nudge."""
    turns = group_turns(PRED)
    assert turns[0].nudges == ("Your last reply contained no tool call",)
    assert turns[1].nudges == ()


def test_format_arguments_writes_pairs_and_shortens_long_values():
    """A long reasoning argument must not push the other arguments out of the header line."""
    text = format_arguments(json.dumps({"period": 3.0, "reasoning": "x" * 200}), width=20)
    assert text.startswith("period=3.0, reasoning=")
    assert len(text.split("reasoning=")[1]) == 20 and text.endswith("…")


def test_turn_header_names_every_call_and_flags_a_nudge():
    """The header alone tells the reader what the turn did and that the loop nudged it."""
    turn = Turn(3, calls=(ToolCall("fold", '{"period": 2}'), ToolCall("odd_even", "{}")), nudges=("go",))
    assert turn_header(turn) == "Turn 3  fold(period=2)  ·  odd_even()  [nudged]"


def test_turn_header_says_when_a_turn_has_no_tool_call():
    """A text-only turn is a common failure, so the header states it."""
    assert turn_header(Turn(4, text="hmm")) == "Turn 4  no tool call"


def test_summarize_numbers_gives_count_min_and_max():
    """The summary keeps the range of the values, which is what a reader checks first."""
    assert summarize_numbers([3.0, -1500.0, 7.0]) == "[3 values, min \u22121,500 … max 7]"


def test_format_result_summarises_a_long_numeric_list():
    """A 40-value phase list becomes one line with its count and range."""
    content = json.dumps({"phase": list(range(40)), "depth_ppm": 900})
    assert format_result(content) == '{"phase": [40 values, min 0 … max 39], "depth_ppm": 900}'


def test_format_result_keeps_a_short_numeric_list():
    """A list at the length limit stays as values, because a summary would hide as much as it saves."""
    assert format_result(json.dumps({"gap": [1, 2, 3, 4, 5, 6, 7, 8]})) == '{"gap": [1, 2, 3, 4, 5, 6, 7, 8]}'


def test_format_result_puts_one_key_per_line_if_too_wide():
    """An object wider than the line width gets one key per line, indented."""
    text = format_result(json.dumps({"first": "a" * 30, "second": "b" * 30}), width=40)
    assert text.splitlines() == ["{", f'  "first": "{"a" * 30}",', f'  "second": "{"b" * 30}"', "}"]


def test_format_result_returns_text_that_is_not_json_unchanged():
    """A tool error message is plain text and must be shown as written."""
    assert format_result("error: bad period") == "error: bad period"


def test_thinking_title_is_none_for_a_short_thought():
    """A one-line thought shows inline; a character count on it would only add noise."""
    assert thinking_title("I'll start by summarizing the light curve.") is None
    assert thinking_title("x" * 2714) == "thinking · 2,714 chars"


def test_summary_line_of_summarize_gives_points_span_scatter_flux_and_gaps():
    """The light-curve summary fits on one line with the numbers a reader checks first."""
    result = {
        "n_points": 26767,
        "span_days": 52.623,
        "cadence_minutes": 2.0,
        "point_scatter_ppm": 4320.4,
        "flux_min": 0.936527,
        "flux_max": 1.13822,
        "gaps_days": [[9.5, 4.9], [24.2, 4.8], [38.3, 4.6]],
    }
    expected = "n=26,767 pts · 52.6 d · 2.0 min · scatter 4,320 ppm · flux 0.937\u20131.138 · 3 gaps"
    assert summary_line(_result("summarize", result)) == expected


def test_summary_line_of_bls_peaks_gives_count_and_top_peak():
    """Only the top peak is on the line; the expanded table holds the others."""
    expected = "2 peaks · top P=14.402 d sde 11.4 depth 14,496 ppm (3 events)"
    assert summary_line(_result("bls_peaks", PEAKS)) == expected


def test_summary_line_of_search_signals_gives_count_top_signal_and_residual():
    """The line shows how many signals the search accepted, the first one, and the SDE it stopped at."""
    result = {
        "signals": [{"rank": 1, "period": 3.0, "sde": 25.3, "depth_ppm": 9800.0}],
        "residual_sde": 6.2,
        "stop": "below_min_sde",
    }
    assert summary_line(_result("search_signals", result)) == (
        "1 signals · top P=3.000 d sde 25.3 depth 9,800 ppm · residual sde 6.2"
    )
    assert summary_line(_result("search_signals", {"signals": [], "residual_sde": 4.0})) == (
        "0 signals · residual sde 4.0"
    )


def test_summary_line_of_fit_transit_gives_shape_and_verdicts():
    """The fit line carries what a reader checks first: depth, duration, geometry and the two verdicts."""
    result = {
        "depth_ppm": 9850.0,
        "duration_hours": 2.61,
        "impact_parameter": 0.31,
        "shape": "flat_bottom",
        "density_verdict": "plausible",
        "significance_sigma": 41.2,
    }
    assert summary_line(_result("fit_transit", result)) == (
        "depth 9,850 ppm · 2.61 h · b 0.31 · flat_bottom · density plausible · 41.2\u03c3"
    )


def test_summary_line_of_single_events_gives_count_and_first_event():
    """The single-event search reports how many events it found and the strongest one."""
    result = {"events": [{"time": 20.0, "snr": 23.12, "depth_ppm": 9800.0}, {"time": 5.0}]}
    expected = "2 events · top t=20.00 d snr 23.1 depth 9,800 ppm"
    assert summary_line(_result("single_events", result)) == expected


def test_summary_line_of_fold_counts_empty_bins_from_the_nulls():
    """An older fold result has no ``n_empty_bins``, so the line counts the null bins itself."""
    result = {"depth_ppm": 15681.0, "flux_ppm": [10.0, None, -15681.0, None]}
    assert summary_line(_result("fold", result)) == "depth 15,681 ppm · 4 bins · 2 empty"


def test_summary_line_of_odd_even_gives_verdict_depths_events_and_sigma():
    """The odd/even line holds everything the agent uses to accept or reject a candidate."""
    result = {
        "verdict": "consistent",
        "odd_depth_ppm": 7953.3,
        "odd_events": 25,
        "even_depth_ppm": 6433.5,
        "even_events": 25,
        "difference_sigma": 1.62,
    }
    expected = "verdict consistent · odd 7,953 ppm (25 ev) · even 6,434 ppm (25 ev) · 1.6\u03c3"
    assert summary_line(_result("odd_even", result)) == expected


def test_summary_line_of_submit_prediction_reads_the_arguments():
    """The submission result is only ``ok``; the line shows what the agent submitted."""
    arguments = {"detected": True, "period": 0.850936, "t0": 0.3613}
    call = _result("submit_prediction", {"ok": True}, arguments)
    assert summary_line(call) == "detected=True P=0.8509 d t0=0.361 d"


def test_summary_line_of_a_single_event_submission_names_the_form():
    """A submission without a period is the single-event form; the line says so and shows the event time."""
    arguments = {"detected": True, "t0": 20.5, "duration": 0.12, "depth": 0.01}
    call = _result("submit_prediction", {"ok": True, "form": "single_event"}, arguments)
    assert summary_line(call) == "detected=True single event t0=20.500 d"


def test_summary_line_of_a_failed_call_is_the_error_text():
    """A rejected call shows why it failed, and :func:`result_error` marks it for the red style."""
    call = _result("submit_prediction", {"error": "argument detected must be boolean, got str"})
    assert summary_line(call) == "argument detected must be boolean, got str"
    assert result_error(call) is not None
    assert result_error(_result("fold", {"depth_ppm": 1.0})) is None


def test_summary_line_falls_back_to_one_line_of_json():
    """A tool without its own summary, or a result in an unexpected layout, still gives one line."""
    assert summary_line(_result("new_tool", {"a": list(range(20))})) == '{"a": [20 values, min 0 … max 19]}'
    assert summary_line(_result("bls_peaks", {"peaks": 5})) == '{"peaks": 5}'


def test_sparkline_maps_the_range_to_eight_levels_and_nulls_to_spaces():
    """A transit is a notch of low blocks, and an empty bin is a gap, not a fake value."""
    assert sparkline([300.0, None, -700.0, 700.0]) == "\u2586 \u2581\u2588"


def test_sparkline_of_constant_values_is_flat():
    """With no range there is nothing to scale, so every value gets the lowest block."""
    assert sparkline([5.0, 5.0]) == "\u2581\u2581"


def test_result_sparkline_is_only_for_fold():
    """Only a fold has a light curve to draw; other tools have no sparkline."""
    assert result_sparkline(_result("fold", {"flux_ppm": [0.0, 1.0]})) == "\u2581\u2588"
    assert result_sparkline(_result("odd_even", {"flux_ppm": [0.0, 1.0]})) == ""


def test_record_table_has_one_row_per_record_and_only_present_columns():
    """Columns that no peak has, for example ``half_period``, are left out."""
    table = record_table(PEAKS["peaks"], PEAK_COLUMNS)
    assert table.row_count == 2
    assert [column.header for column in table.columns] == ["period", "t0", "depth", "sde", "events"]


def test_key_value_rows_format_numbers_and_skip_keys():
    """Numbers get thousands separators and a true minus sign; skipped keys are left out."""
    rows = key_value_rows({"depth_ppm": -6433.5, "odd_n": 872, "peaks": []}, skip=("peaks",))
    assert rows == [("depth_ppm", "\u22126,433.5"), ("odd_n", "872")]


def test_result_detail_of_bls_peaks_is_a_table_not_json():
    """The expanded peaks are a table with one row per peak and the other fields below it."""
    text = _plain(result_detail(_result("bls_peaks", PEAKS)))
    assert "14.40244" in text and "21.52326" in text and "calls_remaining" in text
    assert '"peaks"' not in text


def test_result_detail_of_an_unknown_tool_is_pretty_json():
    """A tool without its own view falls back to JSON with long lists summarised."""
    text = _plain(result_detail(_result("new_tool", {"values": list(range(40))})))
    assert text.strip() == '{"values": [40 values, min 0 … max 39]}'


def test_left_bar_puts_a_bar_on_every_wrapped_line():
    """A long thought wraps inside the bar, so each of its lines is marked as thinking."""
    lines = _plain(LeftBar(Text("word " * 30)), width=40).splitlines()
    assert len(lines) > 1 and all(line.startswith("\u2502 ") for line in lines)


def test_matching_turns_ignores_case_and_searches_tool_results():
    """A search for a value finds the turn whose tool result holds it, in any case."""
    turns = [Turn(1, thinking="Fold first"), Turn(2, calls=(ToolCall("bls", "{}", '{"verdict": "Odd"}'),))]
    assert matching_turns(turns, "odd") == [1]
    assert matching_turns(turns, "FOLD") == [0]


def test_matching_turns_with_empty_query_finds_nothing():
    """An empty search clears the matches; it does not mark every turn."""
    assert matching_turns([Turn(1, thinking="a")], "") == []


def test_adjacent_index_moves_to_the_next_or_previous_target():
    """With search matches 1 and 4, ``n`` from turn 1 goes to 4 and ``p`` from 4 goes back to 1."""
    assert adjacent_index([1, 4], 1, 1) == 4
    assert adjacent_index([1, 4], 4, -1) == 1
    assert adjacent_index([1, 4], -1, 1) == 1


def test_adjacent_index_returns_none_past_the_last_target():
    """At the last target there is nowhere to go, so the viewer rings the bell instead."""
    assert adjacent_index([1, 4], 4, 1) is None
    assert adjacent_index([1, 4], 1, -1) is None


def test_comparison_rows_convert_depth_to_ppm_and_duration_to_hours():
    """The details pane shows depth in ppm and duration in hours for the truth and the prediction."""
    rows = {
        quantity: (true, predicted) for quantity, true, predicted in comparison_rows(PRED, TRUTH["ep_0000"])
    }
    assert rows["depth (ppm)"] == ("1000", "900")
    assert rows["duration (h)"] == ("2.40", "3.00")


def test_comparison_rows_show_a_dash_for_a_null_truth_and_no_detection():
    """A null episode has no true period, and a non-detection has no predicted period."""
    rows = comparison_rows({"detected": False, "period": 5.0}, TRUTH["ep_0001"])
    assert rows[0] == ("period (d)", "—", "—")


def test_run_facts_include_cost_only_if_the_usage_has_it():
    """A usage without a price gives no cost line, so the pane does not invent a cost."""
    labels = [label for label, _ in run_facts(PRED, ROWS["ep_0000"])]
    priced = {**PRED, "usage": {"input_tokens": 1, "usd": 0.25}}
    assert "tokens in" in labels and "cost" not in labels
    assert ("cost", "$0.2500") in run_facts(priced, ROWS["ep_0000"])


def test_episode_label_colours_the_outcome_with_its_theme_tone():
    """A miss is in the error colour of the Textual theme, so it stands out on dark and light themes."""
    assert "[$error]miss" in episode_label(_row("miss"))
    assert "[$success]hit" in episode_label(_row("hit"))


def test_render_trajectory_includes_each_part_in_order():
    """Header, thinking, result, nudge, next call and reasoning appear in the order they happened."""
    text = _plain(render_trajectory(PRED, TRUTH["ep_0000"], ROWS["ep_0000"]))
    markers = (
        "Turn 1  fold(period=3.0)",
        "let me look",
        "fold  depth 900 ppm",
        "Your last reply contained no tool call",
        "submit_prediction",
        "clean dip",
    )
    positions = [text.index(marker) for marker in markers]
    assert positions == sorted(positions)


def test_render_trajectory_shows_result_detail_only_if_full():
    """By default a result is one summary line; with ``full_results`` the table follows it."""
    pred = {
        "conversation": [
            {"role": "assistant", "tool_calls": [_call("a", "bls_peaks", {})]},
            {"role": "tool", "tool_call_id": "a", "content": json.dumps(PEAKS)},
        ]
    }
    assert "21.52326" not in _plain(render_trajectory(pred, TRUTH["ep_0001"], ROWS["ep_0001"]))
    assert "21.52326" in _plain(render_trajectory(pred, TRUTH["ep_0001"], ROWS["ep_0001"], full_results=True))


def test_render_trajectory_marks_a_nudge_with_a_warning_sign():
    """A nudge is the loop correcting the model, so it carries a warning sign."""
    text = _plain(render_trajectory(PRED, TRUTH["ep_0000"], ROWS["ep_0000"]))
    assert "\u26a0 Your last reply contained no tool call" in text


def test_render_trajectory_can_hide_thinking():
    """With ``show_thinking=False`` the thinking is omitted and nothing else changes."""
    text = _plain(render_trajectory(PRED, TRUTH["ep_0000"], ROWS["ep_0000"], show_thinking=False))
    assert "let me look" not in text and "clean dip" in text


def test_render_trajectory_shows_error():
    """A recorded detector error is printed instead of a conversation."""
    pred = {"detected": False, "error": "RuntimeError: boom"}
    assert "RuntimeError: boom" in _plain(render_trajectory(pred, TRUTH["ep_0001"], ROWS["ep_0001"]))


def test_viewer_key_j_shows_the_next_episode():
    """Pressing ``j`` selects the next episode and the trajectory pane renders its turns."""
    pytest.importorskip("textual")
    from stargazer.viewer import make_viewer

    async def press_j():
        app = make_viewer(_results(), TRUTH)
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause()
            before = (app.current, len(app.turns))
            await pilot.press("j")
            await pilot.pause()
            return before, (app.current, len(app.turns))

    assert asyncio.run(press_j()) == (("ep_0000", 2), ("ep_0001", 0))


def test_viewer_enter_expands_the_results_of_the_current_turn():
    """Results start collapsed; ``enter`` opens the results of the current turn and no other."""
    pytest.importorskip("textual")
    from stargazer.viewer import make_viewer

    async def press_enter():
        app = make_viewer(_results(), TRUTH)
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            return [section.collapsed for section in app.query("Collapsible.result")]

    assert asyncio.run(press_enter()) == [False, True]
