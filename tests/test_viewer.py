"""Unit tests for stargazer.viewer: the table rows and the markup rendering, without starting the UI."""

import json

from stargazer.evaluate import Results
from stargazer.viewer import EpisodeRow, _format_result, episode_rows, render_trajectory

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
    "stop": "submitted",
    "n_tool_calls": 2,
    "n_nudges": 1,
    "usage": {"input_tokens": 10},
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


def test_episode_rows_in_order_with_truth_and_score():
    """Each row joins the truth kind and SNR with the score row and the agent's stop reason."""
    rows = episode_rows(_results(), TRUTH)
    assert rows[0] == EpisodeRow("ep_0000", "transit", 12.0, "hit", 1.0, 2, "submitted")
    assert rows[1].kind == "null" and rows[1].snr == 0.0 and rows[1].stop.startswith("RuntimeError")


def test_format_result_shortens_long_lists_unless_full():
    """A 40-value phase list is shown as a count in the short form and in full in the full form."""
    content = json.dumps({"phase": list(range(40)), "depth_ppm": 900})
    assert "[40 values]" in _format_result(content, full=False)
    assert "39" in _format_result(content, full=True)


def test_render_trajectory_includes_each_part_in_order():
    """Thinking, tool call, result, nudge and reasoning appear in the order they happened."""
    text = render_trajectory(PRED, TRUTH["ep_0000"], ROWS["ep_0000"])
    for marker in ("let me look", "fold", "[result]", "[nudge]", "submit_prediction", "clean dip"):
        assert marker in text
    assert text.index("let me look") < text.index("fold") < text.index("[nudge]")
    assert text.index("[nudge]") < text.index("submit_prediction")


def test_render_trajectory_can_hide_thinking():
    """With ``show_thinking=False`` the thinking lines are omitted and nothing else changes."""
    text = render_trajectory(PRED, TRUTH["ep_0000"], ROWS["ep_0000"], show_thinking=False)
    assert "let me look" not in text and "clean dip" in text


def test_render_trajectory_shows_error():
    """A recorded detector error is printed instead of a conversation."""
    pred = {"detected": False, "error": "RuntimeError: boom"}
    assert "RuntimeError: boom" in render_trajectory(pred, TRUTH["ep_0001"], ROWS["ep_0001"])
