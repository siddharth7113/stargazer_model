"""Unit tests for stargazer.agent: the tool-calling loop, with the tools replaced by a fake dispatcher."""

import hashlib
import json

import pytest

from stargazer import agent as agent_module
from stargazer.agent import SYSTEM_PROMPT, AssistantTurn, ToolAgentDetector, ToolCall


class FixedBackend:
    """Returns the given turns in order, then an empty text turn. Records every request it receives."""

    name = "fixed"
    config = {"model": "fixed"}

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []

    def complete(self, messages, tools):
        self.seen.append([dict(message) for message in messages])
        return self.turns.pop(0) if self.turns else AssistantTurn(content="done")


@pytest.fixture
def fake_tools(monkeypatch):
    """Replace the real tool dispatcher with a recorder so only the loop is under test."""
    calls = []

    def fake_call_tool(name, arguments, time, flux):
        calls.append((name, arguments))
        return {"error": "bad"} if name == "fail" else {"ok": name}

    monkeypatch.setattr(agent_module, "call_tool", fake_call_tool)
    return calls


def _submit(**arguments):
    return AssistantTurn(tool_calls=[ToolCall("1", "submit_prediction", arguments)])


def test_submission_becomes_prediction(lc, fake_tools):
    """The arguments of ``submit_prediction`` become the Prediction; ``reasoning`` goes to ``extra``."""
    time, flux, flux_err = lc
    turns = [_submit(detected=True, period=3.0, t0=1.0, reasoning="r")]
    pred = ToolAgentDetector(FixedBackend(turns)).detect(time, flux, flux_err)
    assert pred.detected and pred.period == 3.0 and pred.t0 == 1.0
    assert pred.extra["stop"] == "submitted" and pred.extra["reasoning"] == "r"


def test_loop_ends_after_submission(lc, fake_tools):
    """Once the model submits, the loop must not ask the backend again."""
    time, flux, flux_err = lc
    backend = FixedBackend([_submit(detected=False), AssistantTurn(content="never asked")])
    ToolAgentDetector(backend).detect(time, flux, flux_err)
    assert len(backend.seen) == 1


def test_tool_results_are_appended_in_openai_format(lc, fake_tools):
    """The second request carries the assistant turn with tool_calls and a tool message with the same id."""
    time, flux, flux_err = lc
    backend = FixedBackend(
        [AssistantTurn(tool_calls=[ToolCall("abc", "summarize", {})]), AssistantTurn(content="x")]
    )
    ToolAgentDetector(backend).detect(time, flux, flux_err)
    second_request = backend.seen[1]
    assert second_request[0]["role"] == "system" and second_request[1]["role"] == "user"
    assert second_request[2]["role"] == "assistant" and second_request[2]["tool_calls"][0]["id"] == "abc"
    assert second_request[3] == {
        "role": "tool",
        "tool_call_id": "abc",
        "content": json.dumps({"ok": "summarize", "calls_remaining": 39}),
    }


def test_tool_results_carry_calls_remaining(lc, fake_tools):
    """Each tool result tells the model how many calls are left, counting down from the cap."""
    time, flux, flux_err = lc
    turns = [AssistantTurn(tool_calls=[ToolCall("1", "summarize", {}), ToolCall("2", "summarize", {})])]
    backend = FixedBackend(turns)
    pred = ToolAgentDetector(backend, max_tool_calls=5).detect(time, flux, flux_err)
    remaining = [call["result"]["calls_remaining"] for call in pred.extra["transcript"]]
    assert remaining == [4, 3]
    assert json.loads(backend.seen[1][-1]["content"])["calls_remaining"] == 3


def test_tools_are_called_with_the_episode_arrays(lc, fake_tools):
    """The loop forwards the model's arguments to the dispatcher unchanged."""
    time, flux, flux_err = lc
    backend = FixedBackend([AssistantTurn(tool_calls=[ToolCall("1", "fold", {"period": 2.5, "t0": 0.1})])])
    ToolAgentDetector(backend).detect(time, flux, flux_err)
    assert fake_tools == [("fold", {"period": 2.5, "t0": 0.1})]


def test_model_stopping_without_submission_is_none(lc, fake_tools):
    """A model that answers in prose on every turn is nudged twice, then scores none as model_stopped."""
    time, flux, flux_err = lc
    backend = FixedBackend([AssistantTurn(content="I give up")] * 5)
    pred = ToolAgentDetector(backend).detect(time, flux, flux_err)
    assert not pred.detected and pred.extra["stop"] == "model_stopped" and pred.extra["n_nudges"] == 2
    assert len(backend.seen) == 3


def test_nudge_after_prose_turn_lets_the_model_recover(lc, fake_tools):
    """A prose turn gets a user reminder; a tool call on the next turn continues the episode normally."""
    time, flux, flux_err = lc
    backend = FixedBackend([AssistantTurn(content="Let me think..."), _submit(detected=False)])
    pred = ToolAgentDetector(backend).detect(time, flux, flux_err)
    assert pred.extra["stop"] == "submitted" and pred.extra["n_nudges"] == 1
    assert backend.seen[1][-1] == {"role": "user", "content": ToolAgentDetector.NUDGE}


def test_truncated_turn_gets_the_truncation_nudge(lc, fake_tools):
    """When the backend reports the output was cut at its cap, the reminder says to be brief."""
    time, flux, flux_err = lc
    backend = FixedBackend([AssistantTurn(content="very long", truncated=True), _submit(detected=False)])
    ToolAgentDetector(backend).detect(time, flux, flux_err)
    assert backend.seen[1][-1]["content"] == ToolAgentDetector.NUDGE_TRUNCATED


def test_nudges_are_configurable(lc, fake_tools):
    """``max_nudges=0`` restores the strict behaviour: one prose turn ends the episode."""
    time, flux, flux_err = lc
    backend = FixedBackend([AssistantTurn(content="done")])
    pred = ToolAgentDetector(backend, max_nudges=0).detect(time, flux, flux_err)
    assert pred.extra["stop"] == "model_stopped" and len(backend.seen) == 1


def test_max_tool_calls_cap(lc, fake_tools):
    """A model that loops on a tool is cut off at the cap and the episode is marked ``max_tool_calls``."""
    time, flux, flux_err = lc
    looping = FixedBackend([AssistantTurn(tool_calls=[ToolCall("x", "summarize", {})])] * 10)
    pred = ToolAgentDetector(looping, max_tool_calls=3).detect(time, flux, flux_err)
    assert pred.extra["stop"] == "max_tool_calls" and pred.extra["n_tool_calls"] == 3 and len(fake_tools) == 3


def test_max_turns_cap(lc, fake_tools):
    """The turn cap is independent of the tool-call cap."""
    time, flux, flux_err = lc
    looping = FixedBackend([AssistantTurn(tool_calls=[ToolCall("x", "summarize", {})])] * 10)
    pred = ToolAgentDetector(looping, max_turns=2, max_tool_calls=100).detect(time, flux, flux_err)
    assert pred.extra["stop"] == "max_turns" and pred.extra["n_turns"] == 2


def test_token_cap(lc, fake_tools):
    """Input plus output tokens from the backend's usage stop the loop once they reach the cap."""
    time, flux, flux_err = lc
    turn = AssistantTurn(
        tool_calls=[ToolCall("x", "summarize", {})], usage={"input_tokens": 600, "output_tokens": 0}
    )
    pred = ToolAgentDetector(FixedBackend([turn] * 5), max_tokens_per_episode=1000).detect(
        time, flux, flux_err
    )
    assert pred.extra["stop"] == "max_tokens" and pred.extra["usage"]["input_tokens"] == 1200


def test_invalid_submission_does_not_end_episode(lc, fake_tools):
    """A tool error on ``fail`` is recorded in the transcript and the loop continues to the next turn."""
    time, flux, flux_err = lc
    turns = [AssistantTurn(tool_calls=[ToolCall("1", "fail", {})]), AssistantTurn(content="stop")]
    pred = ToolAgentDetector(FixedBackend(turns)).detect(time, flux, flux_err)
    assert pred.extra["stop"] == "model_stopped"
    assert pred.extra["transcript"] == [
        {"tool": "fail", "arguments": {}, "result": {"error": "bad", "calls_remaining": 39}}
    ]


def test_usage_is_summed_across_turns(lc, fake_tools):
    """Cost and tokens are accumulated over every backend call in the episode."""
    time, flux, flux_err = lc
    turns = [
        AssistantTurn(tool_calls=[ToolCall("1", "summarize", {})], usage={"usd": 0.01, "input_tokens": 5}),
        AssistantTurn(content="x", usage={"usd": 0.02, "input_tokens": 7}),
    ]
    pred = ToolAgentDetector(FixedBackend(turns)).detect(time, flux, flux_err)
    assert pred.extra["usage"]["usd"] == pytest.approx(0.03) and pred.extra["usage"]["input_tokens"] == 12


def test_usage_has_no_cost_when_backend_reports_none(lc, fake_tools):
    """A backend without a price gives no usd key; a zero would be a false statement about cost."""
    time, flux, flux_err = lc
    pred = ToolAgentDetector(FixedBackend([_submit(detected=False)])).detect(time, flux, flux_err)
    assert "usd" not in pred.extra["usage"]


def test_config_records_backend_caps_and_prompt_hash():
    """The config must identify the backend, the caps and the exact prompt, so a run is reproducible."""
    config = ToolAgentDetector(FixedBackend([]), max_tool_calls=5).config
    assert config["backend"] == "fixed" and config["backend_model"] == "fixed"
    assert config["max_tool_calls"] == 5
    assert config["system_prompt_sha256"] == hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
    assert "submit_prediction" in config["tools"]


def test_trajectory_keeps_conversation_and_thinking(lc, fake_tools):
    """The prediction stores the message list the backend saw and the thinking of each turn."""
    time, flux, flux_err = lc
    turns = [
        AssistantTurn(content="I will search.", tool_calls=[ToolCall("1", "summarize", {})], thinking="hmm"),
        _submit(detected=False),
    ]
    pred = ToolAgentDetector(FixedBackend(turns)).detect(time, flux, flux_err)
    roles = [message["role"] for message in pred.extra["conversation"]]
    assert roles == ["user", "assistant", "tool", "assistant", "tool"]
    assert pred.extra["conversation"][1]["content"] == "I will search."
    assert pred.extra["thinking"] == ["hmm", ""]


def test_none_caps_mean_defaults():
    """The CLI passes None for an unset cap; None must select the default, never reach the countdown."""
    config = ToolAgentDetector(FixedBackend([]), max_tool_calls=None, max_turns=None).config
    assert config["max_tool_calls"] == ToolAgentDetector.DEFAULT_MAX_TOOL_CALLS
    assert config["max_turns"] == ToolAgentDetector.DEFAULT_MAX_TURNS
