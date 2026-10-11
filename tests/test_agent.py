"""Offline tests for the tool registry, the agent loop and the backend message rendering."""

import json

from stargazer.agent import AssistantTurn, ToolAgentDetector, ToolCall
from stargazer.backends import TinkerBackend
from stargazer.inject import inject, make_truth
from stargazer.score import score_episode
from stargazer.tools import TOOLS, TOOLS_BY_NAME, call_tool, tool_definitions


class ScriptedBackend:
    """Replays a fixed list of turns. The last turn reads the bls result from the transcript."""

    name = "scripted"
    config = {"script": "bls_then_submit"}

    def __init__(self, submit_from_bls: bool = True, stop_without_submit: bool = False):
        self.submit_from_bls = submit_from_bls
        self.stop_without_submit = stop_without_submit
        self.seen: list[list[dict]] = []

    def complete(self, messages, tools):
        self.seen.append([dict(m) for m in messages])
        n_assistant = sum(m["role"] == "assistant" for m in messages)
        usage = {"input_tokens": 100, "output_tokens": 10, "usd": 0.001}
        if n_assistant == 0:
            return AssistantTurn(tool_calls=[ToolCall("c1", "bls_peaks", {})], usage=usage)
        if self.stop_without_submit:
            return AssistantTurn(content="I give up.", usage=usage)
        last_tool = next(m for m in reversed(messages) if m["role"] == "tool")
        peak = json.loads(last_tool["content"])["peaks"][0]
        args = {
            "detected": True,
            "period": peak["period"],
            "t0": peak["t0"],
            "duration": peak["duration"],
            "depth": peak["depth_ppm"] / 1e6,
            "reasoning": "top BLS peak",
        }
        return AssistantTurn(tool_calls=[ToolCall("c2", "submit_prediction", args)], usage=usage)


def test_every_tool_has_valid_schema_and_callable():
    for spec in TOOLS:
        d = spec.definition()
        assert d["type"] == "function" and d["function"]["name"] == spec.name
        assert spec.parameters["type"] == "object" and "properties" in spec.parameters
        for req in spec.parameters.get("required", []):
            assert req in spec.parameters["properties"]
    assert [t["function"]["name"] for t in tool_definitions()] == [t.name for t in TOOLS]


def test_validation_catches_model_mistakes():
    fold = TOOLS_BY_NAME["fold"]
    assert fold.validate({"period": 3.0, "t0": 1.0}) == []
    assert fold.validate({"period": 3, "t0": 1}) == []  # integers are numbers
    assert "missing required argument: t0" in fold.validate({"period": 3.0})
    assert any("unknown argument" in p for p in fold.validate({"period": 3.0, "t0": 1.0, "bins": 10}))
    assert any("must be number" in p for p in fold.validate({"period": "3.0", "t0": 1.0}))
    bls = TOOLS_BY_NAME["bls_peaks"]
    assert bls.validate({"max_period": None}) == []
    sub = TOOLS_BY_NAME["submit_prediction"]
    assert any("must be boolean" in p for p in sub.validate({"detected": "yes"}))


def test_call_tool_returns_errors_instead_of_raising(lc):
    time, flux, _ = lc
    assert "unknown tool" in call_tool("nope", {}, time, flux)["error"]
    assert "missing required" in call_tool("fold", {"period": 3.0}, time, flux)["error"]
    assert "greater than 0" in call_tool("fold", {"period": 0.0, "t0": 0.0}, time, flux)["error"]
    assert "greater than 0" in call_tool("bls_peaks", {"min_period": -1}, time, flux)["error"]
    assert "error" not in call_tool("bls_peaks", {"max_period": None}, time, flux)


def test_agent_finds_injection_through_tools(lc, star):
    time, flux, fe = lc
    t = time - time.min()
    truth = make_truth(period=3.0, t0=1.0, depth=0.01, b=0.2, star=star)
    f = inject(t, flux, truth)
    backend = ScriptedBackend()
    det = ToolAgentDetector(backend)
    pred = det.detect(t, f, fe)
    assert pred.detected and pred.extra["stop"] == "submitted" and pred.extra["n_tool_calls"] == 2
    row = score_episode(pred, {"kind": "transit", "transit": truth.to_dict()})
    assert row["outcome"] == "hit", (pred, row)
    assert pred.extra["usage"]["usd"] > 0 and pred.extra["transcript"][0]["tool"] == "bls_peaks"
    # the conversation the backend saw follows the OpenAI format
    last = backend.seen[-1]
    assert last[0]["role"] == "system" and last[-1]["role"] == "tool" and "tool_calls" in last[-2]
    assert "max_tool_calls" in det.config and det.config["backend"] == "scripted"


def test_agent_without_submission_is_none(lc):
    time, flux, fe = lc
    pred = ToolAgentDetector(ScriptedBackend(stop_without_submit=True)).detect(time - time.min(), flux, fe)
    assert not pred.detected and pred.extra["stop"] == "model_stopped"


def test_agent_caps_tool_calls(lc):
    class Looper:
        name = "looper"
        config = {}

        def complete(self, messages, tools):
            return AssistantTurn(tool_calls=[ToolCall("x", "summarize", {})], usage={})

    time, flux, fe = lc
    pred = ToolAgentDetector(Looper(), max_tool_calls=3).detect(time - time.min(), flux, fe)
    assert pred.extra["stop"] == "max_tool_calls" and pred.extra["n_tool_calls"] == 3


def test_tinker_rendering_round_trip():
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "a",
                    "type": "function",
                    "function": {"name": "fold", "arguments": json.dumps({"period": 3.0, "t0": 1.0})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "a", "content": '{"depth_ppm": 9000}'},
    ]
    out = TinkerBackend.render_messages(messages, tool_definitions())
    assert [m["role"] for m in out] == ["system", "user", "assistant", "user"]
    assert "# Tools" in out[0]["content"] and '"name": "fold"' in out[0]["content"]
    assert "<tool_call>" in out[2]["content"] and "<tool_response>" in out[3]["content"]
    calls = TinkerBackend.parse_tool_calls(
        'thinking...\n<tool_call>\n{"name": "fold", "arguments": {"period": 3.0, "t0": 1.0}}\n</tool_call>\n'
        "<tool_call>\nnot json\n</tool_call>"
    )
    assert len(calls) == 1 and calls[0].name == "fold" and calls[0].arguments == {"period": 3.0, "t0": 1.0}
