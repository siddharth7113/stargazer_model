"""Unit tests for stargazer.backends: config hygiene and the Tinker text rendering, without any network."""

import json

import pytest

from stargazer.backends import OpenAICompatibleBackend, TinkerBackend
from stargazer.tools import tool_definitions


def test_openai_backend_config_has_no_secret(monkeypatch):
    """The results file stores ``config``; it must name the key's variable, never the key."""
    monkeypatch.setenv("MY_KEY", "sk-secret")
    config = OpenAICompatibleBackend("m", api_key_env="MY_KEY").config
    assert "sk-secret" not in json.dumps(config) and config["api_key_env"] == "MY_KEY"


def test_openai_backend_requires_key(monkeypatch):
    """Without the variable set, client creation fails with a message that names the variable."""
    monkeypatch.delenv("MISSING_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MISSING_KEY"):
        OpenAICompatibleBackend("m", api_key_env="MISSING_KEY")._get_client()


def test_tinker_render_puts_tools_in_system_prompt():
    """Tinker has no function-calling API, so the tool schemas are appended to the system prompt as JSON."""
    out = TinkerBackend.render_messages([{"role": "system", "content": "SYS"}], tool_definitions())
    assert out[0]["role"] == "system" and out[0]["content"].startswith("SYS")
    assert "# Tools" in out[0]["content"] and '"name": "fold"' in out[0]["content"]


def test_tinker_render_assistant_tool_calls_as_text():
    """An assistant turn with tool_calls is rendered in the Qwen3 ``<tool_call>`` convention."""
    call = {
        "id": "a",
        "type": "function",
        "function": {"name": "fold", "arguments": json.dumps({"period": 3.0})},
    }
    out = TinkerBackend.render_messages([{"role": "assistant", "content": "", "tool_calls": [call]}], [])
    expected = '<tool_call>\n{"name": "fold", "arguments": {"period": 3.0}}\n</tool_call>'
    assert out[0]["role"] == "assistant" and expected in out[0]["content"]


def test_tinker_render_merges_tool_results_into_one_user_turn():
    """Two tool results after one assistant turn become one user message with two response blocks."""
    messages = [
        {"role": "tool", "tool_call_id": "a", "content": "{}"},
        {"role": "tool", "tool_call_id": "b", "content": "{}"},
    ]
    out = TinkerBackend.render_messages(messages, [])
    assert len(out) == 1 and out[0]["role"] == "user" and out[0]["content"].count("<tool_response>") == 2


def test_tinker_render_passes_plain_user_messages_through():
    """A user message without tool content is unchanged."""
    out = TinkerBackend.render_messages([{"role": "user", "content": "go"}], [])
    assert out == [{"role": "user", "content": "go"}]


def test_tinker_parse_extracts_valid_calls_and_skips_malformed():
    """One well-formed block and one broken block yield exactly one ToolCall."""
    text = (
        '<tool_call>\n{"name": "fold", "arguments": {"period": 3.0}}\n</tool_call>\n'
        "<tool_call>\nnot json\n</tool_call>"
    )
    calls = TinkerBackend.parse_tool_calls(text)
    assert len(calls) == 1 and calls[0].name == "fold" and calls[0].arguments == {"period": 3.0}


def test_tinker_parse_without_blocks_is_empty():
    """Plain prose contains no tool calls."""
    assert TinkerBackend.parse_tool_calls("I think the period is 3 days.") == []


def test_tinker_config_records_checkpoint_and_renderer():
    """A trained checkpoint path and the chat template must be in ``config`` to reproduce a run."""
    config = TinkerBackend(model_path="tinker://run/weights").config
    assert config["model_path"] == "tinker://run/weights" and config["renderer"] == "qwen3_disable_thinking"
