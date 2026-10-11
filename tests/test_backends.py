"""Unit tests for stargazer.backends: config hygiene and the Tinker text rendering, without any network."""

import json

import pytest

from stargazer.backends import OpenAICompatibleBackend, TinkerBackend, recommended_sampling
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


def test_tinker_split_thinking_with_tags():
    """A ``<think>`` block is returned separately and removed from the answer text."""
    thinking, rest = TinkerBackend.split_thinking("<think>\nreason\n</think>\nanswer")
    assert thinking == "reason" and rest == "answer"


def test_tinker_split_thinking_with_implicit_open_tag():
    """Some reasoning templates emit only the closing tag; everything before it is the trace."""
    thinking, rest = TinkerBackend.split_thinking("reason\n</think>\nanswer")
    assert thinking == "reason" and rest == "answer"


def test_tinker_split_thinking_without_tags():
    """Plain text has no trace."""
    assert TinkerBackend.split_thinking("answer") == ("", "answer")


def test_recommended_sampling_by_mode():
    """Non-thinking renderers get the 0.7/0.8/20 model-card values; thinking renderers get 0.6/0.95/20."""
    assert recommended_sampling("qwen3_8_disable_thinking") == {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
    assert recommended_sampling("qwen3_instruct")["temperature"] == 0.7
    assert recommended_sampling("qwen3_8_xhigh_reasoning") == {"temperature": 0.6, "top_p": 0.95, "top_k": 20}


def test_tinker_backend_uses_recommended_defaults_and_records_seed():
    """Unset sampling values come from the renderer's mode, and the seed is in config for reproducibility."""
    config = TinkerBackend(renderer="qwen3_8_disable_thinking", seed=7).config
    assert config["temperature"] == 0.7 and config["top_p"] == 0.8 and config["top_k"] == 20
    assert config["seed"] == 7


def test_tinker_backend_explicit_values_override_defaults():
    """An explicit temperature wins over the model-card default."""
    assert TinkerBackend(renderer="qwen3", temperature=0.0).config["temperature"] == 0.0


def test_openai_backend_default_temperature_and_seed():
    """The OpenAI-compatible backend samples at 0.7 by default and records its seed."""
    config = OpenAICompatibleBackend("m").config
    assert config["temperature"] == 0.7 and config["seed"] == 0
