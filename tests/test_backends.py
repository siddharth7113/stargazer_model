"""Unit tests for stargazer.backends: config hygiene and the Tinker text rendering, without any network."""

import json

import pytest

from stargazer.backends import OpenAICompatibleBackend, TinkerBackend, recommended_sampling


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


def test_recommended_sampling_by_mode():
    """Non-thinking renderers get the 0.7/0.8/20 model-card values; thinking renderers get 0.6/0.95/20."""
    plain = recommended_sampling("qwen3_8_disable_thinking")
    assert (plain["temperature"], plain["top_p"], plain["top_k"]) == (0.7, 0.8, 20)
    assert recommended_sampling("qwen3_instruct")["temperature"] == 0.7
    thinking = recommended_sampling("qwen3_8_xhigh_reasoning")
    assert (thinking["temperature"], thinking["top_p"], thinking["top_k"]) == (0.6, 0.95, 20)


def test_recommended_max_tokens_is_larger_for_thinking():
    """A thinking model writes its reasoning before the tool call, so its output cap must be larger."""
    thinking = recommended_sampling("qwen3_8_xhigh_reasoning")["max_tokens"]
    assert thinking > recommended_sampling("qwen3_instruct")["max_tokens"]
    assert TinkerBackend(renderer="qwen3_8_xhigh_reasoning").config["max_tokens"] == 8192


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


def test_tinker_to_cookbook_messages_converts_calls_and_results():
    """Assistant tool calls become cookbook ToolCall objects; a tool result gets the call's id and name."""
    pytest.importorskip("tinker_cookbook")
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
                    "function": {"name": "fold", "arguments": json.dumps({"period": 3.0})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "a", "content": "{}"},
    ]
    out = TinkerBackend.to_cookbook_messages(messages)
    assert [message["role"] for message in out] == ["user", "assistant", "tool"]
    call = out[1]["tool_calls"][0]
    assert (
        call.function.name == "fold"
        and json.loads(call.function.arguments) == {"period": 3.0}
        and call.id == "a"
    )
    assert out[2]["tool_call_id"] == "a" and out[2]["name"] == "fold"


def test_tinker_split_content_separates_thinking_and_text():
    """A parts list yields the thinking part and the text part; a plain string is all text."""
    thinking, text = TinkerBackend.split_content([{"thinking": "hmm"}, {"text": "answer"}])
    assert thinking == "hmm" and text == "answer"
    assert TinkerBackend.split_content("plain") == ("", "plain")


def test_tinker_parse_tool_calls_fallback_skips_malformed():
    """The plain-text fallback parser keeps one well-formed block and drops a broken one."""
    text = (
        '<tool_call>\n{"name": "fold", "arguments": {"period": 3.0}}\n</tool_call>\n'
        "<tool_call>\nnot json\n</tool_call>"
    )
    calls = TinkerBackend.parse_tool_calls(text)
    assert len(calls) == 1 and calls[0].name == "fold" and calls[0].arguments == {"period": 3.0}


def test_tinker_price_table_has_the_comparison_models():
    """The models we evaluate must have a price, or their runs report no cost."""
    from stargazer.backends import TINKER_SAMPLING_USD_PER_MILLION

    for model in (
        "Qwen/Qwen3.8-27B",
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        "thinkingmachines/Inkling-Small",
    ):
        assert TINKER_SAMPLING_USD_PER_MILLION[model] > 0
