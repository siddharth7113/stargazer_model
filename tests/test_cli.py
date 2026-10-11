"""Unit tests for stargazer.cli: argument defaults and the wiring into the SDK, without network or models."""

import pytest

from stargazer.cli import build_parser


def test_agent_command_defaults():
    """Unset caps are None (so the detector default applies); limit is 3; backend is openai."""
    args = build_parser().parse_args(["agent", "--episodes", "x", "--model", "m"])
    assert args.max_tool_calls is None and args.limit == 3 and args.backend == "openai"
    assert args.renderer == "qwen3_disable_thinking"
    assert args.temperature is None and args.seed == 0


def test_baseline_command_defaults():
    """The baseline scores at the default period tolerance of 1 percent on all episodes."""
    args = build_parser().parse_args(["baseline", "--episodes", "x"])
    assert args.period_rtol == 0.01 and args.limit == 0


def test_show_command_defaults():
    """The trajectory printer uses a 120-column console, shortened results and visible thinking."""
    args = build_parser().parse_args(["show", "--results", "r", "--episode", "ep_0001"])
    assert args.width == 120 and args.full is False and args.no_thinking is False


def test_view_command_takes_results():
    """The terminal viewer needs only the results file."""
    assert build_parser().parse_args(["view", "--results", "r"]).results == "r"


def test_agent_command_forwards_renderer_temperature_and_seed(monkeypatch):
    """Every sampling flag must reach the backend; a parsed-but-unused flag silently changes an experiment."""
    import stargazer.backends as backends_module
    import stargazer.cli as cli_module
    import stargazer.episodes as episodes_module

    captured = {}

    class RecordingBackend:
        name = "recording"
        config = {}

        def __init__(self, **kwargs):
            captured.update(kwargs)

    class StopHereError(Exception):
        pass

    def stop(path):
        raise StopHereError

    monkeypatch.setattr(backends_module, "TinkerBackend", RecordingBackend)
    monkeypatch.setattr(episodes_module, "EpisodeSet", stop)  # the backend is built before the episode set
    args = cli_module.build_parser().parse_args(
        [
            "agent",
            "--episodes",
            "x",
            "--backend",
            "tinker",
            "--model",
            "m",
            "--renderer",
            "qwen3_8_xhigh_reasoning",
            "--temperature",
            "0.5",
            "--seed",
            "3",
        ]
    )
    with pytest.raises(StopHereError):
        cli_module.cmd_agent(args)
    assert captured["renderer"] == "qwen3_8_xhigh_reasoning"
    assert captured["temperature"] == 0.5 and captured["seed"] == 3 and captured["base_model"] == "m"
