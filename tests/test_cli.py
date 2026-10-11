"""Unit tests for stargazer.cli: argument defaults and the wiring into the SDK, without network or models."""

from stargazer.cli import build_parser


def test_agent_command_defaults():
    """Unset caps are None (so the detector default applies); limit is 3; backend is openai."""
    args = build_parser().parse_args(["agent", "--episodes", "x", "--model", "m"])
    assert args.max_tool_calls is None and args.limit == 3 and args.backend == "openai"
    assert args.renderer == "qwen3_disable_thinking"


def test_baseline_command_defaults():
    """The baseline scores at the default period tolerance of 1 percent on all episodes."""
    args = build_parser().parse_args(["baseline", "--episodes", "x"])
    assert args.period_rtol == 0.01 and args.limit == 0


def test_show_command_has_width_and_full():
    """The trajectory printer truncates messages at a width unless --full is given."""
    args = build_parser().parse_args(["show", "--results", "r", "--episode", "ep_0001"])
    assert args.width == 400 and args.full is False
