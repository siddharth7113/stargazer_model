"""Unit tests for stargazer.cli: argument defaults and the wiring into the SDK, without network or models."""

import json

import pytest

from stargazer.cli import build_parser


def test_agent_command_defaults():
    """Unset caps are None (so the detector default applies); limit is 3; backend is openai."""
    args = build_parser().parse_args(["agent", "--episodes", "x", "--model", "m"])
    assert args.max_tool_calls is None and args.limit == 3 and args.backend == "openai"
    assert args.renderer == "qwen3_disable_thinking"
    assert args.prompt == "base"
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
            "--max-tokens",
            "32768",
            "--seed",
            "3",
        ]
    )
    with pytest.raises(StopHereError):
        cli_module.cmd_agent(args)
    assert captured["renderer"] == "qwen3_8_xhigh_reasoning"
    assert captured["temperature"] == 0.5 and captured["seed"] == 3 and captured["base_model"] == "m"
    assert captured["max_tokens"] == 32768


def test_agent_command_forwards_the_turn_and_tool_call_caps(monkeypatch):
    """--max-turns and --max-tool-calls reach the detector; an unapplied cap cuts a run off silently."""
    import stargazer.agent as agent_module
    import stargazer.backends as backends_module
    import stargazer.cli as cli_module
    import stargazer.episodes as episodes_module

    captured = {}

    class RecordingBackend:
        name = "recording"
        config = {}

        def __init__(self, **kwargs):
            pass

    def fake_detector(backend, **kwargs):
        captured.update(kwargs)
        raise StopHereError

    monkeypatch.setattr(backends_module, "TinkerBackend", RecordingBackend)
    monkeypatch.setattr(agent_module, "ToolAgentDetector", fake_detector)
    monkeypatch.setattr(episodes_module, "EpisodeSet", lambda path: object())  # opened before the detector
    args = cli_module.build_parser().parse_args(
        [
            "agent",
            "--episodes",
            "x",
            "--backend",
            "tinker",
            "--model",
            "m",
            "--max-turns",
            "60",
            "--max-tool-calls",
            "80",
        ]
    )
    with pytest.raises(StopHereError):
        cli_module.cmd_agent(args)
    assert captured["max_turns"] == 60 and captured["max_tool_calls"] == 80


def test_agent_command_leaves_max_tokens_to_the_model_card_by_default(monkeypatch):
    """Without --max-tokens the Tinker backend receives None, so the model card value applies."""
    import stargazer.backends as backends_module
    import stargazer.cli as cli_module
    import stargazer.episodes as episodes_module

    captured = {}

    class RecordingBackend:
        name = "recording"
        config = {}

        def __init__(self, **kwargs):
            captured.update(kwargs)

    def stop(path):
        raise StopHereError

    monkeypatch.setattr(backends_module, "TinkerBackend", RecordingBackend)
    monkeypatch.setattr(episodes_module, "EpisodeSet", stop)
    args = cli_module.build_parser().parse_args(
        ["agent", "--episodes", "x", "--backend", "tinker", "--model", "m"]
    )
    with pytest.raises(StopHereError):
        cli_module.cmd_agent(args)
    assert captured["max_tokens"] is None


def test_build_command_forwards_arguments(monkeypatch, tmp_path):
    """Every build flag reaches BuildConfig; a mismatch here silently builds the wrong set."""
    import stargazer.cli as cli_module
    import stargazer.episodes as episodes_module
    import stargazer.targets as targets_module

    captured = {}

    def fake_build(targets, cache, out_dir, cfg, targets_path=None):
        captured.update({"out": out_dir, "cfg": cfg, "targets_path": targets_path})
        raise StopHereError

    monkeypatch.setattr(targets_module, "load_targets", lambda path: {"targets": []})
    monkeypatch.setattr(episodes_module.EpisodeSet, "build", staticmethod(fake_build))
    args = cli_module.build_parser().parse_args(
        [
            "build",
            "--targets",
            "t.json",
            "--out",
            str(tmp_path),
            "--n",
            "7",
            "--seed",
            "3",
            "--max-period",
            "60",
        ]
    )
    with pytest.raises(StopHereError):
        cli_module.cmd_build(args)
    assert captured["cfg"].n_episodes == 7 and captured["cfg"].seed == 3
    assert captured["cfg"].period_range == (0.5, 60.0) and captured["cfg"].depth_range == (1e-4, 1e-2)


class StopHereError(Exception):
    """Raised by a stub to end a command once the call under test has been captured."""


def test_targets_command_forwards_arguments(monkeypatch):
    """The targets flags reach select_targets with the right names."""
    import stargazer.cli as cli_module
    import stargazer.targets as targets_module

    captured = {}

    def fake_select(sector, n_targets, seed, tmag_max):
        captured.update({"sector": sector, "n_targets": n_targets, "seed": seed, "tmag_max": tmag_max})
        raise StopHereError

    monkeypatch.setattr(targets_module, "select_targets", fake_select)
    args = cli_module.build_parser().parse_args(["targets", "--sector", "98", "--n", "12", "--seed", "4"])
    args.out = "x"
    with pytest.raises(StopHereError):
        cli_module.cmd_targets(args)
    assert captured == {"sector": 98, "n_targets": 12, "seed": 4, "tmag_max": 12.0}


def test_remove_binaries_command_reads_one_targets_file_and_writes_another(monkeypatch, tmp_path):
    """The command filters the --targets file and saves to --out; the input file is not overwritten."""
    import stargazer.cli as cli_module
    import stargazer.targets as targets_module

    source = tmp_path / "in.json"
    source.write_text(json.dumps({"created": "t", "targets": [{"tic": 1}, {"tic": 2}]}))
    monkeypatch.setattr(
        targets_module,
        "remove_binaries",
        lambda targets: {
            **targets,
            "n_selected": 1,
            "removed_binaries": [{"tic": 2, "evidence": ["tess_eb"]}],
        },
    )
    args = cli_module.build_parser().parse_args(
        ["remove-binaries", "--targets", str(source), "--out", str(tmp_path / "out.json")]
    )
    cli_module.cmd_remove_binaries(args)
    assert json.loads((tmp_path / "out.json").read_text())["removed_binaries"][0]["tic"] == 2
    assert json.loads(source.read_text())["targets"] == [{"tic": 1}, {"tic": 2}]


def test_context_command_takes_an_episode_set_and_a_targets_file():
    """The context command names the set to annotate and the targets file that holds the positions."""
    args = build_parser().parse_args(["context", "--episodes", "x", "--targets", "t.json"])
    assert args.episodes == "x" and args.targets == "t.json"
