"""End-to-end checks across modules. Each unit is covered in tests/; these check that the parts fit."""

import json

import pytest

from stargazer.agent import AssistantTurn, ToolAgentDetector, ToolCall
from stargazer.baseline import BLSDetector
from stargazer.calibrate import choose_threshold, threshold_sweep
from stargazer.episodes import BuildConfig, EpisodeSet
from stargazer.evaluate import Results, evaluate
from stargazer.score import score_episode

pytestmark = pytest.mark.integration


def test_build_evaluate_save_reload_calibrate(fake_cache, tmp_path):
    """Build a set, run the baseline, save, reload and sweep: hashes agree and deep injections are found."""
    cache, targets, targets_path = fake_cache
    cfg = BuildConfig(n_episodes=12, seed=2, depth_range=(0.005, 0.02))
    episodes = EpisodeSet.build(targets, cache, tmp_path / "set", cfg, targets_path)
    results = evaluate(BLSDetector(sde_threshold=9.0), episodes)
    back = Results.load(results.save(tmp_path / "r.json"))
    assert back.episode_set_hash == episodes.hash
    assert back.summary["completeness_exact"] >= 0.8
    sweep = threshold_sweep(back, episodes.truth, [5, 9, 50])
    assert sweep[0]["false_alarm_rate"] >= sweep[1]["false_alarm_rate"] >= sweep[2]["false_alarm_rate"] == 0.0
    assert choose_threshold(back, episodes.truth, max_false_alarm=0.0) > 0


class BLSFollowingBackend:
    """A scripted agent: run bls_peaks, then submit the strongest peak."""

    name = "scripted"
    config = {}

    def complete(self, messages, tools):
        if not any(message["role"] == "assistant" for message in messages):
            return AssistantTurn(tool_calls=[ToolCall("1", "bls_peaks", {})])
        last_tool = next(message for message in reversed(messages) if message["role"] == "tool")
        peak = json.loads(last_tool["content"])["peaks"][0]
        arguments = {
            "detected": True,
            "period": peak["period"],
            "t0": peak["t0"],
            "duration": peak["duration"],
            "depth": peak["depth_ppm"] / 1e6,
        }
        return AssistantTurn(tool_calls=[ToolCall("2", "submit_prediction", arguments)])


def test_agent_with_real_tools_recovers_injection(injected):
    """The agent loop, the real tools and the scorer together produce a ``hit`` on a strong injection."""
    time, flux, flux_err, truth = injected
    pred = ToolAgentDetector(BLSFollowingBackend()).detect(time, flux, flux_err)
    row = score_episode(pred, {"kind": "transit", "transit": truth.to_dict()})
    assert row["outcome"] == "hit" and row["depth_ok"]
