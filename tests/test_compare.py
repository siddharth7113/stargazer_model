"""Unit tests for stargazer.compare: summarising results files and the comparison table."""

import pytest

from stargazer.compare import SystemSummary, compare, format_table, summarize_results
from stargazer.evaluate import Results

SUMMARY = {
    "n_episodes": 2,
    "false_alarm_rate": 0.5,
    "completeness_exact": 1.0,
    "completeness_with_aliases": 1.0,
    "mean_score": 0.75,
    "by_snr": {"0-5": {"n": 0, "completeness": None}, "10-20": {"n": 1, "completeness": 1.0}},
}


def _results(path, system="tool_agent", episode_set_hash="h", n=2):
    preds = {
        f"ep_{index:04d}": {
            "detected": True,
            "usage": {"input_tokens": 100, "output_tokens": 10},
            "seconds": 2.0,
        }
        for index in range(n)
    }
    rows = {key: {"outcome": "hit", "score": 1.0} for key in preds}
    config = {"backend_base_model": "m"} if system == "tool_agent" else {}
    results = Results(system, config, "set", episode_set_hash, preds, rows, {**SUMMARY, "n_episodes": n})
    return results.save(path)


def test_summarize_results_reads_metrics_and_usage(tmp_path):
    """The row carries the summary metrics, the model label and the mean tokens per episode."""
    row = summarize_results(Results.load(_results(tmp_path / "a.json")))
    assert isinstance(row, SystemSummary)
    assert row.label == "tool_agent m" and row.false_alarm_rate == 0.5 and row.tokens_per_episode == 110.0
    assert row.by_snr["10-20"] == 1.0 and row.seconds_per_episode == 2.0


def test_compare_requires_one_episode_set(tmp_path):
    """Files from different episode sets cannot be compared."""
    first = _results(tmp_path / "a.json")
    other = _results(tmp_path / "b.json", episode_set_hash="other")
    with pytest.raises(ValueError, match="different episode sets"):
        compare([first, other])


def test_compare_requires_same_number_of_episodes(tmp_path):
    """A three-episode smoke run is not comparable with a full run."""
    with pytest.raises(ValueError, match="different numbers"):
        compare([_results(tmp_path / "a.json"), _results(tmp_path / "b.json", n=3)])


def test_format_table_has_one_row_per_system(tmp_path):
    """The Markdown table lists each system once in the metrics block and once in the SNR block."""
    rows, _ = compare([_results(tmp_path / "a.json"), _results(tmp_path / "b.json", system="baseline_bls")])
    table = format_table(rows)
    assert table.count("| tool_agent m |") == 2 and table.count("| baseline_bls |") == 2
    assert "50%" in table and "100%" in table
