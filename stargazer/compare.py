"""Compare several results files on the same episode set.

A fair comparison between a thresholded detector and an agent needs two things: the
same episodes, checked by the episode set hash, and the false-alarm rate reported next
to the completeness, because an agent has no threshold to sweep. This module builds
that table from results files, with completeness in each expected signal-to-noise
ratio (SNR) bin.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stargazer.evaluate import Results
from stargazer.score import SNR_BINS


@dataclass(frozen=True)
class SystemSummary:
    """One row of the comparison table.

    Parameters
    ----------
    label : str
        System name plus the model when there is one.
    n_episodes : int
    false_alarm_rate : float or None
    completeness_exact : float or None
    completeness_with_aliases : float or None
    mean_score : float
    tokens_per_episode : float or None
        Mean of input plus output tokens, for systems that report usage.
    seconds_per_episode : float
    by_snr : dict
        Completeness for each SNR bin label.
    """

    label: str
    n_episodes: int
    false_alarm_rate: float | None
    completeness_exact: float | None
    completeness_with_aliases: float | None
    mean_score: float
    tokens_per_episode: float | None
    seconds_per_episode: float
    by_snr: dict[str, float | None]


def summarize_results(results: Results) -> SystemSummary:
    """Reduce one results file to a comparison row."""
    config = results.config
    model = config.get("backend_base_model") or config.get("backend_model") or ""
    label = results.system + (f" {model}" if model else "")
    usage = [pred.get("usage") for pred in results.preds.values() if isinstance(pred.get("usage"), dict)]
    tokens = (
        float(np.mean([entry.get("input_tokens", 0) + entry.get("output_tokens", 0) for entry in usage]))
        if usage
        else None
    )
    seconds = float(np.mean([pred.get("seconds", 0.0) for pred in results.preds.values()]))
    summary = results.summary
    return SystemSummary(
        label=label,
        n_episodes=summary["n_episodes"],
        false_alarm_rate=summary["false_alarm_rate"],
        completeness_exact=summary["completeness_exact"],
        completeness_with_aliases=summary["completeness_with_aliases"],
        mean_score=summary["mean_score"],
        tokens_per_episode=tokens,
        seconds_per_episode=seconds,
        by_snr={label_: bin_["completeness"] for label_, bin_ in summary["by_snr"].items()},
    )


def compare(paths: list[Path]) -> tuple[list[SystemSummary], str]:
    """Load results files, check they share one episode set, and summarise each.

    Returns
    -------
    rows : list of SystemSummary
    episode_set_hash : str

    Raises
    ------
    ValueError
        If the files were scored on different episode sets or different subsets.
    """
    loaded = [Results.load(path) for path in paths]
    hashes = {results.episode_set_hash for results in loaded}
    if len(hashes) != 1:
        raise ValueError(f"results come from different episode sets: {sorted(hashes)}")
    sizes = {len(results.rows) for results in loaded}
    if len(sizes) != 1:
        raise ValueError(f"results cover different numbers of episodes: {sorted(sizes)}")
    return [summarize_results(results) for results in loaded], hashes.pop()


def format_table(rows: list[SystemSummary]) -> str:
    """Render the comparison as a Markdown table, main metrics then SNR bins."""

    def pct(value: float | None) -> str:
        return "-" if value is None else f"{100 * value:.0f}%"

    columns = [
        "system",
        "episodes",
        "false alarm",
        "completeness",
        "with aliases",
        "mean score",
        "tokens/ep",
        "s/ep",
    ]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        tokens = "-" if row.tokens_per_episode is None else f"{row.tokens_per_episode:,.0f}"
        cells = [
            row.label,
            str(row.n_episodes),
            pct(row.false_alarm_rate),
            pct(row.completeness_exact),
            pct(row.completeness_with_aliases),
            f"{row.mean_score:.3f}",
            tokens,
            f"{row.seconds_per_episode:.0f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    bin_labels = [f"{lo}-{'inf' if hi == np.inf else hi}" for lo, hi in SNR_BINS]
    lines += ["", "| system | " + " | ".join(f"SNR {label}" for label in bin_labels) + " |"]
    lines.append("|---|" + "---|" * len(bin_labels))
    for row in rows:
        cells = [pct(row.by_snr.get(label)) for label in bin_labels]
        lines.append(f"| {row.label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
