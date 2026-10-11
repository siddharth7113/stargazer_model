"""Run a :class:`Detector` over an :class:`EpisodeSet` and write a provenance-carrying result."""

from __future__ import annotations

import json
import time as _time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stargazer.episodes import EpisodeSet
from stargazer.provenance import git_commit, utc_now
from stargazer.score import score_set
from stargazer.types import Detector, Prediction


@dataclass
class Results:
    """The data needed to reproduce and audit one evaluation run.

    Parameters
    ----------
    system : str
        Detector name.
    config : dict
        Detector configuration plus scoring tolerances.
    episode_set, episode_set_hash : str
        The episode set path and the manifest hash that identifies it.
    preds, rows : dict
        Per-episode predictions and per-episode score rows.
    summary : dict
        Output of :func:`stargazer.score.summarize`.
    """

    system: str
    config: dict
    episode_set: str
    episode_set_hash: str
    preds: dict[str, dict]
    rows: dict[str, dict]
    summary: dict
    created: str = ""
    git_commit: str | None = None

    def to_dict(self) -> dict:
        """Return the JSON document layout."""
        return {
            "created": self.created or utc_now(),
            "git_commit": self.git_commit if self.git_commit is not None else git_commit(),
            "system": self.system,
            "config": self.config,
            "episode_set": self.episode_set,
            "episode_set_hash": self.episode_set_hash,
            "summary": self.summary,
            "episodes": {k: {"pred": self.preds.get(k), "score": self.rows[k]} for k in sorted(self.rows)},
        }

    def save(self, path: Path) -> Path:
        """Write the results JSON. Create parent directories if needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True, default=_json_default))
        return path

    @classmethod
    def load(cls, path: Path) -> Results:
        """Read a results JSON written by :meth:`save`."""
        d = json.loads(Path(path).read_text())
        return cls(
            system=d["system"],
            config=d["config"],
            episode_set=d["episode_set"],
            episode_set_hash=d["episode_set_hash"],
            preds={k: v["pred"] for k, v in d["episodes"].items()},
            rows={k: v["score"] for k, v in d["episodes"].items()},
            summary=d["summary"],
            created=d["created"],
            git_commit=d["git_commit"],
        )


def evaluate(
    detector: Detector,
    episodes: EpisodeSet,
    ids: list[str] | None = None,
    period_rtol: float = 0.01,
    epoch_frac: float = 0.5,
    progress: Callable[[str, Prediction, dict], None] | None = None,
) -> Results:
    """Run a detector on each episode and score the predictions.

    Parameters
    ----------
    detector : Detector
    episodes : EpisodeSet
    ids : list of str, optional
        Subset of episode ids. Default is all episodes.
    period_rtol, epoch_frac : float
        Scoring tolerances, recorded in the results config.
    progress : callable, optional
        Called with ``(ep_id, prediction, truth_record)`` after each episode.
    """
    truth = episodes.truth
    ids = list(ids) if ids is not None else episodes.ids
    preds: dict[str, dict] = {}
    t_start = _time.time()
    for k in ids:
        t, f, fe = episodes.load(k)
        t1 = _time.time()
        p = detector.detect(t, f, fe)
        p.extra["seconds"] = round(_time.time() - t1, 2)
        preds[k] = p.to_dict()
        if progress:
            progress(k, p, truth[k])
    truth_sub = {k: truth[k] for k in ids}
    rows, summary = score_set(preds, truth_sub, period_rtol=period_rtol, epoch_frac=epoch_frac)
    config = {
        **detector.config,
        "period_rtol": period_rtol,
        "epoch_frac": epoch_frac,
        "wall_seconds": round(_time.time() - t_start, 1),
        "n_requested": len(ids),
    }
    return Results(
        system=detector.name,
        config=config,
        episode_set=str(episodes.root),
        episode_set_hash=episodes.hash,
        preds=preds,
        rows=rows,
        summary=summary,
    )


def _json_default(o):
    if isinstance(o, np.floating | np.integer):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))
