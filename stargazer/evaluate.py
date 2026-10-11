"""Run a :class:`Detector` over an :class:`EpisodeSet` and write a provenance-carrying result."""

from __future__ import annotations

import json
import time as _time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stargazer.episodes import EpisodeSet
from stargazer.provenance import GitState, utc_timestamp
from stargazer.score import score_set
from stargazer.types import Detector, Prediction


@dataclass
class Results:
    """The output of one evaluation run, with its provenance.

    An evaluation run applies one detector to each episode of one episode set and
    scores each prediction. This record holds each prediction, each score row, the
    set-level summary, the detector configuration, the hash of the episode set, the
    git commit and the time. With these a reader can rerun the evaluation and get
    the same numbers. :meth:`save` writes it as JSON. :meth:`load` reads it back.

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
            "created": self.created or utc_timestamp(),
            "git_commit": self.git_commit if self.git_commit is not None else str(GitState.current()),
            "system": self.system,
            "config": self.config,
            "episode_set": self.episode_set,
            "episode_set_hash": self.episode_set_hash,
            "summary": self.summary,
            "episodes": {
                episode_id: {"pred": self.preds.get(episode_id), "score": self.rows[episode_id]}
                for episode_id in sorted(self.rows)
            },
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
        doc = json.loads(Path(path).read_text())
        return cls(
            system=doc["system"],
            config=doc["config"],
            episode_set=doc["episode_set"],
            episode_set_hash=doc["episode_set_hash"],
            preds={episode_id: entry["pred"] for episode_id, entry in doc["episodes"].items()},
            rows={episode_id: entry["score"] for episode_id, entry in doc["episodes"].items()},
            summary=doc["summary"],
            created=doc["created"],
            git_commit=doc["git_commit"],
        )


def evaluate(
    detector: Detector,
    episodes: EpisodeSet,
    ids: list[str] | None = None,
    period_rtol: float = 0.01,
    epoch_frac: float = 0.5,
    single_event_credit: float = 0.5,
    progress: Callable[[str, Prediction, dict], None] | None = None,
    record_errors: bool = True,
    workers: int = 1,
) -> Results:
    """Run a detector on each episode and score the predictions.

    Parameters
    ----------
    detector : Detector
    episodes : EpisodeSet
    ids : list of str, optional
        Subset of episode ids. Default is all episodes.
    period_rtol, epoch_frac, single_event_credit : float
        Scoring tolerances and the credit of a single-event submission on an episode
        with two or more transits, recorded in the results config.
    progress : callable, optional
        Called with ``(ep_id, prediction, truth_record)`` after each episode.
    record_errors : bool
        If True, an exception in ``detect`` becomes a prediction of "none" with the
        error text in ``extra["error"]``, and the episode id goes into
        ``config["errors"]``. If False, the exception propagates.
    workers : int
        Episodes evaluated at the same time in a thread pool. A detector that waits
        on a remote model is idle most of the time, so 8 to 16 workers give a near
        linear speed-up. The detector must be safe to call from several threads;
        the baseline and the agent are. Results are in episode order whatever the
        completion order. ``progress`` is called from the main thread as each
        episode completes, so a slow episode does not hide the finished ones.

    Notes
    -----
    A detector that crashes on one light curve must not lose a run of two hundred.
    The crash is scored as a miss or a true null, which is what a reader would see
    from that system in practice, and the error is kept so it can be fixed.
    """
    truth = episodes.truth
    ids = list(ids) if ids is not None else episodes.ids
    preds: dict[str, dict] = {}
    errors: list[str] = []
    t_start = _time.time()

    def run_one(episode_id: str) -> Prediction:
        time, flux, flux_err = episodes.load(episode_id)
        t_one = _time.time()
        try:
            prediction = detector.detect(time, flux, flux_err)
        except Exception as error:  # a detector failure is a recorded miss, not a lost run
            if not record_errors:
                raise
            prediction = Prediction(detected=False, extra={"error": f"{type(error).__name__}: {error}"[:500]})
        prediction.extra["seconds"] = round(_time.time() - t_one, 2)
        return prediction

    predictions: dict[str, Prediction] = {}
    if workers <= 1:
        for episode_id in ids:
            predictions[episode_id] = run_one(episode_id)
            if progress:
                progress(episode_id, predictions[episode_id], truth[episode_id])
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_one, episode_id): episode_id for episode_id in ids}
            for future in as_completed(futures):
                episode_id = futures[future]
                predictions[episode_id] = future.result()
                if progress:
                    progress(episode_id, predictions[episode_id], truth[episode_id])
    for episode_id in ids:
        prediction = predictions[episode_id]
        if "error" in prediction.extra:
            errors.append(episode_id)
        preds[episode_id] = prediction.to_dict()
    truth_sub = {episode_id: truth[episode_id] for episode_id in ids}
    rows, summary = score_set(
        preds,
        truth_sub,
        period_rtol=period_rtol,
        epoch_frac=epoch_frac,
        single_event_credit=single_event_credit,
    )
    config = {
        **detector.config,
        "period_rtol": period_rtol,
        "epoch_frac": epoch_frac,
        "single_event_credit": single_event_credit,
        "wall_seconds": round(_time.time() - t_start, 1),
        "n_requested": len(ids),
        "errors": errors,
        "workers": workers,
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


def _json_default(obj):
    if isinstance(obj, np.floating | np.integer):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(str(type(obj)))
