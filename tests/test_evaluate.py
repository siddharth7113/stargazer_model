"""Unit tests for stargazer.evaluate: running a trivial detector over a small set and the Results record."""

import pytest

from stargazer.episodes import BuildConfig, EpisodeSet
from stargazer.evaluate import Results, evaluate
from stargazer.types import Prediction


class AlwaysNone:
    """A detector that never detects; its score is the null fraction, which the tests can compute."""

    name = "always_none"
    config = {"threshold": 1}

    def detect(self, time, flux, flux_err):
        return Prediction(detected=False, extra={"stat": 0.5})


@pytest.fixture
def episodes(fake_cache, tmp_path):
    """Four episodes, seed 3."""
    cache, targets, _ = fake_cache
    return EpisodeSet.build(targets, cache, tmp_path / "set", BuildConfig(n_episodes=4, seed=3))


def test_evaluate_scores_every_episode(episodes):
    """Each episode gets one prediction and one score row."""
    results = evaluate(AlwaysNone(), episodes)
    assert set(results.preds) == set(results.rows) == set(episodes.ids)
    assert results.summary["n_episodes"] == 4


def test_evaluate_mean_score_of_always_none_is_null_fraction(episodes):
    """Saying none scores 1.0 on each null and 0.0 on each transit, so the mean equals the null fraction."""
    results = evaluate(AlwaysNone(), episodes)
    n_null = sum(record["kind"] == "null" for record in episodes.truth.values())
    assert results.summary["mean_score"] == pytest.approx(n_null / 4)


def test_evaluate_subset_of_ids(episodes):
    """A subset runs only those ids and records how many were requested."""
    results = evaluate(AlwaysNone(), episodes, ids=episodes.ids[:2])
    assert set(results.preds) == set(episodes.ids[:2]) and results.config["n_requested"] == 2


def test_evaluate_config_merges_detector_and_tolerances(episodes):
    """The results config holds the detector config, the scoring tolerances and the wall time."""
    config = evaluate(AlwaysNone(), episodes, period_rtol=0.02, single_event_credit=0.4).config
    assert config["threshold"] == 1 and config["period_rtol"] == 0.02 and config["wall_seconds"] >= 0
    assert config["single_event_credit"] == 0.4


def test_evaluate_records_episode_set_hash(episodes):
    """The results file names the exact set by its manifest hash."""
    assert evaluate(AlwaysNone(), episodes).episode_set_hash == episodes.hash


def test_evaluate_progress_callback_sees_each_episode_in_order(episodes):
    """The CLI prints progress from this callback; it must fire once per episode, in id order."""
    seen = []
    evaluate(AlwaysNone(), episodes, progress=lambda episode_id, pred, truth: seen.append(episode_id))
    assert seen == episodes.ids


def test_evaluate_records_detector_error_as_none(episodes):
    """A crash on every second episode becomes a "none" prediction with the error text; the run completes."""

    class Flaky(AlwaysNone):
        calls = 0

        def detect(self, time, flux, flux_err):
            Flaky.calls += 1
            if Flaky.calls % 2 == 0:
                raise RuntimeError("boom")
            return Prediction(detected=False)

    results = evaluate(Flaky(), episodes)
    assert results.config["errors"] == ["ep_0001", "ep_0003"]
    assert results.preds["ep_0001"]["error"] == "RuntimeError: boom"
    assert results.preds["ep_0001"]["detected"] is False


def test_evaluate_can_propagate_errors(episodes):
    """With ``record_errors=False`` the exception reaches the caller."""

    class Broken(AlwaysNone):
        def detect(self, time, flux, flux_err):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        evaluate(Broken(), episodes, record_errors=False)


def test_results_round_trip(episodes, tmp_path):
    """What ``save`` writes, ``load`` reads back with the same predictions, rows, summary and commit."""
    results = evaluate(AlwaysNone(), episodes)
    back = Results.load(results.save(tmp_path / "r.json"))
    assert back.summary == results.summary and back.preds == results.preds and back.rows == results.rows
    assert back.git_commit == results.to_dict()["git_commit"]


def test_results_to_dict_layout(episodes):
    """The JSON layout is what the LOGBOOK and the calibrate command read."""
    doc = evaluate(AlwaysNone(), episodes).to_dict()
    expected_keys = {
        "created",
        "git_commit",
        "system",
        "config",
        "episode_set",
        "episode_set_hash",
        "summary",
        "episodes",
    }
    assert set(doc) == expected_keys
    assert set(doc["episodes"]["ep_0000"]) == {"pred", "score"}


def test_evaluate_with_workers_matches_serial_order_and_values(episodes):
    """A thread pool must give the same predictions in episode order and call progress once per episode."""
    seen = []
    serial = evaluate(AlwaysNone(), episodes)
    parallel = evaluate(
        AlwaysNone(), episodes, workers=4, progress=lambda episode_id, *_: seen.append(episode_id)
    )
    assert list(parallel.preds) == list(serial.preds) == episodes.ids  # results stay in order
    for episode_id in episodes.ids:
        without_timing = {key: value for key, value in parallel.preds[episode_id].items() if key != "seconds"}
        assert without_timing == {
            key: value for key, value in serial.preds[episode_id].items() if key != "seconds"
        }
    assert sorted(seen) == episodes.ids and parallel.config["workers"] == 4  # progress in completion order


def test_evaluate_with_workers_records_errors(episodes):
    """Error recording works the same inside the pool."""

    class Broken(AlwaysNone):
        def detect(self, time, flux, flux_err):
            raise RuntimeError("boom")

    assert evaluate(Broken(), episodes, workers=3).config["errors"] == episodes.ids


def test_evaluate_passes_the_context_only_to_a_detector_that_takes_it(episodes):
    """A detector with a ``context`` keyword receives the episode's context, or None for a set without one."""
    seen = []

    class Reads:
        name = "reads"
        config = {}

        def detect(self, time, flux, flux_err, context=None):
            seen.append(context)
            return Prediction(detected=False)

    evaluate(Reads(), episodes)
    assert len(seen) == len(episodes) and all(context is None for context in seen)
