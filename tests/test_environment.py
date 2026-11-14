"""Unit tests for stargazer.environment: the episode toolbox, the reward, the group builder and the dataset.

The model is a stub here: the tests feed assistant messages with tool calls straight into the cookbook's
message environment, which is what the sampled tokens become after the renderer parses them.
"""

import asyncio
import json
import pickle

import pytest

pytest.importorskip("tinker_cookbook")

from tinker_cookbook.renderers import ToolCall

from stargazer.agent import PROMPTS
from stargazer.environment import (
    USER_REQUEST,
    CheckpointPruner,
    EpisodeDatasetBuilder,
    EpisodeGroupBuilder,
    EpisodeReward,
    EpisodeTools,
    PenalisedEpisodeReward,
    TrainingRun,
)
from stargazer.episodes import BuildConfig, EpisodeSet

MODEL, RENDERER = "thinkingmachines/Inkling-Small", "tml_v0"


def _assistant_call(name: str, arguments: dict, call_id: str = "c1") -> dict:
    """An assistant message with one parsed tool call, as the renderer produces it."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            ToolCall(id=call_id, function=ToolCall.FunctionBody(name=name, arguments=json.dumps(arguments)))
        ],
    }


@pytest.fixture
def episodes(fake_cache, tmp_path):
    """A six-episode synthetic set: transits and nulls from the three fake hosts."""
    cache, targets, targets_path = fake_cache
    return EpisodeSet.build(targets, cache, tmp_path / "set", BuildConfig(n_episodes=6, seed=1), targets_path)


def _builder(episodes, episode_id, group_size=1):
    return EpisodeGroupBuilder(
        episode_dir=str(episodes.root),
        episode_id=episode_id,
        model_name=MODEL,
        renderer_name=RENDERER,
        system_prompt=PROMPTS["rules"],
        group_size=group_size,
    )


def _null_and_transit(episodes):
    truth = episodes.truth
    null_id = next(ep for ep, record in truth.items() if record["kind"] == "null")
    transit_id = next(ep for ep, record in truth.items() if record["kind"] == "transit")
    return null_id, transit_id


def test_episode_tools_count_calls_and_keep_the_accepted_submission(episodes):
    """Each call counts down ``calls_remaining`` as the detector does, and an accepted ``submit_prediction``
    becomes the toolbox's submission with the arguments coerced, so the reward reads the same prediction
    the benchmark would score.
    """
    null_id, _ = _null_and_transit(episodes)
    time, flux, _ = episodes.load(null_id)
    tools = EpisodeTools(time, flux, None, max_tool_calls=5)
    result = tools.run("summarize", {})
    assert result["calls_remaining"] == 4 and tools.submission is None
    tools.run("submit_prediction", {"detected": "false", "reasoning": "nothing"})
    assert tools.submission is not None and tools.submission.detected is False
    assert tools.n_calls == 2 and [entry["tool"] for entry in tools.transcript] == [
        "summarize",
        "submit_prediction",
    ]


def test_reward_is_the_benchmark_score_with_the_outcome_as_metrics(episodes):
    """A correct decline of a null scores 1 with ``outcome/true_null`` set; a false alarm scores 0 with
    ``outcome/false_alarm`` set; no submission scores as ``detected=false``.
    """
    null_id, _ = _null_and_transit(episodes)
    time, flux, _ = episodes.load(null_id)
    truth = episodes.truth[null_id]

    declined = EpisodeTools(time, flux, None, 40)
    declined.run("submit_prediction", {"detected": False})
    reward, metrics = asyncio.run(EpisodeReward(declined, truth)([]))
    assert reward == 1.0 and metrics["outcome/true_null"] == 1.0 and metrics["submitted"] == 1.0

    alarm = EpisodeTools(time, flux, None, 40)
    alarm.run(
        "submit_prediction", {"detected": True, "period": 3.0, "t0": 1.0, "duration": 0.1, "depth": 0.001}
    )
    reward, metrics = asyncio.run(EpisodeReward(alarm, truth)([]))
    assert reward == 0.0 and metrics["outcome/false_alarm"] == 1.0

    silent = EpisodeTools(time, flux, None, 40)
    reward, metrics = asyncio.run(EpisodeReward(silent, truth)([]))
    assert reward == 1.0 and metrics["submitted"] == 0.0 and metrics["n_tool_calls"] == 0.0


def test_group_builder_pickles_and_declares_the_tools_through_the_renderer(episodes):
    """The builder carries strings and numbers only, so it pickles for the cookbook's workers, and its
    initial messages are the renderer's own tool declaration for the twelve tools plus the user's request.
    """
    null_id, _ = _null_and_transit(episodes)
    builder = _builder(episodes, null_id, group_size=2)
    assert pickle.loads(pickle.dumps(builder)).episode_id == null_id
    messages = builder.initial_messages(builder._renderer())
    assert [message["role"] for message in messages] == ["system", "tool_declare", "user"]
    assert messages[-1]["content"] == USER_REQUEST
    assert "submit_prediction" in str(messages[1]["content"]) and "search_signals" in str(
        messages[1]["content"]
    )


def test_environment_runs_a_stub_trajectory_to_the_submission_and_the_reward(episodes):
    """Stepping the message environment with a ``summarize`` call then an accepted ``submit_prediction``
    returns the tool result as a tool message, ends the episode on the submission and pays the score as
    the reward: 1 for a declined null.
    """
    null_id, _ = _null_and_transit(episodes)
    env = asyncio.run(_builder(episodes, null_id).make_envs())[0]
    message_env = env.message_env
    asyncio.run(message_env.initial_observation())

    step = asyncio.run(message_env.step(_assistant_call("summarize", {})))
    tool_message = step.next_messages[-1]
    assert not step.episode_done and tool_message["role"] == "tool"
    assert json.loads(tool_message["content"])["calls_remaining"] == 39

    step = asyncio.run(message_env.step(_assistant_call("submit_prediction", {"detected": False}, "c2")))
    assert step.episode_done and step.reward == 1.0 and step.metrics["outcome/true_null"] == 1.0


def test_environment_rewards_a_found_planet_and_scores_the_truth_period(episodes):
    """On a transit episode a submission at the truth period and epoch pays 1, so the reward is the score
    the LOGBOOK reports and nothing shaped on top of it.
    """
    _, transit_id = _null_and_transit(episodes)
    truth = episodes.truth[transit_id]["transit"]
    env = asyncio.run(_builder(episodes, transit_id).make_envs())[0]
    message_env = env.message_env
    asyncio.run(message_env.initial_observation())
    arguments = {
        "detected": True,
        "period": truth["period"],
        "t0": truth["t0"],
        "duration": truth["duration"],
        "depth": truth["depth"],
    }
    step = asyncio.run(message_env.step(_assistant_call("submit_prediction", arguments)))
    assert step.episode_done and step.reward == 1.0 and step.metrics["outcome/hit"] == 1.0


def test_token_environment_renders_the_initial_observation(episodes):
    """The token-level environment's first observation is a rendered prompt with stop sequences, the form the
    training loop samples from.
    """
    null_id, _ = _null_and_transit(episodes)
    env = asyncio.run(_builder(episodes, null_id).make_envs())[0]
    observation, stop = asyncio.run(env.initial_observation())
    assert observation.length > 100 and len(stop) > 0


def test_dataset_builder_batches_every_episode_and_records_the_sets_by_hash(episodes):
    """The training dataset holds one group for each episode in a seeded order, the test dataset one
    environment for each episode in file order, and the record names both sets by hash with the prompt
    hash and the tools.
    """
    builder = EpisodeDatasetBuilder(
        episode_dir=str(episodes.root),
        model_name=MODEL,
        renderer_name=RENDERER,
        batch_size=2,
        group_size=3,
        test_episode_dir=str(episodes.root),
    )
    train, test = asyncio.run(builder())
    assert len(train) == 3 and len(train.get_batch(0)) == 2 and train.get_batch(0)[0].group_size == 3
    assert [group.episode_id for group in test.get_batch(0)] == episodes.ids[:2]
    assert test.get_batch(0)[0].group_size == 1
    record = builder.record
    assert record["episode_set_hash"] == episodes.hash and record["test_episode_set_hash"] == episodes.hash
    assert len(record["tools"]) == 12 and record["system_prompt"] == "rules"


def test_training_run_fails_fast_on_rollout_errors_and_records_it(episodes):
    """By default a run stops on the first rollout error instead of skipping the group, so the record's
    batch size is what was trained on; the cookbook config carries the choice and the record names it.
    """
    dataset = EpisodeDatasetBuilder(episode_dir=str(episodes.root), model_name=MODEL, renderer_name=RENDERER)
    run = TrainingRun(dataset, log_path=str(episodes.root / "log"), learning_rate=1e-5)
    assert run.config().rollout_error_tolerance is False and run.record["tolerate_rollout_errors"] is False
    tolerant = TrainingRun(dataset, log_path="x", learning_rate=1e-5, tolerate_rollout_errors=True)
    assert tolerant.config().rollout_error_tolerance is None


def test_training_run_refuses_more_steps_than_the_set_has_whole_batches(episodes):
    """Check that a training run refuses a step count above the number of whole batches in its set.

    One step trains on one batch of episode groups. The cookbook's loop stops at the last whole batch
    and does not report the stop. A larger ``max_steps`` would make ``record.json``, the run's record,
    name steps that did not run. The six-episode set in batches of two has three whole batches, so the
    run accepts three steps and refuses four.
    """
    dataset = EpisodeDatasetBuilder(
        episode_dir=str(episodes.root), model_name=MODEL, renderer_name=RENDERER, batch_size=2
    )
    assert TrainingRun(dataset, log_path="x", learning_rate=1e-5, max_steps=3).record["max_steps"] == 3
    with pytest.raises(ValueError, match="whole batches"):
        TrainingRun(dataset, log_path="x", learning_rate=1e-5, max_steps=4)


def test_environment_carries_the_detectors_caps_on_both_sides_and_grades_a_capped_episode(episodes):
    """The runner's limits and the message environment's cap are the detector's 30 turns and 40 calls, not
    the cookbook's 10-turn preset for Inkling models, and the termination policy leaves a capped episode's
    reward as the benchmark scores it; on 2026-11-01 the preset cut up to a third of a step's rollouts at
    10 turns with reward 0 and no grading.
    """
    null_id, _ = _null_and_transit(episodes)
    builder = _builder(episodes, null_id)
    env = asyncio.run(builder.make_envs())[0]
    assert env.rollout_limits.max_turns == 30 and env.rollout_limits.max_tool_calls == 40
    assert env.message_env.max_turns == 30 and env.message_env.max_tool_calls == 40
    assert builder.rollout_config().termination.zero_reward_on_limit is False
    dataset = EpisodeDatasetBuilder(episode_dir=str(episodes.root), model_name=MODEL, renderer_name=RENDERER)
    assert (
        TrainingRun(dataset, log_path="x", learning_rate=1e-5).config().termination.zero_reward_on_limit
        is False
    )


def test_training_run_discards_the_checkpoints_of_a_run_without_saving(tmp_path):
    """A run with ``save_every`` 0 keeps no checkpoint: every path in ``checkpoints.jsonl`` is deleted through
    the REST client, the file is emptied so a resume loads nothing, and the paths are recorded.
    """

    class FakeFuture:
        def result(self):
            return None

    class FakeRest:
        def __init__(self):
            self.deleted = []

        def delete_checkpoint_from_tinker_path(self, path):
            self.deleted.append(path)
            return FakeFuture()

    (tmp_path / "checkpoints.jsonl").write_text(
        json.dumps(
            {
                "name": "final",
                "state_path": "tinker://a/weights/final",
                "sampler_path": "tinker://a/sampler_weights/final",
            }
        )
        + "\n"
    )
    rest = FakeRest()
    deleted = TrainingRun.discard_checkpoints(tmp_path, rest)
    assert deleted == rest.deleted == ["tinker://a/weights/final", "tinker://a/sampler_weights/final"]
    assert (tmp_path / "checkpoints.jsonl").read_text() == ""
    assert (tmp_path / "discarded.txt").read_text().splitlines() == deleted
    assert TrainingRun.discard_checkpoints(tmp_path / "none", rest) == []


def test_checkpoint_pruner_keeps_every_sampler_and_only_the_newest_state(tmp_path):
    """As checkpoints are listed, the training state of every one but the newest is deleted once, the
    sampler weights are never touched, a half-written line is skipped, and the deleted paths are recorded;
    the free store then holds one state and every sampler checkpoint.
    """

    class FakeFuture:
        def result(self):
            return None

    class FakeRest:
        def __init__(self):
            self.deleted = []

        def delete_checkpoint_from_tinker_path(self, path):
            self.deleted.append(path)
            return FakeFuture()

    def entry(name):
        return json.dumps(
            {
                "name": name,
                "state_path": f"tinker://r/weights/{name}",
                "sampler_path": f"tinker://r/sampler_weights/{name}",
            }
        )

    listing = tmp_path / "checkpoints.jsonl"
    rest = FakeRest()
    pruner = CheckpointPruner(tmp_path, rest, interval_seconds=0.01)
    assert pruner.prune() == []
    listing.write_text(entry("000010") + "\n")
    assert pruner.prune() == []
    listing.write_text(entry("000010") + "\n" + entry("000020") + "\n" + '{"name": "0000')
    assert pruner.prune() == ["tinker://r/weights/000010"]
    assert pruner.prune() == []
    listing.write_text(entry("000010") + "\n" + entry("000020") + "\n" + entry("final") + "\n")
    assert pruner.stop() == ["tinker://r/weights/000020"]
    assert rest.deleted == ["tinker://r/weights/000010", "tinker://r/weights/000020"]
    assert all("sampler" not in path for path in rest.deleted)
    assert (tmp_path / "pruned.txt").read_text().splitlines() == rest.deleted


def test_penalised_reward_pays_a_false_alarm_below_zero_and_keeps_the_score_in_the_metrics(episodes):
    """The penalised reward differs from the plain score on one outcome only: a submission on a null star
    pays -1 instead of 0, while a decline pays 1 and a found planet 1; ``metrics["score"]`` stays the
    benchmark score, so the training log still reports the LOGBOOK's number.
    """
    null_id, transit_id = _null_and_transit(episodes)
    time, flux, _ = episodes.load(null_id)
    alarm = EpisodeTools(time, flux, None, 40)
    alarm.run(
        "submit_prediction", {"detected": True, "period": 3.0, "t0": 1.0, "duration": 0.1, "depth": 0.001}
    )
    reward, metrics = asyncio.run(PenalisedEpisodeReward(alarm, episodes.truth[null_id])([]))
    assert reward == -1.0 and metrics["score"] == 0.0 and metrics["reward"] == -1.0
    declined = EpisodeTools(time, flux, None, 40)
    declined.run("submit_prediction", {"detected": False})
    assert asyncio.run(PenalisedEpisodeReward(declined, episodes.truth[null_id])([]))[0] == 1.0
    truth = episodes.truth[transit_id]["transit"]
    time, flux, _ = episodes.load(transit_id)
    found = EpisodeTools(time, flux, None, 40)
    found.run("submit_prediction", {"detected": True, "period": truth["period"], "t0": truth["t0"]})
    assert asyncio.run(PenalisedEpisodeReward(found, episodes.truth[transit_id])([]))[0] == 1.0
    missed = EpisodeTools(time, flux, None, 40)
    missed.run("submit_prediction", {"detected": False})
    assert asyncio.run(PenalisedEpisodeReward(missed, episodes.truth[transit_id])([]))[0] == 0.0


def test_group_builder_and_dataset_carry_the_reward_name_and_refuse_an_unknown_one(episodes):
    """The reward is chosen by name, recorded in the dataset's record and refused when unknown, so a run's
    record always says which reward trained it.
    """
    null_id, _ = _null_and_transit(episodes)
    with pytest.raises(ValueError, match="reward"):
        EpisodeGroupBuilder(str(episodes.root), null_id, MODEL, RENDERER, PROMPTS["rules"], 1, reward="bonus")
    builder = EpisodeDatasetBuilder(
        episode_dir=str(episodes.root), model_name=MODEL, renderer_name=RENDERER, reward="penalised"
    )
    train, _ = asyncio.run(builder())
    assert train.get_batch(0)[0].reward == "penalised" and builder.record["reward"] == "penalised"
    assert (
        EpisodeDatasetBuilder(
            episode_dir=str(episodes.root), model_name=MODEL, renderer_name=RENDERER
        ).record["reward"]
        == "score"
    )
