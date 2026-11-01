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
    EpisodeDatasetBuilder,
    EpisodeGroupBuilder,
    EpisodeReward,
    EpisodeTools,
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
