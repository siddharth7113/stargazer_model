"""Tier 3: the benchmark as a reinforcement-learning environment on Tinker.

An episode is one light curve with its truth record. In the benchmark a detector
reads the light curve through tools and submits a prediction; the score of that
prediction against the truth is the number the LOGBOOK reports. In reinforcement
learning the same episode is an environment: the observation is the conversation so
far (the system prompt, the tool declarations, the tool results), an action is the
model's next message (a tool call), and the reward at the end of the episode is the
score of the submission. Nothing about the task changes between evaluation and
training; the policy that is trained is the same model that the benchmark samples.

The module maps the benchmark onto the Tinker cookbook's interface [1]_ with no
parsing of its own. The cookbook's tool-use layer takes a list of tools, the initial
messages and a reward function, renders the conversation with the model's own
renderer (the same renderer the sampling backend uses) and parses each response with
it. The twelve tools of :mod:`stargazer.tools` are wrapped one for one; the reward is
:func:`stargazer.score.score_episode`.

Terms
-----
episode
    One light curve (time, flux, flux error) with a hidden truth record: a planet with
    its period and epoch, or no planet.
observation
    The conversation rendered to tokens: system prompt, tool declarations, the user's
    request and every earlier tool call and result.
action
    The tokens the model samples in one turn, which the renderer parses into a
    message with tool calls.
reward
    The score of the submitted prediction against the truth, from 0 to 1, given once
    at the end of the episode. An episode with no submission scores as
    ``detected=false``.
group
    Several environments of the same episode sampled side by side. The cookbook's
    policy-gradient loss centres each trajectory's reward on its group's mean, so a
    group measures what the policy did differently on the same star.

References
----------
.. [1] Thinking Machines Lab (2025). Tinker cookbook: reinforcement learning.
   https://github.com/thinking-machines-lab/tinker-cookbook
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chz
import numpy as np
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tool_use import ToolInput, ToolResult, build_agent_tool_env, simple_tool_result

from stargazer.agent import PROMPTS, ToolAgentDetector
from stargazer.episodes import EpisodeSet
from stargazer.score import score_episode
from stargazer.tools import TOOLS, ToolSpec, call_tool, coerce_arguments
from stargazer.types import Prediction

USER_REQUEST = "Analyse this light curve and submit a prediction."
"""The one user message of every episode, the same as the detector's."""


class EpisodeTools:
    """The toolbox of one episode: its arrays, its context, the call count and the submission.

    Parameters
    ----------
    time, flux : ndarray
        The light curve the tools read.
    context : EpisodeContext or None
        The star's catalogue context, for the tools that read it.
    max_tool_calls : int
        The number each tool result counts down from in ``calls_remaining``, as the
        detector does, so the model paces itself the same way in training.

    Notes
    -----
    One instance belongs to one environment. The cookbook calls one :class:`EpisodeTool`
    for each tool call; all of them share this object, which runs the call through
    :func:`stargazer.tools.call_tool`, counts it and keeps the prediction that an
    accepted ``submit_prediction`` makes.
    """

    def __init__(self, time: np.ndarray, flux: np.ndarray, context, max_tool_calls: int):
        self._time = time
        self._flux = flux
        self._context = context
        self._max_tool_calls = max_tool_calls
        self.n_calls = 0
        self.submission: Prediction | None = None
        self.transcript: list[dict] = []

    def run(self, name: str, arguments: dict) -> dict:
        """Run one tool call and return its result with ``calls_remaining`` added."""
        self.n_calls += 1
        coerced = coerce_arguments(name, arguments)
        result = call_tool(name, coerced, self._time, self._flux, self._context)
        result["calls_remaining"] = self._max_tool_calls - self.n_calls
        self.transcript.append({"tool": name, "arguments": arguments, "result": result})
        if name == "submit_prediction" and "error" not in result:
            self.submission = ToolAgentDetector.submitted_prediction(coerced)
        return result

    def cookbook_tools(self) -> list[EpisodeTool]:
        """The twelve tools as cookbook ``Tool`` objects bound to this episode."""
        return [EpisodeTool(spec, self) for spec in TOOLS]


class EpisodeTool:
    """One benchmark tool in the cookbook's ``Tool`` protocol, bound to one episode's toolbox.

    Parameters
    ----------
    spec : ToolSpec
        The tool's name, description and JSON Schema, as the detector declares them.
    tools : EpisodeTools
        The toolbox that runs the call.

    Notes
    -----
    The cookbook declares the tool to the model through the renderer from
    ``to_spec`` and calls ``run`` with the parsed arguments. The tool's work is numpy
    and runs in a thread so that one slow search does not block the other rollouts
    of the batch. A ``submit_prediction`` call that is accepted stops the episode.
    """

    def __init__(self, spec: ToolSpec, tools: EpisodeTools):
        self._spec = spec
        self._tools = tools

    @property
    def name(self) -> str:
        """The tool name the model calls."""
        return self._spec.name

    @property
    def description(self) -> str:
        """What the tool does, as the detector declares it."""
        return self._spec.description

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """The JSON Schema of the arguments."""
        return self._spec.parameters

    def to_spec(self) -> dict:
        """The declaration the renderer renders: name, description and parameters."""
        return {"name": self.name, "description": self.description, "parameters": self.parameters_schema}

    async def run(self, tool_input: ToolInput) -> ToolResult:
        """Run the call in a thread and return a tool message; an accepted submission stops the episode."""
        result = await asyncio.to_thread(self._tools.run, self.name, tool_input.arguments)
        submitted = self.name == "submit_prediction" and "error" not in result
        return simple_tool_result(
            json.dumps(result), call_id=tool_input.call_id or "", name=self.name, should_stop=submitted
        )


class EpisodeReward:
    """The reward of one episode: the benchmark score of the submission against the truth.

    Parameters
    ----------
    tools : EpisodeTools
        The toolbox that holds the submission.
    truth : dict
        The episode's truth record from the set's ``truth.json``.

    Notes
    -----
    The cookbook calls the reward once at the end of the episode with the message
    history. The submission is read from the toolbox and not parsed from the
    messages, so the reward is the score the benchmark would give the same
    trajectory. With no submission the prediction is ``detected=false``, as in
    :class:`stargazer.agent.ToolAgentDetector`. The metrics carry the score, whether
    a submission was made, the call count and the outcome as one-hot numbers, so the
    training log shows false alarms and misses separately from the mean reward.
    """

    OUTCOMES = (
        "true_null",
        "false_alarm",
        "miss",
        "hit",
        "alias",
        "event_hit",
        "wrong_period",
        "wrong_event",
    )

    def __init__(self, tools: EpisodeTools, truth: dict):
        self._tools = tools
        self._truth = truth

    async def __call__(self, messages: list[dict]) -> tuple[float, dict[str, float]]:
        """Return the score as the reward and the metrics; ``messages`` is the history, not read."""
        prediction = self._tools.submission or Prediction(detected=False)
        row = score_episode(prediction, self._truth)
        metrics = {
            "score": float(row["score"]),
            "submitted": float(self._tools.submission is not None),
            "n_tool_calls": float(self._tools.n_calls),
        }
        for outcome in self.OUTCOMES:
            metrics[f"outcome/{outcome}"] = float(row["outcome"] == outcome)
        return float(row["score"]), metrics


class EpisodeGroupBuilder(EnvGroupBuilder):
    """Build a group of environments of one episode.

    Parameters
    ----------
    episode_dir : str
        The episode set's directory.
    episode_id : str
    model_name : str
        The Tinker base model, which names the tokenizer.
    renderer_name : str
        The cookbook renderer of the model family, for example ``tml_v0``.
    system_prompt : str
        The text of the system prompt (one of :data:`stargazer.agent.PROMPTS`).
    group_size : int
        Environments of this episode sampled side by side.
    max_turns, max_tool_calls : int
        The detector's safety stops, applied by the cookbook.
    max_trajectory_tokens : int
        Conversation length at which the cookbook ends the episode.

    Notes
    -----
    The builder holds strings and numbers only, so it pickles across the cookbook's
    workers; the arrays, the renderer and the tools are built in ``make_envs``. The
    initial messages are the renderer's own tool declaration from the twelve tool
    specs and the system prompt, followed by the user's request, exactly as
    :class:`stargazer.backends.TinkerBackend` builds them for evaluation.
    """

    def __init__(
        self,
        episode_dir: str,
        episode_id: str,
        model_name: str,
        renderer_name: str,
        system_prompt: str,
        group_size: int,
        max_turns: int = ToolAgentDetector.DEFAULT_MAX_TURNS,
        max_tool_calls: int = ToolAgentDetector.DEFAULT_MAX_TOOL_CALLS,
        max_trajectory_tokens: int = 60_000,
    ):
        self.episode_dir = episode_dir
        self.episode_id = episode_id
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.system_prompt = system_prompt
        self.group_size = group_size
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls
        self.max_trajectory_tokens = max_trajectory_tokens
        self.example_id = episode_id

    def _renderer(self):
        from tinker_cookbook import renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        return renderers.get_renderer(self.renderer_name, get_tokenizer(self.model_name))

    def initial_messages(self, renderer) -> list[dict]:
        """The tool declaration, the system prompt and the user's request, in the renderer's format."""
        specs = [
            {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
            for tool in TOOLS
        ]
        prefix = renderer.create_conversation_prefix_with_tools(specs, system_prompt=self.system_prompt)
        return [*prefix, {"role": "user", "content": USER_REQUEST}]

    def make_episode_tools(self, episodes: EpisodeSet) -> EpisodeTools:
        """The toolbox of one new environment of this episode."""
        time, flux, _flux_err = episodes.load(self.episode_id)
        return EpisodeTools(time, flux, episodes.context(self.episode_id), self.max_tool_calls)

    async def make_envs(self) -> Sequence[Env]:
        """Load the episode once and build ``group_size`` environments of it."""
        episodes = EpisodeSet(Path(self.episode_dir))
        truth = episodes.truth[self.episode_id]
        renderer = self._renderer()
        initial = self.initial_messages(renderer)
        envs = []
        for _member in range(self.group_size):
            tools = self.make_episode_tools(episodes)
            envs.append(
                build_agent_tool_env(
                    renderer=renderer,
                    tools=tools.cookbook_tools(),
                    initial_messages=initial,
                    reward_fn=EpisodeReward(tools, truth),
                    model_name=self.model_name,
                    max_turns=self.max_turns,
                    max_tool_calls=self.max_tool_calls,
                    max_trajectory_tokens=self.max_trajectory_tokens,
                )
            )
        return envs

    def logging_tags(self) -> list[str]:
        """The episode's kind (``transit`` or ``null``), so the log splits the reward by it."""
        episodes = EpisodeSet(Path(self.episode_dir))
        return [str(episodes.truth[self.episode_id].get("kind", "unknown"))]


class EpisodeDataset(RLDataset):
    """Batches of episode group builders in a fixed order.

    Parameters
    ----------
    builders : list of EpisodeGroupBuilder
    batch_size : int
        Groups in one batch, so a batch samples ``batch_size * group_size`` episodes.
    """

    def __init__(self, builders: list[EpisodeGroupBuilder], batch_size: int):
        self._builders = builders
        self._batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """The builders of batch ``index``."""
        start = index * self._batch_size
        return self._builders[start : start + self._batch_size]

    def __len__(self) -> int:
        """Whole batches in the set."""
        return len(self._builders) // self._batch_size


@chz.chz
class EpisodeDatasetBuilder(RLDatasetBuilder):
    """Build the training dataset from an episode set, and a test dataset from another.

    Parameters
    ----------
    episode_dir : str
        The training episode set.
    model_name, renderer_name : str
    prompt : str
        A name in :data:`stargazer.agent.PROMPTS`.
    batch_size, group_size : int
    max_turns, max_tool_calls, max_trajectory_tokens : int
    seed : int
        Order of the episodes; shuffled once.
    test_episode_dir : str or None
        An episode set evaluated by the cookbook at ``eval_every`` steps, one group of
        one environment for each episode. The held-out set s92_v1 is the choice that
        keeps the LOGBOOK's rule: never a set that was used to design anything.
    """

    episode_dir: str
    model_name: str
    renderer_name: str
    prompt: str = "rules"
    batch_size: int = 8
    group_size: int = 4
    max_turns: int = ToolAgentDetector.DEFAULT_MAX_TURNS
    max_tool_calls: int = ToolAgentDetector.DEFAULT_MAX_TOOL_CALLS
    max_trajectory_tokens: int = 60_000
    seed: int = 0
    test_episode_dir: str | None = None

    def builders(self, episode_dir: str, group_size: int, shuffle: bool) -> list[EpisodeGroupBuilder]:
        """One group builder for each episode of a set."""
        ids = list(EpisodeSet(Path(episode_dir)).ids)
        if shuffle:
            random.Random(self.seed).shuffle(ids)
        return [
            EpisodeGroupBuilder(
                episode_dir=episode_dir,
                episode_id=episode_id,
                model_name=self.model_name,
                renderer_name=self.renderer_name,
                system_prompt=PROMPTS[self.prompt],
                group_size=group_size,
                max_turns=self.max_turns,
                max_tool_calls=self.max_tool_calls,
                max_trajectory_tokens=self.max_trajectory_tokens,
            )
            for episode_id in ids
        ]

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        """The training dataset and, when a test set is named, the test dataset."""
        train = EpisodeDataset(
            self.builders(self.episode_dir, self.group_size, shuffle=True), self.batch_size
        )
        test = None
        if self.test_episode_dir:
            test = EpisodeDataset(self.builders(self.test_episode_dir, 1, shuffle=False), self.batch_size)
        return train, test

    @property
    def record(self) -> dict[str, Any]:
        """What a results file records about the dataset: the sets by hash and every setting."""
        return {
            "episode_set": self.episode_dir,
            "episode_set_hash": EpisodeSet(Path(self.episode_dir)).hash,
            "test_episode_set": self.test_episode_dir,
            "test_episode_set_hash": (
                EpisodeSet(Path(self.test_episode_dir)).hash if self.test_episode_dir else None
            ),
            "system_prompt": self.prompt,
            "system_prompt_sha256": hashlib.sha256(PROMPTS[self.prompt].encode()).hexdigest(),
            "tools": [tool.name for tool in TOOLS],
            "batch_size": self.batch_size,
            "group_size": self.group_size,
            "max_turns": self.max_turns,
            "max_tool_calls": self.max_tool_calls,
            "max_trajectory_tokens": self.max_trajectory_tokens,
            "seed": self.seed,
        }


class TrainingRun:
    """One LoRA training run of a model on an episode set through the cookbook's loop.

    Parameters
    ----------
    dataset : EpisodeDatasetBuilder
    log_path : str
        Where the cookbook writes its metrics, checkpoints list and rollouts.
    learning_rate : float
    lora_rank : int
    max_tokens : int
        Output tokens for each sampled turn, thinking included.
    max_steps : int or None
        Training steps; None runs through the dataset once.
    eval_every, save_every : int
        Steps between evaluations on the test set and between checkpoints.
    temperature : float
        Sampling temperature during training; the model card value is the default.
    tolerate_rollout_errors : bool
        False (the default) stops the run on the first rollout that raises, so that
        a run never trains on fewer groups than its record says. The cookbook's own
        default for Inkling models skips a failed group and goes on; on 2026-11-01 a
        run whose every group failed at import then ended with exit code 0 and
        three empty steps. True restores the cookbook's default.

    Notes
    -----
    The run's record is written as ``record.json`` in ``log_path`` before training
    starts (the cookbook writes its own ``config.json`` there, without the hashes):
    the git commit, the sets by hash, the prompt hash, the tools and every number,
    so that the LOGBOOK entry and the checkpoint point at the same thing.
    The cookbook's own ``metrics.jsonl`` in the same directory holds the step by
    step reward and the one-hot outcome metrics of :class:`EpisodeReward`.
    """

    def __init__(
        self,
        dataset: EpisodeDatasetBuilder,
        log_path: str,
        learning_rate: float,
        lora_rank: int = 32,
        max_tokens: int = 4096,
        max_steps: int | None = None,
        eval_every: int = 0,
        save_every: int = 20,
        temperature: float = 1.0,
        tolerate_rollout_errors: bool = False,
    ):
        self._dataset = dataset
        self._log_path = log_path
        self._learning_rate = learning_rate
        self._lora_rank = lora_rank
        self._max_tokens = max_tokens
        self._max_steps = max_steps
        self._eval_every = eval_every
        self._save_every = save_every
        self._temperature = temperature
        self._tolerate_rollout_errors = tolerate_rollout_errors

    @property
    def record(self) -> dict[str, Any]:
        """The run's settings with the dataset's record and the git commit."""
        from stargazer.provenance import GitState, utc_timestamp

        return {
            "created": utc_timestamp(),
            "git_commit": str(GitState.current()),
            "model_name": self._dataset.model_name,
            "renderer": self._dataset.renderer_name,
            "learning_rate": self._learning_rate,
            "lora_rank": self._lora_rank,
            "max_tokens": self._max_tokens,
            "max_steps": self._max_steps,
            "eval_every": self._eval_every,
            "save_every": self._save_every,
            "temperature": self._temperature,
            "tolerate_rollout_errors": self._tolerate_rollout_errors,
            "log_path": self._log_path,
            "dataset": self._dataset.record,
        }

    def config(self):
        """The cookbook ``train.Config`` of this run."""
        from tinker_cookbook.rl import train

        return train.Config(
            model_name=self._dataset.model_name,
            recipe_name="stargazer",
            renderer_name=self._dataset.renderer_name,
            log_path=self._log_path,
            dataset_builder=self._dataset,
            learning_rate=self._learning_rate,
            lora_rank=self._lora_rank,
            max_tokens=self._max_tokens,
            max_steps=self._max_steps,
            eval_every=self._eval_every,
            save_every=self._save_every,
            temperature=self._temperature,
            rollout_error_tolerance=None if self._tolerate_rollout_errors else False,
        )

    def run(self) -> Path:
        """Write the record, run the training loop and return the log directory."""
        from tinker_cookbook.rl import train

        log_dir = Path(self._log_path)
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "record.json").write_text(json.dumps(self.record, indent=1, sort_keys=True))
        asyncio.run(train.main(self.config()))
        return log_dir
