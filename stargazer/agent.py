"""A tool-using language-model agent as a detector, independent of the model provider.

The agent loop lives here. A :class:`ChatBackend` only has to turn a list of chat
messages and a list of tool definitions into one assistant turn. Backends for
OpenAI-compatible APIs (OpenRouter, vLLM, Ollama) and for Tinker are in
:mod:`stargazer.backends`. The messages use the OpenAI chat format, which every
backend either accepts directly or renders to text.

The agent never sees raw data. It calls the tools in :mod:`stargazer.tools` and ends
by calling ``submit_prediction``. The benchmark scores that call with the same scorer
as the baseline.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from stargazer.tools import call_tool, tool_definitions
from stargazer.types import Prediction

BASE_PROMPT = """You are an exoplanet astronomer. You have one TESS light curve of one star. Decide
whether it contains a transiting planet and, if so, report its period and epoch, or the
time of its one transit if the data hold only one.

How you are scored: full marks for the right period (within 1 percent) and an epoch within
half a transit duration of a real transit; half marks for the half or double period; full
marks for a single event whose t0 falls on the only transit in the data, half marks if the
data held more transits; full marks for detected=false on a star with no planet; zero for
a false alarm, a wrong period, an event on no transit, a missed planet, or no submission.

You cannot see the data. You see tool results. The tools are the ones an astronomer uses
to find and vet a transit; each returns numbers and, where there is a judgement, a verdict
word. Use them as you see fit, then call submit_prediction once. Every tool result has
calls_remaining, a safety limit; submit before it reaches 0. Units: days from the start of
the light curve for time; fractional depth for depth (0.0005 means 500 ppm)."""

PROMPTS: dict[str, str] = {}
"""The system prompts by name. ``base`` states the task, the scoring and the units and
leaves the method to the model. ``rules`` adds the procedure and the decision rules that
the runs of 2026-10-11 produced, one for each failure read in a trajectory. The gap
between the two on one model is the value of the hand-written policy, which the
reinforcement-learning phase must learn from the reward instead."""

RULES_PROMPT = """You are an exoplanet astronomer. You have one TESS light curve of one star. Decide
whether it contains a transiting planet and, if so, report its period and epoch, or the
time of its one transit if the data hold only one.

How you are scored: full marks for the right period (within 1 percent) and an epoch within
half a transit duration of a real transit; half marks for the half or double period; full
marks for a single event whose t0 falls on the only transit in the data, half marks if the
data held more transits and a period was there to find; full marks for detected=false on a
star with no planet; zero for a false alarm, a wrong period, an event on no transit, a
missed planet, or no submission.

You cannot see the data. You see tool results. Follow this procedure:
1. summarize, once. Note point_scatter_ppm and whether the star is active.
2. search_signals, once. It finds the strongest peak, masks it and searches again until the
   rest is noise. Treat every signal as a hypothesis, not a result. If residual_sde is near
   min_sde, bls_peaks with the signals in masks shows what is left. Read n_events_in_data:
   a signal with 1 is one dip paired with empty time, so it is a single event and not a
   period; go to step 6 for it.
3. For each signal, strongest first: fold at phase_half_width 0.5, then 0.05, then fit_transit,
   then odd_even, then secondary_eclipse. fit_transit must say density plausible; an
   implausible density or a v_shaped shape means an eclipsing binary or a wrong period.
   grazing_uncertain means a short transit the fit cannot settle: decide by odd_even,
   secondary_eclipse and the depth, and do not reject on the shape alone. odd_even
   insufficient_data with one event of one parity and none of the other means the fold
   holds one dip: a single event, not a period. secondary_eclipse must say no_secondary;
   secondary_at_half_phase is a binary.
4. If the full-orbit fold shows a second dip at phase 0.5, test double_period. If the zoomed
   fold shows no clear dip, test half_period or the next signal.
5. If a signal fails the checks, go to the next one: a strong feature that is not a planet
   still hides weaker signals, and search_signals already removed it for you.
6. If no periodic signal convinces you, call single_events. A single deep event can be a
   long-period planet with one transit in the data; look for a second event of the same
   depth and fold at the implied period before you decide. If there is no partner, submit
   the event alone: detected=true with t0, duration and depth and no period.
7. submit_prediction, once.

Decision rules: a periodic planet is one clean dip at phase 0 in the zoomed fold, a flat
curve away from it, a fit_transit density plausible and a shape that is not v_shaped, odd_even
verdict consistent, and secondary_eclipse verdict no_secondary. A single event is one dip with snr
above about 10, a duration of hours, a depth below 5 percent (50,000 ppm; deeper is a
stellar companion, not a planet), and no partner of the same depth. On a star whose flux
range is more than 5 percent, a lone dip is variability unless a second event of the same
depth and duration exists. A high sde alone is not evidence; on real stars noise reaches
sde 10 to 14. If nothing meets the rules, submit detected=false.

Every tool result has calls_remaining, a safety limit. Take the calls you need; a careful
analysis uses about 6 to 12. Submit before it reaches 0; when it is 1, submit what you have.
Units: days from the start of the light curve for time; fractional depth for depth
(0.0005 means 500 ppm)."""

PROMPTS.update({"base": BASE_PROMPT, "rules": RULES_PROMPT})
SYSTEM_PROMPT = BASE_PROMPT
"""The default system prompt, ``base``."""


@dataclass
class ToolCall:
    """One tool call the model asked for.

    Parameters
    ----------
    id : str
        Identifier the backend uses to match the result to the call.
    name : str
    arguments : dict
        Parsed JSON arguments.
    """

    id: str
    name: str
    arguments: dict


@dataclass
class AssistantTurn:
    """What a backend returns for one model call.

    Parameters
    ----------
    content : str
        Text the model wrote, possibly empty.
    tool_calls : list of ToolCall
    usage : dict
        ``input_tokens`` and ``output_tokens``; ``usd`` only when the backend knows the
        cost, from the provider or from a price table. Never zero as a placeholder.
    thinking : str
        The model's reasoning trace when the backend exposes one, else empty. It is
        stored in the trajectory and never sent back to the model.
    truncated : bool
        True when the backend cut the output at its token cap. In such a turn, a
        missing tool call can mean an unfinished turn. It is then not a decision to stop.
    """

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, float] = field(default_factory=dict)
    thinking: str = ""
    truncated: bool = False


class ChatBackend(Protocol):
    """The one method a model provider has to implement to run the agent.

    ``messages`` is a list of OpenAI-format chat messages: ``system``, ``user``,
    ``assistant`` (with optional ``tool_calls``) and ``tool`` (with ``tool_call_id``).
    ``tools`` is the list from :func:`stargazer.tools.tool_definitions`.
    """

    name: str

    @property
    def config(self) -> dict[str, Any]:
        """Model name and sampling settings. Must fully determine the output distribution."""
        ...

    def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        """Return the next assistant turn."""
        ...


class ToolAgentDetector:
    """Run a chat model with analysis tools on one light curve and return its prediction.

    It implements the :class:`~stargazer.types.Detector` protocol. In each step, the
    loop asks the backend for a turn, runs each tool call and appends the results. It
    ends when the model calls ``submit_prediction``, stops calling tools, or hits a
    cap. ``Prediction.extra`` keeps the whole trajectory: ``conversation`` is the
    message list as the backend saw it, with the model's text and tool calls;
    ``transcript`` is the list of tool calls with their results; ``thinking`` is the
    per-turn reasoning trace when the backend exposes one. A reader can replay the
    path to the answer from the results file alone.

    Parameters
    ----------
    backend : ChatBackend
    max_tool_calls : int, optional
        Safety stop, so that a model that loops cannot run forever. It is not a budget.
        Reaching it ends the episode as "none". Each tool result carries
        ``calls_remaining`` so the model can pace itself; a careful run uses 6 to 12.
        None means the class default of 40.
    max_turns : int, optional
        Hard cap on model calls per episode. None means 30.
    max_nudges : int
        How many times a turn without a tool call gets a reminder before the episode
        ends as ``model_stopped``. A thinking model that runs out of output tokens
        mid-sentence, or answers in prose, gets a second chance this way.
    max_tokens_per_episode : int, optional
        Hard cap on input plus output tokens per episode, from the backend's usage. The
        conversation is re-sent on every turn, so the total grows with the square of the
        number of calls; 20 calls with lean tool outputs cost about 150,000 tokens.
    system_prompt : str
        One of :data:`PROMPTS` or any other text. The config records its name when it
        is one of the named prompts, and its hash in every case.
    """

    name = "tool_agent"

    DEFAULT_MAX_TOOL_CALLS = 40
    DEFAULT_MAX_TURNS = 30
    DEFAULT_MAX_TOKENS = 400_000
    DEFAULT_MAX_NUDGES = 2
    NUDGE = (
        "Your last reply contained no tool call, so nothing happened. Reply with a tool call: "
        "continue the analysis, or call submit_prediction with your answer."
    )
    NUDGE_TRUNCATED = (
        "Your last reply was cut off at the output limit before any tool call. Keep your text "
        "short and reply with a tool call: continue the analysis, or call submit_prediction."
    )

    def __init__(
        self,
        backend: ChatBackend,
        max_tool_calls: int | None = None,
        max_turns: int | None = None,
        max_tokens_per_episode: int | None = None,
        max_nudges: int | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self._backend = backend
        self._max_tool_calls = self.DEFAULT_MAX_TOOL_CALLS if max_tool_calls is None else max_tool_calls
        self._max_turns = self.DEFAULT_MAX_TURNS if max_turns is None else max_turns
        self._max_tokens = (
            self.DEFAULT_MAX_TOKENS if max_tokens_per_episode is None else max_tokens_per_episode
        )
        self._max_nudges = self.DEFAULT_MAX_NUDGES if max_nudges is None else max_nudges
        self._system_prompt = system_prompt

    @property
    def config(self) -> dict[str, Any]:
        """Backend config plus the caps and a hash of the system prompt."""
        return {
            "backend": self._backend.name,
            **{f"backend_{key}": value for key, value in self._backend.config.items()},
            "max_tool_calls": self._max_tool_calls,
            "max_turns": self._max_turns,
            "max_tokens_per_episode": self._max_tokens,
            "max_nudges": self._max_nudges,
            "system_prompt": next(
                (name for name, text in PROMPTS.items() if text == self._system_prompt), "custom"
            ),
            "system_prompt_sha256": hashlib.sha256(self._system_prompt.encode()).hexdigest(),
            "tools": [tool["function"]["name"] for tool in tool_definitions()],
        }

    def detect(self, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray, context=None) -> Prediction:
        """Run the agent loop on one light curve.

        Parameters
        ----------
        time, flux, flux_err : ndarray
        context : EpisodeContext, optional
            The star's catalogue context, passed to the tools that read it.
        """
        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": "Analyse this light curve and submit a prediction."},
        ]
        tools = tool_definitions()
        transcript: list[dict] = []
        thinking: list[str] = []
        usage: dict[str, float] = {"input_tokens": 0.0, "output_tokens": 0.0}
        prediction: Prediction | None = None
        stop = "no_submission"
        n_calls = 0
        n_nudges = 0
        for _turn in range(self._max_turns):
            turn = self._backend.complete(messages, tools)
            for key, value in turn.usage.items():
                usage[key] = usage.get(key, 0.0) + float(value)
            messages.append(_assistant_message(turn))
            thinking.append(turn.thinking)
            if not turn.tool_calls:
                if n_nudges >= self._max_nudges:
                    stop = "model_stopped"
                    break
                n_nudges += 1
                nudge = self.NUDGE_TRUNCATED if turn.truncated else self.NUDGE
                messages.append({"role": "user", "content": nudge})
                continue
            for call in turn.tool_calls:
                n_calls += 1
                result = call_tool(call.name, call.arguments, time, flux, context)
                result["calls_remaining"] = self._max_tool_calls - n_calls
                transcript.append({"tool": call.name, "arguments": call.arguments, "result": result})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
                if call.name == "submit_prediction" and "error" not in result:
                    prediction = Prediction.from_dict(
                        {key: value for key, value in call.arguments.items() if key != "reasoning"}
                    )
                    prediction.extra["reasoning"] = call.arguments.get("reasoning", "")
                    stop = "submitted"
            if prediction is not None:
                break
            if n_calls >= self._max_tool_calls:
                stop = "max_tool_calls"
                break
            if usage["input_tokens"] + usage["output_tokens"] >= self._max_tokens:
                stop = "max_tokens"
                break
        else:
            stop = "max_turns"
        pred = prediction or Prediction(detected=False)
        pred.extra.update(
            {
                "stop": stop,
                "n_tool_calls": n_calls,
                "n_nudges": n_nudges,
                "n_turns": len([message for message in messages if message["role"] == "assistant"]),
                "usage": usage,
                "transcript": transcript,
                "conversation": messages[1:],  # the system prompt is identified by its hash in config
                "thinking": thinking,
            }
        )
        return pred


def _assistant_message(turn: AssistantTurn) -> dict:
    msg: dict[str, Any] = {"role": "assistant", "content": turn.content or ""}
    if turn.tool_calls:
        msg["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in turn.tool_calls
        ]
    return msg
