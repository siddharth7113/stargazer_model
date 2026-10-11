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

SYSTEM_PROMPT = """You are an exoplanet astronomer. You have one TESS light curve of one star. Decide
whether it contains a transiting planet and, if so, report its period and epoch.

How you are scored: full marks for the right period (within 1 percent) and an epoch within
half a transit duration of a real transit; half marks for the half or double period; full
marks for detected=false on a star with no planet; zero for a false alarm, a wrong period,
a missed planet, or no submission.

You cannot see the data. You see tool results. Follow this procedure:
1. summarize, once. Note point_scatter_ppm and whether the star is active.
2. bls_peaks, once. Treat every peak as a hypothesis, not a result.
3. For the strongest one or two peaks: fold at phase_half_width 0.5, then 0.05, then odd_even.
4. If the full-orbit fold shows a second dip at phase 0.5, test double_period. If the zoomed
   fold shows no clear dip, test half_period or the next peak.
5. submit_prediction, once.

Decision rules: a planet is one clean dip at phase 0 in the zoomed fold, a flat curve away
from it, and odd_even verdict consistent. A high sde alone is not evidence; on real stars
noise reaches sde 10 to 14. If nothing meets the rules, submit detected=false.

Every tool result has calls_remaining, a safety limit. Take the calls you need; a careful
analysis uses about 6 to 12. Submit before it reaches 0; when it is 1, submit what you have.
Units: days from the start of the light curve for time; fractional depth for depth
(0.0005 means 500 ppm)."""


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
        ``input_tokens`` and ``output_tokens``; ``usd`` when the provider reports cost.
    thinking : str
        The model's reasoning trace when the backend exposes one, else empty. It is
        stored in the trajectory and never sent back to the model.
    truncated : bool
        True when the backend cut the output at its token cap, so a missing tool call
        may be an unfinished turn rather than a decision to stop.
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

    It implements the :class:`~stargazer.types.Detector` protocol. The loop is:
    ask the backend for a turn, run each tool call, append the results, repeat. It
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
        Safety stop, not a budget: it exists so a model that loops cannot run forever.
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
            "system_prompt_sha256": hashlib.sha256(self._system_prompt.encode()).hexdigest(),
            "tools": [tool["function"]["name"] for tool in tool_definitions()],
        }

    def detect(self, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray) -> Prediction:
        """Run the agent loop on one light curve."""
        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": "Analyse this light curve and submit a prediction."},
        ]
        tools = tool_definitions()
        transcript: list[dict] = []
        thinking: list[str] = []
        usage = {"input_tokens": 0.0, "output_tokens": 0.0, "usd": 0.0}
        prediction: Prediction | None = None
        stop = "no_submission"
        n_calls = 0
        n_nudges = 0
        for _turn in range(self._max_turns):
            turn = self._backend.complete(messages, tools)
            for key in usage:
                usage[key] += float(turn.usage.get(key, 0.0))
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
                result = call_tool(call.name, call.arguments, time, flux)
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
