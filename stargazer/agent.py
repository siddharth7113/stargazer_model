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

SYSTEM_PROMPT = """You are an astronomer searching one TESS light curve for a transiting planet.

You cannot see the data directly. Use the tools. Start with summarize, then bls_peaks.
Check the strongest peaks with fold at the full orbit and zoomed on the transit, and with
odd_even. A real transit is a single dip at phase zero that is present in both odd and even
events. Consider the aliases at half and twice the best period. A long-period planet may
show only one or two transits, which BLS underweights.

When you are done, call submit_prediction exactly once. If no convincing signal exists,
submit detected=false. Do not guess a detection on noise: a false alarm scores zero.
Time is in days from the start of the light curve. Depth is fractional, not ppm."""


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
    """

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, float] = field(default_factory=dict)


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
    cap. The full transcript of tool calls is kept in ``Prediction.extra`` so a reader
    can audit the path to the answer.

    Parameters
    ----------
    backend : ChatBackend
    max_tool_calls : int
        Hard cap on tool calls per episode. Reaching it ends the episode as "none".
    max_turns : int
        Hard cap on model calls per episode.
    max_tokens_per_episode : int
        Hard cap on input plus output tokens per episode, from the backend's usage.
    system_prompt : str
    """

    name = "tool_agent"

    def __init__(
        self,
        backend: ChatBackend,
        max_tool_calls: int = 12,
        max_turns: int = 16,
        max_tokens_per_episode: int = 150_000,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self._backend = backend
        self._max_tool_calls = max_tool_calls
        self._max_turns = max_turns
        self._max_tokens = max_tokens_per_episode
        self._system_prompt = system_prompt

    @property
    def config(self) -> dict[str, Any]:
        """Backend config plus the caps and a hash of the system prompt."""
        return {
            "backend": self._backend.name,
            **{f"backend_{k}": v for k, v in self._backend.config.items()},
            "max_tool_calls": self._max_tool_calls,
            "max_turns": self._max_turns,
            "max_tokens_per_episode": self._max_tokens,
            "system_prompt_sha256": hashlib.sha256(self._system_prompt.encode()).hexdigest(),
            "tools": [t["function"]["name"] for t in tool_definitions()],
        }

    def detect(self, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray) -> Prediction:
        """Run the agent loop on one light curve."""
        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": "Analyse this light curve and submit a prediction."},
        ]
        tools = tool_definitions()
        transcript: list[dict] = []
        usage = {"input_tokens": 0.0, "output_tokens": 0.0, "usd": 0.0}
        prediction: Prediction | None = None
        stop = "no_submission"
        n_calls = 0
        for _turn in range(self._max_turns):
            turn = self._backend.complete(messages, tools)
            for k in usage:
                usage[k] += float(turn.usage.get(k, 0.0))
            messages.append(_assistant_message(turn))
            if not turn.tool_calls:
                stop = "model_stopped"
                break
            for call in turn.tool_calls:
                n_calls += 1
                result = call_tool(call.name, call.arguments, time, flux)
                transcript.append({"tool": call.name, "arguments": call.arguments, "result": result})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
                if call.name == "submit_prediction" and "error" not in result:
                    prediction = Prediction.from_dict(
                        {k: v for k, v in call.arguments.items() if k != "reasoning"}
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
                "n_turns": len([m for m in messages if m["role"] == "assistant"]),
                "usage": usage,
                "transcript": transcript,
            }
        )
        return pred


def _assistant_message(turn: AssistantTurn) -> dict:
    msg: dict[str, Any] = {"role": "assistant", "content": turn.content or ""}
    if turn.tool_calls:
        msg["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
            }
            for c in turn.tool_calls
        ]
    return msg
