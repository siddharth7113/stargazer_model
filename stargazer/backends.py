"""Model providers for :class:`stargazer.agent.ToolAgentDetector`.

:class:`OpenAICompatibleBackend` speaks the chat-completions API with function calling.
OpenRouter, vLLM, Ollama and many hosted providers accept it. :class:`TinkerBackend`
samples from a base model or a trained LoRA checkpoint on Tinker. The
reinforcement-learning phase uses the same path, so a policy is evaluated with the same
tools it was trained on.

Neither backend is imported until it is constructed, so the core package has no
dependency on either SDK. Install with ``uv pip install -e ".[openai]"`` or
``".[tinker]"``.
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from typing import Any

from stargazer.agent import AssistantTurn, ToolCall


class OpenAICompatibleBackend:
    """Chat completions with function calling over any OpenAI-compatible endpoint.

    Parameters
    ----------
    model : str
        Provider model id, for example ``"qwen/qwen3-32b"`` on OpenRouter.
    base_url : str
        Default is OpenRouter. Use ``http://localhost:11434/v1`` for Ollama or a vLLM URL.
    api_key_env : str
        Environment variable that holds the key. The key itself is never stored in
        config or results.
    max_tokens : int
        Output cap per turn.
    temperature : float
        Default 0.7, the value model cards and the Tinker cookbook use for tool use.
        Temperature 0 does not make a hosted model deterministic and degrades quality.
    seed : int
        Sent to the provider when it supports seeded sampling; recorded in config.

    Notes
    -----
    The request sets ``extra_body={"usage": {"include": True}}``. OpenRouter then
    returns the cost of the call in US dollars in ``usage.cost``, which the backend
    reports as ``usd``. Other providers ignore the field and no ``usd`` is reported.
    """

    name = "openai_compatible"

    def __init__(
        self,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key_env: str = "OPENROUTER_API_KEY",
        max_tokens: int = 4000,
        temperature: float = 0.7,
        seed: int = 0,
    ):
        self._model = model
        self._base_url = base_url
        self._api_key_env = api_key_env
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._seed = seed
        self._client = None
        self._client_lock = threading.Lock()

    @property
    def config(self) -> dict[str, Any]:
        """Model, endpoint and sampling settings. No secrets."""
        return {
            "model": self._model,
            "base_url": self._base_url,
            "api_key_env": self._api_key_env,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "seed": self._seed,
        }

    def _get_client(self):
        with self._client_lock:
            if self._client is not None:
                return self._client
            from openai import OpenAI

            key = os.environ.get(self._api_key_env)
            if not key:
                raise RuntimeError(f"set {self._api_key_env} to use {self._base_url}")
            self._client = OpenAI(base_url=self._base_url, api_key=key)
        return self._client

    def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        """One chat-completions call with ``tool_choice="auto"``."""
        resp = self._get_client().chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            seed=self._seed,
            extra_body={"usage": {"include": True}},
        )
        choice = resp.choices[0]
        calls = []
        for tc in choice.message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_malformed": tc.function.arguments}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        usage = {
            "input_tokens": float(getattr(resp.usage, "prompt_tokens", 0) or 0),
            "output_tokens": float(getattr(resp.usage, "completion_tokens", 0) or 0),
        }
        cost = getattr(resp.usage, "cost", None)
        if cost is not None:
            usage["usd"] = float(cost)
        return AssistantTurn(
            content=choice.message.content or "",
            tool_calls=calls,
            usage=usage,
            truncated=choice.finish_reason == "length",
        )


SAMPLING_DEFAULTS: dict[str, dict[str, float | int]] = {
    "thinking": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 8192},
    "non_thinking": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_tokens": 2048},
}
"""Sampling parameters recommended by the Qwen3 model cards [1]_ for each mode.

The cards discourage greedy decoding (temperature 0) because it degrades quality and can
cause endless repetition. The Tinker cookbook uses 0.7 for its tool-use benchmark, which
agrees with the non-thinking row. ``max_tokens`` is the output cap per turn: a thinking
model writes its reasoning before the tool call, so it needs several thousand tokens;
the cards suggest 32k for hard problems, and 8192 is enough for one analysis step.

References
----------
.. [1] Qwen Team (2025). Qwen3 model cards, "Best Practices" section.
   https://huggingface.co/Qwen/Qwen3-8B
"""


TINKER_SAMPLING_USD_PER_MILLION: dict[str, float] = {
    "Qwen/Qwen3-8B": 0.60,
    "Qwen/Qwen3.5-4B": 1.005,
    "Qwen/Qwen3.5-9B": 0.807,
    "Qwen/Qwen3.5-35B-A3B-Base": 1.335,
    "Qwen/Qwen3.6-35B-A3B": 1.335,
    "Qwen/Qwen3.6-27B": 5.595,
    "Qwen/Qwen3.8-27B": 5.595,
    "Qwen/Qwen3.5-397B-A17B": 7.50,
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16": 0.495,
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": 0.495,
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": 1.44,
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16": 6.225,
    "thinkingmachines/Inkling-Small": 1.44,
    "thinkingmachines/Inkling": 4.68,
    "moonshotai/Kimi-K2.6": 5.49,
    "openai/gpt-oss-20b": 0.45,
    "openai/gpt-oss-120b": 0.84,
    "zai-org/GLM-5.3-Flash:peft:262144": 2.11,
    "zai-org/GLM-5.3:peft:262144": 12.15,
    "deepseek-ai/DeepSeek-V4.1-Flash:peft:262144": 3.60,
    "deepseek-ai/DeepSeek-V3.1": 4.215,
}
"""Tinker sampling price in US dollars per million tokens, from the models and pricing
page read on 2026-10-11. Sampling is billed on prompt plus sampled tokens. The table is
an estimate for reporting; the Tinker console is the bill. A model not listed here
gets no cost field.
"""


def recommended_sampling(renderer: str) -> dict[str, float | int]:
    """Return the sampling defaults for a renderer name.

    A renderer whose name contains ``disable_thinking`` or ``instruct`` runs the model in
    non-thinking mode; every other renderer is treated as thinking mode.
    """
    mode = "non_thinking" if ("disable_thinking" in renderer or "instruct" in renderer) else "thinking"
    return dict(SAMPLING_DEFAULTS[mode])


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


class TinkerBackend:
    """Sample from a base model or a LoRA checkpoint on Tinker, in the model's native tool format.

    Parameters
    ----------
    base_model : str
        For example ``"Qwen/Qwen3.8-27B"``.
    model_path : str or None
        A ``tinker://`` path to saved weights. None samples the base model.
    renderer : str
        Chat template name from ``tinker_cookbook.renderers``, for example
        ``qwen3_8_disable_thinking``, ``nemotron3_disable_thinking`` or ``tml_v0``.
    max_tokens, temperature, top_p, top_k : int or float, optional
        None means the value from :func:`recommended_sampling` for the renderer.
    seed : int
        Sampling seed, recorded in config. Repeat a run with other seeds to measure
        variance. A run with one seed gives one sample of the result.

    Notes
    -----
    Tinker exposes sampling and has no chat-completions API. The cookbook renderer for
    each model family knows that family's wire format: how tools are declared, how a
    tool call is written and how a tool result is fed back. This backend therefore
    does three things through the renderer and nothing by hand:

    1. ``create_conversation_prefix_with_tools`` turns the tool definitions and the
       system prompt into the family's own tool declaration.
    2. ``build_generation_prompt`` renders the conversation, including earlier tool
       calls and results, into tokens.
    3. ``parse_response`` turns the sampled tokens into a message with parsed
       ``tool_calls``, the text, and any thinking part.

    A tool call the renderer could not parse is reported in ``unparsed_tool_calls``;
    the backend keeps its raw text in the turn content so the trajectory shows what
    the model wrote, and the agent loop's nudge asks for a proper call.
    """

    name = "tinker"

    def __init__(
        self,
        base_model: str = "Qwen/Qwen3-8B",
        model_path: str | None = None,
        renderer: str = "qwen3_disable_thinking",
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int = 0,
    ):
        self._base_model = base_model
        self._model_path = model_path
        self._renderer_name = renderer
        defaults = recommended_sampling(renderer)
        self._max_tokens = defaults["max_tokens"] if max_tokens is None else max_tokens
        self._temperature = defaults["temperature"] if temperature is None else temperature
        self._top_p = defaults["top_p"] if top_p is None else top_p
        self._top_k = defaults["top_k"] if top_k is None else top_k
        self._seed = seed
        self._sampler = None
        self._renderer = None
        self._tokenizer = None
        self._setup_lock = threading.Lock()

    @property
    def config(self) -> dict[str, Any]:
        """Base model, checkpoint path, renderer and sampling settings."""
        return {
            "base_model": self._base_model,
            "model_path": self._model_path,
            "renderer": self._renderer_name,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "top_k": self._top_k,
            "seed": self._seed,
        }

    def _setup(self):
        with self._setup_lock:
            if self._sampler is not None:
                return
            import tinker
            from tinker_cookbook import renderers
            from tinker_cookbook.tokenizer_utils import get_tokenizer

            service = tinker.ServiceClient()
            self._tokenizer = get_tokenizer(self._base_model)
            self._renderer = renderers.get_renderer(self._renderer_name, self._tokenizer)
            if self._model_path:
                self._sampler = service.create_sampling_client(model_path=self._model_path)
            else:
                self._sampler = service.create_sampling_client(base_model=self._base_model)

    @staticmethod
    def to_cookbook_messages(messages: list[dict]) -> list[dict]:
        """Convert OpenAI-format messages (without the system message) to cookbook messages.

        Returns
        -------
        list of dict
            Cookbook ``Message`` dicts. Assistant tool calls become cookbook ``ToolCall``
            objects. A tool result carries ``tool_call_id`` and the function ``name``,
            looked up from the assistant call it answers, because some formats need
            the id and others the name.
        """
        from tinker_cookbook.renderers import ToolCall as CookbookToolCall

        names_by_id: dict[str, str] = {}
        out: list[dict] = []
        for message in messages:
            role = message["role"]
            if role == "system":
                continue
            if role == "assistant":
                converted: dict = {"role": "assistant", "content": message.get("content") or ""}
                calls = []
                for call in message.get("tool_calls", []):
                    function = call["function"]
                    arguments = function["arguments"]
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments)
                    names_by_id[call["id"]] = function["name"]
                    calls.append(
                        CookbookToolCall(
                            function=CookbookToolCall.FunctionBody(
                                name=function["name"], arguments=arguments
                            ),
                            id=call["id"],
                        )
                    )
                if calls:
                    converted["tool_calls"] = calls
                out.append(converted)
            elif role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "content": message["content"],
                        "tool_call_id": message.get("tool_call_id", ""),
                        "name": names_by_id.get(message.get("tool_call_id", ""), ""),
                    }
                )
            else:
                out.append({"role": role, "content": message["content"]})
        return out

    @staticmethod
    def split_content(content) -> tuple[str, str]:
        """Return ``(thinking, text)`` from a cookbook message content.

        Content is a string or a list of parts; a part with a ``thinking`` key is the
        reasoning trace and a part with a ``text`` key is the answer.
        """
        if isinstance(content, str):
            return "", content
        thinking, text = [], []
        for part in content:
            if "thinking" in part:
                thinking.append(part["thinking"])
            elif "text" in part:
                text.append(part["text"])
        return "\n".join(thinking).strip(), "".join(text).strip()

    @staticmethod
    def parse_tool_calls(text: str) -> list[ToolCall]:
        """Extract ``<tool_call>`` JSON blocks from plain text. A fallback for text without a renderer."""
        calls = []
        for raw in TOOL_CALL_RE.findall(text):
            try:
                parsed = json.loads(raw)
                calls.append(
                    ToolCall(
                        id=uuid.uuid4().hex[:8],
                        name=str(parsed["name"]),
                        arguments=dict(parsed.get("arguments", {})),
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        return calls

    def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        """Render with the model's own tool format, sample once, parse with the renderer."""
        from tinker import types

        self._setup()
        system_prompt = next((message["content"] for message in messages if message["role"] == "system"), "")
        tool_specs = [tool["function"] for tool in tools]
        prefix = self._renderer.create_conversation_prefix_with_tools(tool_specs, system_prompt=system_prompt)
        prompt = self._renderer.build_generation_prompt(prefix + self.to_cookbook_messages(messages))
        params = types.SamplingParams(
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
            top_k=self._top_k,
            seed=self._seed,
            stop=self._renderer.get_stop_sequences(),
        )
        result = self._sampler.sample(prompt=prompt, num_samples=1, sampling_params=params).result()
        sequence = result.sequences[0]
        tokens = list(sequence.tokens)
        message, termination = self._renderer.parse_response(tokens)
        thinking, text = self.split_content(message.get("content", ""))
        calls = [
            ToolCall(
                id=call.id or uuid.uuid4().hex[:8],
                name=call.function.name,
                arguments=_arguments_dict(call.function.arguments),
            )
            for call in message.get("tool_calls", [])
        ]
        unparsed = message.get("unparsed_tool_calls") or []
        if unparsed:
            text = (
                text + "\n" + "\n".join(getattr(call, "raw_text", str(call)) for call in unparsed)
            ).strip()
        usage = {"input_tokens": float(prompt.length), "output_tokens": float(len(tokens))}
        price = TINKER_SAMPLING_USD_PER_MILLION.get(self._base_model)
        if price is not None:
            usage["usd"] = (prompt.length + len(tokens)) * price / 1e6
        return AssistantTurn(
            content=text,
            tool_calls=calls,
            usage=usage,
            thinking=thinking,
            truncated=getattr(sequence, "stop_reason", "stop") == "length" or str(termination) == "malformed",
        )


def _arguments_dict(arguments) -> dict:
    """Parse a tool call's arguments, which the renderer gives as a JSON string."""
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {"_malformed": arguments}
    return parsed if isinstance(parsed, dict) else {"_malformed": arguments}
