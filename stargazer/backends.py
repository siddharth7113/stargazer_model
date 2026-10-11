"""Model providers for :class:`stargazer.agent.ToolAgentDetector`.

:class:`OpenAICompatibleBackend` speaks the chat-completions API with function calling.
OpenRouter, vLLM, Ollama and many hosted providers accept it. :class:`TinkerBackend`
samples from a base model or a trained LoRA checkpoint on Tinker, which is also the
path the reinforcement-learning phase uses, so a policy is evaluated with the same
tools it was trained on.

Neither backend is imported until it is constructed, so the core package has no
dependency on either SDK. Install with ``uv pip install -e ".[openai]"`` or
``".[tinker]"``.
"""

from __future__ import annotations

import json
import os
import re
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
        Zero for reproducible evaluation.

    Notes
    -----
    The request sets ``extra_body={"usage": {"include": True}}``. OpenRouter then
    returns the cost of the call in US dollars in ``usage.cost``, which the backend
    reports as ``usd`` so a run can stop at a spend cap. Other providers ignore the
    field and ``usd`` stays zero; set prices on the command line instead.
    """

    name = "openai_compatible"

    def __init__(
        self,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key_env: str = "OPENROUTER_API_KEY",
        max_tokens: int = 4000,
        temperature: float = 0.0,
    ):
        self._model = model
        self._base_url = base_url
        self._api_key_env = api_key_env
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = None

    @property
    def config(self) -> dict[str, Any]:
        """Model, endpoint and sampling settings. No secrets."""
        return {
            "model": self._model,
            "base_url": self._base_url,
            "api_key_env": self._api_key_env,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }

    def _get_client(self):
        if self._client is None:
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
            "usd": float(getattr(resp.usage, "cost", 0.0) or 0.0),
        }
        return AssistantTurn(content=choice.message.content or "", tool_calls=calls, usage=usage)


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


class TinkerBackend:
    """Sample from a base model or a LoRA checkpoint on Tinker, with tool calls parsed from text.

    Parameters
    ----------
    base_model : str
        For example ``"Qwen/Qwen3-8B"``.
    model_path : str or None
        A ``tinker://`` path to saved weights. None samples the base model.
    renderer : str
        Chat template name from ``tinker_cookbook.renderers``. ``qwen3_disable_thinking``
        gives fast, deterministic tool use; ``qwen3`` enables the thinking mode.
    max_tokens : int
    temperature : float

    Notes
    -----
    Tinker exposes sampling, not a chat-completions API, so this backend renders the
    conversation itself. The tool definitions go into the system prompt as JSON in the
    Qwen3 convention, and the model is asked to emit each call as
    ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``. Tool results are sent
    back as a user message wrapped in ``<tool_response>`` tags. The same text format is
    what the RL phase trains on, so evaluation and training see one convention.
    """

    name = "tinker"

    def __init__(
        self,
        base_model: str = "Qwen/Qwen3-8B",
        model_path: str | None = None,
        renderer: str = "qwen3_disable_thinking",
        max_tokens: int = 2000,
        temperature: float = 0.0,
    ):
        self._base_model = base_model
        self._model_path = model_path
        self._renderer_name = renderer
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._sampler = None
        self._renderer = None
        self._tokenizer = None

    @property
    def config(self) -> dict[str, Any]:
        """Base model, checkpoint path, renderer and sampling settings."""
        return {
            "base_model": self._base_model,
            "model_path": self._model_path,
            "renderer": self._renderer_name,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }

    def _setup(self):
        if self._sampler is None:
            import tinker
            from tinker_cookbook import renderers
            from tinker_cookbook.tokenizer_utils import get_tokenizer

            service = tinker.ServiceClient()
            if self._model_path:
                self._sampler = service.create_sampling_client(model_path=self._model_path)
            else:
                self._sampler = service.create_sampling_client(base_model=self._base_model)
            self._tokenizer = get_tokenizer(self._base_model)
            self._renderer = renderers.get_renderer(self._renderer_name, self._tokenizer)

    @staticmethod
    def render_messages(messages: list[dict], tools: list[dict]) -> list[dict]:
        """Convert OpenAI-format messages to plain text turns with the tool protocol inline.

        Returns
        -------
        list of dict
            ``system``, ``user`` and ``assistant`` messages with string content only.
        """
        tool_block = "\n".join(json.dumps(t["function"]) for t in tools)
        protocol = (
            "\n\n# Tools\n\nYou may call these functions. For each call, output exactly:\n"
            '<tool_call>\n{"name": "<function-name>", "arguments": {...}}\n</tool_call>\n'
            "Results come back inside <tool_response> tags.\n\n" + tool_block
        )
        out: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                out.append({"role": "system", "content": m["content"] + protocol})
            elif m["role"] == "assistant":
                text = m.get("content") or ""
                for tc in m.get("tool_calls", []):
                    fn = tc["function"]
                    args = (
                        fn["arguments"] if isinstance(fn["arguments"], dict) else json.loads(fn["arguments"])
                    )
                    text += (
                        "\n<tool_call>\n"
                        + json.dumps({"name": fn["name"], "arguments": args})
                        + "\n</tool_call>"
                    )
                out.append({"role": "assistant", "content": text})
            elif m["role"] == "tool":
                content = "<tool_response>\n" + m["content"] + "\n</tool_response>"
                if out and out[-1]["role"] == "user":
                    out[-1]["content"] += "\n" + content
                else:
                    out.append({"role": "user", "content": content})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    @staticmethod
    def parse_tool_calls(text: str) -> list[ToolCall]:
        """Extract ``<tool_call>`` JSON blocks from model text. Malformed JSON is skipped."""
        calls = []
        for raw in TOOL_CALL_RE.findall(text):
            try:
                d = json.loads(raw)
                calls.append(
                    ToolCall(
                        id=uuid.uuid4().hex[:8], name=str(d["name"]), arguments=dict(d.get("arguments", {}))
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        return calls

    def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        """Render, sample one sequence, parse tool calls from the text."""
        from tinker import types

        self._setup()
        prompt = self._renderer.build_generation_prompt(self.render_messages(messages, tools))
        params = types.SamplingParams(
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            stop=self._renderer.get_stop_sequences(),
        )
        result = self._sampler.sample(prompt=prompt, num_samples=1, sampling_params=params).result()
        tokens = list(result.sequences[0].tokens)
        text = self._tokenizer.decode(tokens, skip_special_tokens=True)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        usage = {"input_tokens": float(prompt.length), "output_tokens": float(len(tokens)), "usd": 0.0}
        return AssistantTurn(
            content=TOOL_CALL_RE.sub("", text).strip(), tool_calls=self.parse_tool_calls(text), usage=usage
        )
