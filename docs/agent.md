# Language-model agent

{class}`~stargazer.agent.ToolAgentDetector` runs a chat model on one episode. The
model sees a system prompt with the scoring rules, calls analysis tools on the light
curve and ends the episode with `submit_prediction`.

## Loop

```{mermaid}
sequenceDiagram
    participant A as ToolAgentDetector
    participant B as ChatBackend
    participant T as tools.call_tool
    A->>B: complete(system + user, tool definitions)
    B-->>A: AssistantTurn(tool_calls=[bls_peaks])
    A->>T: bls_peaks(time, flux)
    T-->>A: peaks
    A->>B: complete(... + tool result)
    B-->>A: AssistantTurn(tool_calls=[fold, odd_even])
    A->>T: fold, odd_even
    T-->>A: results
    A->>B: complete(...)
    B-->>A: AssistantTurn(tool_calls=[submit_prediction])
    A-->>A: Prediction and trajectory
```

On each turn:

1. The backend returns one assistant turn with text, optional reasoning and tool calls.
2. Each tool call is validated against the JSON Schema of the tool and run on the
   episode arrays. An invalid call returns an error with the expected arguments, and
   the episode continues.
3. Each tool result carries `calls_remaining`.
4. A turn without a tool call receives a reminder; after two reminders the episode
   ends with "none".
5. A valid `submit_prediction` ends the episode.

## Tools

| Tool | Arguments | Returns |
|---|---|---|
| `summarize` | none | cadence, point scatter, flux range, gaps |
| `bls_peaks` | `min_period`, `max_period`, `detrend_window_days`, `masks` | the five strongest distinct BLS peaks with SDE, depth, duration, alias periods and event counts |
| `search_signals` | `min_period`, `max_period`, `detrend_window_days`, `min_sde`, `max_signals` | the signals found in turn by search, mask and search again, with the residual SDE and the stop reason |
| `single_events` | `detrend_window_days` | the strongest individual dips, ranked by the single-event statistic, with the noise at each event's timescale |
| `fold` | `period`, `t0`, `phase_half_width`, `detrend_window_days` | a 40-bin phase curve and its depth |
| `fit_transit` | `period`, `t0`, `duration`, `detrend_window_days` | a batman fit: depth, duration, impact parameter, host density with a verdict, shape, significance |
| `odd_even` | `period`, `t0`, `duration` | odd and even depths, their difference in sigma, and a verdict |
| `secondary_eclipse` | `period`, `t0`, `duration`, `detrend_window_days` | the depth at phase 0.5 and the strongest other dip, in sigma, with a verdict |
| `ephemeris_match` | `period`, `t0` | the star's TOI and eclipsing-binary entries that match the period or an alias, the epoch offset, the Gaia class, with a verdict; needs the episode's context |
| `nearby_stars` | none | Gaia sources within two pixels with their flux fraction and the depth each could make, with a verdict; needs the episode's context |
| `submit_prediction` | `detected`, `period`, `t0`, `duration`, `depth`, `reasoning` | ends the episode |

`masks` takes a list of `{period, t0, duration}` objects. The masked points are removed
before the search, so a weaker periodic signal can appear once a stronger one is
removed.

The single-event statistic of a box of width $d$ is

$$
S = \frac{\tilde{F}_d - \bar{F}_\mathrm{box}}{\sigma_d}
$$

where $\tilde{F}_d$ is the median of all box means, $\bar{F}_\mathrm{box}$ the mean
flux in the box, and $\sigma_d$ is 1.4826 times the median absolute deviation of the
box means. The odd-even difference is

$$
\Delta_\mathrm{oe} = \frac{|\delta_\mathrm{odd} - \delta_\mathrm{even}|}{\sqrt{\epsilon_\mathrm{odd}^2 + \epsilon_\mathrm{even}^2}}
$$

with each standard error $\epsilon$ taken from the scatter between individual events,
which includes residual stellar variability.

## Submission forms

| Form | Arguments | Meaning |
|---|---|---|
| null | `detected=false` | no transit |
| periodic | `detected=true`, `period`, `t0` | a periodic transit |
| single event | `detected=true`, `t0`, `duration`, `depth`, no `period` | one transit-like dip |

The single-event form is for a planet with one transit in the data, which has no
measurable period. See {doc}`scoring` for the credit of each form.

## Backends

A backend implements one method, `complete(messages, tools)`, that returns an
{class}`~stargazer.agent.AssistantTurn`.

{class}`~stargazer.backends.OpenAICompatibleBackend`
: The chat-completions API with function calling. Works with OpenRouter (the default
  base URL), vLLM and Ollama. Needs the `openai` extra.

{class}`~stargazer.backends.TinkerBackend`
: Samples from a base model or a trained checkpoint on Tinker. It formats the prompt
  and parses tool calls with the Tinker cookbook renderer of the model family, so each
  family uses its native tool format. Needs the `tinker` extra.

Renderers used so far:

| Model | Renderer |
|---|---|
| `Qwen/Qwen3.8-27B` | `qwen3_8_disable_thinking`, `qwen3_8_xhigh_reasoning` |
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | `nemotron3_disable_thinking` |
| `thinkingmachines/Inkling-Small` | `tml_v0` |

## Sampling

The defaults follow the Qwen3 model cards {cite:p}`qwen2025`:

| Mode | Temperature | top-p | top-k | Output tokens per turn |
|---|---|---|---|---|
| non-thinking | 0.7 | 0.8 | 20 | 2048 |
| thinking | 0.6 | 0.95 | 20 | 8192 |

Temperature 0 is not used: the model cards discourage greedy decoding, and it does not
make a hosted model deterministic. The seed is recorded in the results file. To
measure run-to-run variance, repeat a run with different `--seed` values.

## Limits

The limits stop a model that loops. They are not part of the task.

| Limit | Default | Stop reason |
|---|---|---|
| tool calls | 40 | `max_tool_calls` |
| turns | 30 | `max_turns` |
| input plus output tokens | 400 000 | `max_tokens` |
| reminders after a turn without a tool call | 2 | `model_stopped` |

## Cost

When the backend knows a price, each episode records its cost in `usage.usd`. The
Tinker backend estimates it from a price table per model. When the price is unknown,
the key is absent.
