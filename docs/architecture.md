# Architecture

Star Gazer is a Python library with a thin command line over it. The command line
calls the same functions that a training loop or a language-model harness imports.

## Components

```{mermaid}
classDiagram
    class Detector {
        <<protocol>>
        +str name
        +dict config
        +detect(time, flux, flux_err) Prediction
    }
    class Prediction {
        +bool detected
        +float period
        +float t0
        +float duration
        +float depth
        +dict extra
        +to_dict() dict
        +from_dict(d) Prediction
    }
    class BLSDetector {
        +str name = "baseline_bls"
        +dict config
        +detect(time, flux, flux_err) Prediction
    }
    class LightCurveCache {
        +Path root
        +path(tic, sector) Path
        +has(tic, sector) bool
        +load(tic, sector) LightCurve
        +fetch(tic, sector) Path
    }
    class LightCurve {
        +ndarray time
        +ndarray flux
        +ndarray flux_err
        +dict meta
    }
    class BuildConfig {
        +int n
        +int seed
        +float null_frac
        +tuple period_range
        +tuple depth_range
    }
    class EpisodeSet {
        +Path root
        +list ids
        +dict truth
        +dict config
        +str hash
        +load(ep_id) tuple
        +verify() list
        +build(targets, cache, out_dir, cfg) EpisodeSet$
    }
    class TransitTruth {
        +float period
        +float t0
        +float rp_rs
        +float a_rs
        +float inc_deg
        +float b
        +float duration
        +float depth
    }
    class Results {
        +str system
        +dict config
        +str episode_set_hash
        +dict preds
        +dict rows
        +dict summary
        +save(path) Path
        +load(path) Results$
    }
    Detector <|.. BLSDetector
    Detector ..> Prediction : returns
    LightCurveCache ..> LightCurve : loads
    EpisodeSet ..> LightCurveCache : reads
    EpisodeSet ..> BuildConfig : uses
    EpisodeSet ..> TransitTruth : injects
    Results ..> Detector : records config
    Results ..> EpisodeSet : records hash
```

## Data flow

```{mermaid}
flowchart LR
    MAST[(MAST)] -->|SPOC 120 s| cache[LightCurveCache]
    TIC[(TIC)] --> targets[select_targets]
    TOI[(ExoFOP TOI)] -->|exclude| targets
    targets --> tfile[targets_sNN.json]
    tfile --> cache
    cache --> build[EpisodeSet.build]
    tfile --> build
    build --> ep[episodes/*.npz]
    build --> truth[truth.json]
    build --> manifest[manifest.json]
    ep --> det[Detector.detect]
    det --> pred[Prediction]
    pred --> score[score_episode]
    truth --> score
    score --> res[Results]
    manifest -->|hash| res
    res --> file[results/*.json]
```

A detector reads only the episode file. The scorer reads the prediction and the truth
record.

## An evaluation run

```{mermaid}
sequenceDiagram
    participant U as User or training loop
    participant E as evaluate
    participant S as EpisodeSet
    participant D as Detector
    participant C as score_set
    U->>E: evaluate(detector, episodes)
    E->>S: truth, ids
    loop each episode
        E->>S: load(ep_id)
        S-->>E: time, flux, flux_err
        E->>D: detect(time, flux, flux_err)
        D-->>E: Prediction
    end
    E->>C: score_set(preds, truth)
    C-->>E: rows, summary
    E->>S: hash
    E-->>U: Results
    U->>U: results.save(path)
```

## The agent

A language model is run through :class:`stargazer.agent.ToolAgentDetector`, which
holds the system prompt, the tool definitions and the loop. A backend implements one
method, ``complete(messages, tools)``, that returns the next assistant turn.

```{mermaid}
sequenceDiagram
    participant A as ToolAgentDetector
    participant B as ChatBackend
    participant T as tools.call_tool
    A->>B: complete(system + user, tool definitions)
    B-->>A: AssistantTurn(tool_calls=[bls_peaks])
    A->>T: bls_peaks(time, flux)
    T-->>A: {peaks: [...]}
    A->>B: complete(... + tool result)
    B-->>A: AssistantTurn(tool_calls=[fold, odd_even])
    A->>T: fold, odd_even
    T-->>A: results
    A->>B: complete(...)
    B-->>A: AssistantTurn(tool_calls=[submit_prediction])
    A-->>A: Prediction + transcript
```

Two backends exist. ``OpenAICompatibleBackend`` speaks the chat-completions API with
function calling, which OpenRouter, vLLM and Ollama accept. ``TinkerBackend`` samples
from a base model or a trained checkpoint on Tinker. It declares the tools and parses
replies through the cookbook renderer of the model family, so Qwen, Nemotron and
Inkling each see their own native tool format. The tool definitions live once, in
``stargazer.tools.TOOLS``, with a JSON Schema for each, and the loop validates every
call before it runs.

The tools are ``summarize``, ``bls_peaks`` (with optional masks of rejected signals so a
strong feature cannot hide a weaker periodic one), ``single_events`` (the strongest
individual dips, for planets with one or two transits in the data), ``fold``,
``odd_even`` and ``submit_prediction``. Each tool result carries ``calls_remaining``. A
turn without a tool call gets a reminder before the episode ends.

## The reinforcement-learning view

In tier 3 the same objects form an environment. `EpisodeSet` supplies the episode on
`reset`, the functions in `tools.py` are the actions, `score_episode` is the reward and
`Results` is the evaluation log. The training loop owns the policy and the optimiser;
the package supplies the data, the tools and the reward.
