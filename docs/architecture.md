# Architecture

Star Gazer is a Python library with a thin command line over it. The command line
calls the same functions that a training loop or a language-model harness imports.

## Class diagram

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

## Module map

```
stargazer/
  types.py       Prediction and the Detector protocol
  data.py        LightCurveCache: download and cache TESS 2-minute light curves
  targets.py     select_targets: host stars with no TOI, with TIC parameters
  inject.py      batman transit injection and the truth record
  episodes.py    EpisodeSet and BuildConfig
  score.py       scoring rules and set-level metrics
  evaluate.py    evaluate and Results, with a worker pool
  baseline.py    BLSDetector
  tools.py       analysis tools with JSON Schemas, one registry for every backend
  agent.py       ToolAgentDetector and the ChatBackend protocol
  backends.py    OpenAICompatibleBackend (OpenRouter, vLLM, Ollama) and TinkerBackend
  calibrate.py   threshold choice from recorded statistics
  compare.py     one table for several results files
  viewer.py      terminal trajectory viewer
  provenance.py  hashes, manifests, git commit
  cli.py         command line
tests/           one unit-test file for each module; tests/integration/ for cross-module checks
scripts/         pre-commit lint hooks
docs/            Sphinx documentation with Mermaid diagrams
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

## Evaluation sequence

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

## Agent

The agent loop, the tools and the backends are described in {doc}`agent`.

## Reinforcement-learning environment

In tier 3 the same objects form an environment:

| Environment part | Star Gazer object |
|---|---|
| reset | {meth}`~stargazer.episodes.EpisodeSet.load` of the next episode |
| actions | the functions in {mod}`stargazer.tools` |
| reward | {func}`~stargazer.score.score_episode` |
| evaluation log | {class}`~stargazer.evaluate.Results` |

The training loop owns the policy and the optimiser.
