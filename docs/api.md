# API reference

The public names are importable from the top-level package, for example
`from stargazer import EpisodeSet, evaluate`. Each module page below documents every
public class and function, with the equations in the Notes sections and the sources in
the References sections.

| Module | Contents |
|---|---|
| {doc}`api/types` | `Prediction` and the `Detector` protocol |
| {doc}`api/data` | download and cache of TESS light curves |
| {doc}`api/targets` | host-star selection |
| {doc}`api/inject` | transit geometry, injection and the truth record |
| {doc}`api/episodes` | episode-set build, load and verification |
| {doc}`api/score` | scoring rules and set metrics |
| {doc}`api/evaluate` | evaluation runs and results files |
| {doc}`api/calibrate` | threshold choice from null episodes |
| {doc}`api/baseline` | the Box Least Squares detector |
| {doc}`api/tools` | analysis tools for the agent |
| {doc}`api/agent` | the agent loop and the backend protocol |
| {doc}`api/backends` | OpenAI-compatible and Tinker backends |
| {doc}`api/compare` | comparison tables |
| {doc}`api/viewer` | the terminal trajectory viewer |
| {doc}`api/provenance` | hashes, manifests and git state |

```{toctree}
:hidden:
:glob:

api/*
```
