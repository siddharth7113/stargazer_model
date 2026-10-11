# Star Gazer

Star Gazer is a Python package for measuring how well a language model finds
transiting planets in TESS light curves. It builds test sets by injecting synthetic
transits into real light curves, runs a detector on each light curve, scores the
predictions against the injected truth and records the provenance of every run.

The package contains:

- an episode-set builder that selects host stars, downloads their light curves and
  injects transits with the batman model;
- a scorer with credit for period aliases and for single-transit answers;
- a classical Box Least Squares (BLS) detector as a baseline;
- analysis tools that a language model calls through a JSON Schema interface;
- an agent loop with backends for OpenAI-compatible servers and for Tinker;
- a terminal viewer for the saved trajectories.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Installation
:link: install
:link-type: doc
Requirements, optional extras and API keys.
:::

:::{grid-item-card} Quickstart
:link: quickstart
:link-type: doc
From an episode set to a comparison table in seven commands.
:::

:::{grid-item-card} User guide
:link: concepts
:link-type: doc
Episode sets, the baseline, the agent, scoring and reproducibility.
:::

:::{grid-item-card} Command line
:link: cli
:link-type: doc
Every `stargazer` command and its options.
:::

:::{grid-item-card} API reference
:link: api
:link-type: doc
Every module, class and function, with equations and references.
:::

:::{grid-item-card} Logbook
:link: logbook
:link-type: doc
Every run so far, with the command that produced it.
:::
::::

```{toctree}
:maxdepth: 1
:caption: Getting started
:hidden:

install
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User guide
:hidden:

concepts
episode_sets
baseline
agent
scoring
trajectories
reproducibility
```

```{toctree}
:maxdepth: 1
:caption: Reference
:hidden:

cli
api
architecture
references
```

```{toctree}
:maxdepth: 1
:caption: Project
:hidden:

plan
logbook
```
