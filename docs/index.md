# Star Gazer

Star Gazer is a benchmark and a reinforcement-learning environment for exoplanet
detection in light curves. The model acts as an astronomer with the field's classical
tools; the benchmark says how well it does. Every number it produces is reproducible.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Overview
:link: overview
:link-type: doc
The problem, the injection-and-recovery method, and the three task tiers.
:::

:::{grid-item-card} Architecture
:link: architecture
:link-type: doc
The SDK, the agent loop, the backends, and the diagrams that tie them together.
:::

:::{grid-item-card} Scoring
:link: scoring
:link-type: doc
The exact rules, the alias credit, and how a threshold is chosen from data.
:::

:::{grid-item-card} Usage
:link: usage
:link-type: doc
Build a set, run the baseline, run an agent, read a trajectory, compare systems.
:::

:::{grid-item-card} Logbook
:link: logbook
:link-type: doc
Every number produced so far, with the command that produced it.
:::

:::{grid-item-card} API reference
:link: api
:link-type: doc
Every module, class and function, with equations and references.
:::
::::

```{toctree}
:maxdepth: 2
:caption: Guide

overview
architecture
data
scoring
verifiability
usage
```

```{toctree}
:maxdepth: 1
:caption: Project

plan
logbook
style
api
references
```
