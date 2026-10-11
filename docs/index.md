# Star Gazer

Star Gazer is a benchmark and a reinforcement-learning environment for exoplanet
detection in light curves. A language model uses the classical tools of an astronomer,
and the benchmark scores its answers against an exact answer key. Every number that the
benchmark produces is reproducible.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Overview
:link: overview
:link-type: doc
The problem, the injection-and-recovery method and the task tiers.
:::

:::{grid-item-card} Architecture
:link: architecture
:link-type: doc
The software development kit (SDK), the agent loop and the backends, with class, flow and sequence diagrams.
:::

:::{grid-item-card} Scoring
:link: scoring
:link-type: doc
The scoring rules, the alias credit and the threshold choice from recorded data.
:::

:::{grid-item-card} Usage
:link: usage
:link-type: doc
Commands to build an episode set, run the baseline and an agent, read a trajectory and compare systems.
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
api
references
```
