# Concepts

## Terms

The documentation and the code use these terms with one meaning each.

light curve
: A time series of the brightness of one star. Star Gazer uses TESS 2-minute data as
  three arrays: `time` in days, `flux` normalised to a median of one, and `flux_err`.

transit
: The periodic dip in a light curve when a planet passes in front of its star. A
  transit has a period $P$, an epoch $t_0$ (the centre time of one transit), a depth
  $\delta$ and a duration $T_{14}$.

injection
: The multiplication of a synthetic transit model into a real light curve.

episode
: One light curve given to a detector, with or without an injected transit. An
  episode without a transit is a *null* episode.

episode set
: A directory of episodes built from one configuration, with a truth file, a
  configuration file and a manifest of file hashes.

truth
: The record of what was injected into an episode. Only the scorer reads it.

detector
: Any object with a method `detect(time, flux, flux_err)` that returns a
  {class}`~stargazer.types.Prediction`.

prediction
: The claim of a detector about one episode: detected or not, and if detected, a
  period and an epoch, or a single event time.

trajectory
: For an agent, the full record of one episode: the reasoning, each tool call, each
  tool result and the submission.

## Injection and recovery

Star Gazer evaluates detectors by injection and recovery, the method used to measure
the completeness of the Kepler and TESS pipelines {cite:p}`christiansen2013`:

1. Take a real light curve of a star with no detected planet.
2. Multiply a synthetic transit with known parameters into it, or leave it unchanged
   for a null episode.
3. Give the result to the detector.
4. Compare the prediction with the injected parameters.

Because the truth is the injected signal, a detector earns credit for a planet that no
catalogue lists. The period and depth ranges chosen at build time set the difficulty of
a set.

## Task tiers

| Tier | Input | Output | Truth |
|---|---|---|---|
| 1, detection | one light curve | period, epoch, depth and duration, a single event, or "none" | injection |
| 2, vetting | a candidate, target pixels, Gaia neighbours | planet, eclipsing binary, blend or systematic | injection and ExoFOP dispositions |
| 3, agentic | a star and a toolbox | a vetting report with a verdict | tiers 1 and 2 combined |

Tier 3 is the reinforcement-learning environment. Tier 1 is implemented; tiers 2 and 3
are planned. The checklist is in {doc}`plan`.

## Components

```{mermaid}
flowchart LR
    build[EpisodeSet.build] --> ep[episode files]
    build --> truth[truth.json]
    ep --> det[Detector]
    det --> pred[Prediction]
    pred --> score[score_episode]
    truth --> score
    score --> res[Results]
```

- {doc}`episode_sets` describes how the episode files and the truth are made.
- {doc}`baseline` and {doc}`agent` describe the two detectors in the package.
- {doc}`scoring` describes how a prediction is compared with the truth.
- {doc}`reproducibility` describes what a results file records.
