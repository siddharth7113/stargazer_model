# Overview

## Method

Star Gazer evaluates detectors by injection and recovery, the method used to measure
the completeness of the Kepler and TESS pipelines. A synthetic transit with known
parameters is multiplied into a real TESS light curve of a star with no detected
planet, and the detector is asked to find it. A fraction of the light curves receives
no transit, which gives the false-alarm rate. The truth of every episode is the
injected signal, so a detector earns credit for a planet that no catalogue lists. The
period and depth ranges chosen at build time fix the difficulty of a set.

## Task tiers

Tier 1, detection
: The input is one raw light curve. The output is a period, an epoch, a depth and a
  duration, or "none". The truth comes from injection.

Tier 2, vetting
: The input is a candidate, the target pixels and the Gaia neighbours. The output is a
  class: planet, eclipsing binary, blend or systematic. The truth comes from injection
  and from ExoFOP dispositions.

Tier 3, agentic
: The input is a masked star and a toolbox. The output is a vetting report with a
  verdict. The reward combines tiers 1 and 2. This tier is the reinforcement-learning
  environment.

## Status

Tier 1 is in progress. The checklist is in {doc}`plan` and each run is recorded in
{doc}`logbook`.
