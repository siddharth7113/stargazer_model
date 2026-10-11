# Overview

## The problem

Catalogues of known planets are incomplete. A detector that uses "match the catalogue"
as its reward is punished when it finds a real planet the catalogue does not list. A
benchmark for discovery needs a truth source that does not depend on past discoveries.

## The method

Star Gazer uses injection and recovery. It takes a real TESS light curve of a star with
no known signal and injects a synthetic transit with known parameters. The detector
then has to find the transit. The truth is therefore exact. The build configuration
sets the difficulty, and the number of episodes has no limit.

## The tiers

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

## The status

Tier 1 is in progress. See {doc}`plan` for the checklist and {doc}`logbook` for each run.
