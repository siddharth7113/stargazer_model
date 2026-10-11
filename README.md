# Star Gazer

A verifiable benchmark and reinforcement-learning environment for finding exoplanets in
TESS light curves. The goal is a public, reproducible test of how well any system, from a
plain Box Least Squares search to a tool-using language-model agent, can detect and vet
transit signals, and then to point the best system at unlabelled data and find things.

Status: **tier 1 (detection) under construction**. See [PLAN.md](PLAN.md) for the phased plan
and [LOGBOOK.md](LOGBOOK.md) for a dated record of every run.

## Why this exists

Catalogues of known planets are incomplete and biased, so "match the catalogue" is a bad
reward. Star Gazer instead uses **injection and recovery**: synthetic transits with known
parameters are injected into real TESS light curves of stars with no known signal. Ground
truth is then exact, difficulty is controllable, and the number of episodes is unlimited.
This is the same method the Kepler and TESS teams use to measure their own pipelines'
completeness.

## Verifiability rules

Every number in this repo must be reproducible by someone else. Concretely:

1. **Seeds.** Every random choice takes an explicit integer seed recorded in the output.
2. **Manifests.** Every downloaded or generated data file has its SHA-256 recorded in a
   manifest committed to git. `stargazer verify` recomputes and compares them.
3. **Truth is separated.** The agent-facing episode files contain no injection parameters,
   no TIC ID and no coordinates. Truth lives in a separate file the scorer reads.
4. **Results carry provenance.** Every results JSON records the git commit, the episode-set
   hash, the full config, and per-episode scores, not just a summary.
5. **Tests.** `pytest` covers the scorer, the injector and the baseline on synthetic data
   with no network access, so the core can be checked offline.
6. **Logbook.** Every run that produced a number in a document has a LOGBOOK entry with
   the exact command and the results file it wrote.

## Layout

```
stargazer/
  data.py       download and cache TESS 2-minute light curves, write manifests
  targets.py    select stars with no known signal (TOI cross-match), stellar params
  inject.py     batman transit injection, EB-like false-positive injection
  episodes.py   build an episode set: agent-facing files + separate truth file
  score.py      detection scorer with alias handling, episode and set-level metrics
  baseline.py   BLS baseline detector (detrend -> BLS -> threshold)
  cli.py        command line entry point
tests/          offline unit tests
data/episodes/  generated episode sets (small ones are committed)
results/        results JSON files, one per run
PLAN.md         phases, checklists, acceptance criteria
LOGBOOK.md      dated record of runs and decisions
```

## Quick start

```bash
uv venv && uv pip install -e ".[dev]"
pytest                                  # offline checks
stargazer targets  --sector 98 --n 60   # pick candidate host stars, cross-match TOIs
stargazer fetch    --targets data/targets_s98.json
stargazer build    --targets data/targets_s98.json --n 200 --seed 1 --out data/episodes/s98_v1
stargazer baseline --episodes data/episodes/s98_v1
stargazer verify   --episodes data/episodes/s98_v1
```

## Task tiers

| Tier | Input | Output | Truth source |
|------|-------|--------|--------------|
| 1 Detection | one raw light curve | period, epoch, depth, duration, or "none" | injection |
| 2 Vetting | candidate + pixels + Gaia neighbours | planet / EB / blend / systematic | injection + ExoFOP dispositions |
| 3 Agentic | TIC ID and a toolbox | vetting report with verdict | composite of 1 and 2 |

## Scoring (tier 1)

An episode is scored 1.0 when the predicted period matches the injected period within a
relative tolerance, counting the 1/2, 2, 1/3 and 3 aliases as partial credit, and the
predicted epoch lands within half a transit duration of a true transit. Null episodes score
1.0 for a correct "none" and 0.0 for a false alarm. Set-level metrics are completeness as a
function of injected signal-to-noise and number of transits, and the false-alarm rate on
null episodes. See `stargazer/score.py` for the exact rules.

## Data

TESS SPOC 2-minute cadence PDCSAP light curves from MAST, downloaded with lightkurve.
Host stars are chosen from a sector's 2-minute target list and cross-matched against the
ExoFOP TOI catalogue; any star with a TOI is excluded. The TOI CSV used, with its SHA-256
and download date, is recorded in the targets file.

## License

MIT for the code. TESS data is public domain, NASA.
