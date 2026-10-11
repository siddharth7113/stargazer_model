# Star Gazer

Star Gazer is a benchmark and a reinforcement-learning environment for exoplanet
detection in TESS light curves. Every number it produces is reproducible.

Status: **tier 1 (detection) in progress**. [PLAN.md](PLAN.md) holds the phased plan.
[LOGBOOK.md](LOGBOOK.md) holds a dated record of each run. [STYLE.md](STYLE.md) holds the
writing rules for this repository.

## Purpose

Catalogues of known planets are incomplete. A reward that says "match the catalogue"
punishes a detector that finds a real planet the catalogue does not list.

Star Gazer uses injection and recovery instead. It injects synthetic transits with known
parameters into real TESS light curves of stars with no known signal. The truth is then
exact. The difficulty is adjustable. The number of episodes has no limit. The Kepler and
TESS teams use the same method to measure their own pipelines.

## Verifiability rules

1. **Seeds.** Each random choice uses an explicit integer seed. The output records the seed.
2. **Manifests.** Each data file has a SHA-256 in a manifest. `stargazer verify` recomputes
   the hashes and reports each mismatch.
3. **Separated truth.** An episode file holds only time, flux and flux error. It holds no
   TIC ID, no coordinates, no stellar parameters and no injection parameters. The truth
   file holds those. Only the scorer reads the truth file.
4. **Provenance in results.** Each results file records the git commit, the episode set
   hash, the full configuration and the score of each episode.
5. **Offline tests.** `pytest` checks the scorer, the injector and the baseline on
   synthetic data. The tests do not use the network.
6. **Logbook.** Each run that gives a number in a document has a LOGBOOK entry. The entry
   gives the exact command and the results file.

## Design

The package is a software development kit (SDK). The command line is a thin layer over it.
A training loop or a language-model harness imports the SDK directly.

| Object | Role |
|--------|------|
| `LightCurveCache` | Fetch and load TESS light curves |
| `select_targets` | Choose host stars with no TESS Object of Interest (TOI) |
| `EpisodeSet`, `BuildConfig` | Build and read an episode set |
| `Detector`, `Prediction` | The interface each detection system implements |
| `BLSDetector` | The Box Least Squares (BLS) baseline |
| `evaluate`, `Results` | Run a detector and write a results file |
| `score_episode` | The scoring rule. Use it directly as a reward |

Example:

```python
from stargazer import BLSDetector, EpisodeSet, evaluate

episodes = EpisodeSet("data/episodes/s98_v1")
results = evaluate(BLSDetector(sde_threshold=7.0), episodes)
results.save("results/my_run.json")
print(results.summary["completeness_exact"])
```

To add a detector, implement `detect(time, flux, flux_err) -> Prediction`, a `name` and
a `config` property. The `config` must fully determine the detector's behaviour.

## Layout

```
stargazer/
  types.py       Prediction and the Detector protocol
  data.py        LightCurveCache: download and cache TESS 2-minute light curves
  targets.py     select_targets: host stars with no TOI, with TIC parameters
  inject.py      batman transit injection and the truth record
  episodes.py    EpisodeSet and BuildConfig
  score.py       scoring rules and set-level metrics
  evaluate.py    evaluate and Results
  baseline.py    BLSDetector
  provenance.py  hashes, manifests, git commit, timestamps
  cli.py         command line
tests/           offline tests
data/            targets files and episode sets
results/         results files, one for each run
```

## Quick start

```bash
uv venv
uv pip install -e ".[dev]"
pre-commit install
pytest
stargazer targets  --sector 98 --n 60 --seed 1
stargazer fetch    --targets data/targets_s98.json
stargazer build    --targets data/targets_s98.json --n 200 --seed 1 --out data/episodes/s98_v1
stargazer baseline --episodes data/episodes/s98_v1
stargazer verify   data/episodes/s98_v1
```

## Task tiers

| Tier | Input | Output | Truth |
|------|-------|--------|-------|
| 1 Detection | one raw light curve | period, epoch, depth, duration, or "none" | injection |
| 2 Vetting | candidate, pixels, Gaia neighbours | planet, EB, blend or systematic | injection and ExoFOP |
| 3 Agentic | TIC ID and a toolbox | vetting report with a verdict | tiers 1 and 2 |

## Scoring, tier 1

The scorer gives period credit 1.0 at the true period. It gives 0.5 at one half or two
times the true period. It gives 0.25 at one third or three times the true period. A period
matches if the relative error is below `period_rtol`, default 1 percent. The epoch is
correct if the predicted epoch is within half a transit duration of a true transit centre.
The score is `period_credit * (0.7 + 0.3 * epoch_ok)`. A null episode scores 1.0 for
"none" and 0.0 for a false alarm. Set-level metrics are completeness as a function of
expected signal-to-noise ratio (SNR), completeness as a function of transit count, and
the false-alarm rate on null episodes. `stargazer/score.py` holds the exact rules.

## Data

The data are TESS Science Processing Operations Center (SPOC) 2-minute cadence PDCSAP
light curves from the Mikulski Archive for Space Telescopes (MAST). `select_targets`
takes the 2-minute target list of a sector, applies brightness and dwarf cuts from the
TESS Input Catalog (TIC), and removes each star in the ExoFOP TOI list. The targets file
records the TOI CSV hash and download time.

## License

The code is MIT. TESS data are public domain, NASA.
