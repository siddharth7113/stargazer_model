# Logbook

Newest entry at the bottom. Each entry has: date, what was done, exact command, evidence.

## 2026-10-11

**Context.** Project started after reading about an independent researcher's AI-assisted
candidate around TIC 4206066 (3.18 day, about 500 ppm, Zenodo preprint, not confirmed).
Decision: build a general injection-recovery benchmark rather than chase one result.

**Environment.** Linux, Python 3.13.9 via uv venv, RTX 4050 6 GB, 12 cores, 15 GB RAM.
MAST and PyPI reachable.

**Setup.**
- `git init`, `pyproject.toml` with dependencies, no lock file (owner's request).
- `uv pip install -e ".[dev]"`. batman 2.5.1 needs `distutils`, which Python 3.13 removed,
  so `setuptools` was added as a runtime dependency. Verified imports:
  lightkurve 2.6.0, batman 2.5.1, astropy 8.0.1, numpy 2.5.4.

**Target selection, sector 98.**
```
stargazer targets --sector 98 --n 60 --seed 1
```
Output: `data/targets_s98.json`. 13,000 stars have SPOC 2-minute data in sector 98.
435 of them have a TOI and were excluded. From a seeded draw of 240, the cuts rejected
138 for Tmag, 28 for Teff and 33 for radius. 41 of the 60 requested passed. The run
took 4.5 minutes, mostly the MAST observation query. The targets file records the TOI
CSV SHA-256 and download time.

**Fetch.**
```
stargazer fetch --targets data/targets_s98.json
```
41 cached, 0 failed, about 40 seconds for each download. `stargazer verify data/cache`
passes. Log: `results/fetch_s98.log` (not in git).

**Design changes during the day.** Recorded here because they change what a reader
sees in the code.
- SDK first: `Detector` protocol, `EpisodeSet`, `evaluate`, `Results`. The CLI is a thin layer.
- Absolute imports only, enforced by ruff TID252.
- Equations live in numpydoc Notes sections with units and sources.
- wotan removed. Detrending uses `lightkurve.LightCurve.flatten` (scipy Savitzky-Golay).
  Reason: wotan has no recent release; lightkurve is a standard, maintained dependency.
- Provenance is three classes: `Digest`, `Manifest`, `GitState`. Hashing uses
  `hashlib.file_digest` from the standard library. `GitState` exists so a results file
  names the exact code behind its numbers; a version string cannot do that between
  releases.
- Limb darkening is a replaceable table with a binary search, not an `if` chain.

**Episode set s98_v1, first build, discarded.** Built from all 41 hosts at 100 ppm to
1 percent depth. The baseline run crashed at episode 88 with a NaN light curve. Cause:
the TESS Input Catalog has no radius or surface gravity for TIC 298957105 and TIC
150391195. The catalogue cell parser let NaN through as a number, the cuts cannot
reject NaN, and the injection computed a NaN orbit. Fixes: the parser returns None for a
non-finite cell; `select_targets` requires finite `logg`; `make_truth` raises on a
non-finite stellar parameter instead of substituting a solar value; `EpisodeSet.build`
skips such hosts and records them in `config.json`. Also: `evaluate` now records a
detector exception as a prediction of "none" with the error text, so one bad episode
cannot lose a run.

**Episode set s98_v1.**
```
stargazer build --targets data/targets_s98.json --n 200 --seed 1 --out data/episodes/s98_v1
stargazer verify data/episodes/s98_v1
```
Episode set hash `1ed1abbb5f8656816aab907021837eddba7d883acb3bb9978b02c23590779eb4`.
39 hosts, 2 skipped for missing parameters. Depth range 100 ppm to 1 percent. Composition:
57 null, 1 injected with no transit in the data (scored as null), 142 with at least one
transit. Expected SNR percentiles 10/25/50/75/90: 1.7, 4.7, 19.2, 80.5, 189. Every episode
file is finite. Size 62 MB, not in git; the hash is the reference. Note: sector 98 light
curves span 57 days, not 27. TESS now observes longer sectors. The injection period
range 0.5 to 20 days therefore gives at least two transits in nearly every episode.
A later set should extend the period range to 60 days to exercise single transits.

**Agent design.** The language-model detector is provider-agnostic. `ToolAgentDetector`
owns the loop. A backend implements one method, `complete(messages, tools)`.
`OpenAICompatibleBackend` covers OpenRouter, vLLM and Ollama. `TinkerBackend` samples
from a base model or a `tinker://` checkpoint and parses `<tool_call>` JSON from text in
the Qwen3 convention, the same format the RL phase will train on. Tools are defined once
in `stargazer.tools.TOOLS` with JSON Schemas; the loop validates each call, including
positive-number constraints, and returns an error to the model instead of raising. The
agent has not yet been run against a paid model; no key is set in this environment.
Offline tests cover the loop with a scripted backend, the schemas, and the Tinker
rendering and parsing.

**Baseline run.**
```
stargazer baseline --episodes data/episodes/s98_v1 --out results/baseline_bls_s98_v1_sde7.json
```
Default config: Savitzky-Golay window 0.75 days, 10-minute bins, periods 0.5 days to
half the baseline, SDE threshold 7. About 3 seconds for each episode, 200 episodes, no
detector errors. Results: `results/baseline_bls_s98_v1_sde7.json`, episode set hash
`1ed1abbb...`, commit `6a2814a`.

At the default threshold of 7: false-alarm rate 0.69, completeness 0.42 exact, 0.43 with
aliases, mean score 0.39. The textbook threshold does not transfer to real 57-day light
curves. The SDE of a null episode is 9.5 at the median, 14.0 at the 95th percentile and
20.8 at the maximum, because stellar variability leaves structure in the power spectrum.

**Threshold sweep.**
```
stargazer calibrate --results results/baseline_bls_s98_v1_sde7.json \
    --thresholds 5,7,9,11,13,15,18,21,25,30,40 --max-false-alarm 0.05 \
    --out results/calibrate_bls_s98_v1.json
```

| SDE threshold | false-alarm rate | completeness exact | mean score |
|---|---|---|---|
| 7 | 0.690 | 0.423 | 0.393 |
| 11 | 0.276 | 0.408 | 0.502 |
| 13 | 0.069 | 0.387 | 0.545 |
| 15 | 0.052 | 0.345 | 0.520 |
| 18.63 | 0.034 | 0.317 | 0.505 |
| 21 | 0.000 | 0.239 | 0.460 |

The lowest threshold with a false-alarm rate at or below 0.05 on this set is 18.63.
Decision: the reported baseline number for s98_v1 is completeness 0.32 at threshold
18.63 and false-alarm rate 0.03. The mean score peaks at threshold 13, but that
threshold has a 7 percent false-alarm rate, above the 5 percent acceptance criterion.

Completeness by expected SNR at threshold 7 (the by-bin numbers at 18.63 are in the
calibrate file): 0-5: 0.00 (38 episodes); 5-7: 0.00 (6); 7-10: 0.22 (9); 10-20: 0.48 (21);
20-50: 0.48 (25); above 50: 0.86 (43). By transit count: one transit 0.20 (5), two 0.00
(3), three to five 0.31 (26), six or more 0.48 (108).

**Reading.** The baseline is weak exactly where a reasoning system can gain: SNR 7 to
50 with few transits. Nearly every episode has six or more transits because sector 98 is
57 days long and the period range stops at 20 days. Set s98_v2 should extend periods to
60 days so single and double transits are common.
