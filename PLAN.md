# Plan

Tick a checkbox only when two conditions are true. The acceptance criterion is met. The
LOGBOOK has an entry that points to the evidence: a results file, a test run or a commit.

## Phase 0: Scaffold

- [x] Git repository, `pyproject.toml` only, no lock file
- [x] README, PLAN, LOGBOOK, STYLE
- [x] SDK design: `Detector` protocol, `EpisodeSet`, `evaluate`, `Results`; thin CLI
- [x] ruff with the numpy docstring convention, pre-commit hooks installed
- [x] Docstrings and documents follow STYLE.md
- [x] `pytest` passes offline

Acceptance: a fresh clone with `uv pip install -e ".[dev]"` and `pytest` passes.
Evidence: LOGBOOK 2026-10-11.

## Phase 1: Tier 1 detection benchmark

### 1a Data
- [x] `select_targets` reads the 2-minute target list of a sector from MAST
- [x] The TOI cross-match removes each star with a TOI. The targets file records the TOI CSV hash and date
- [x] The targets file records radius, Teff, logg, Tmag and mass from the TIC
- [x] `LightCurveCache.fetch` caches light curves as `.npz`. A manifest records each SHA-256
- [x] At least 40 host stars cached for one sector (41 in sector 98)

Acceptance: `stargazer verify data/cache` passes. The targets file lists the TOI CSV hash.

### 1b Injection
- [x] batman injection: period, epoch, rp/rs, a/rs from stellar density, impact parameter
- [x] Test: the depth at mid-transit matches rp/rs squared within the limb-darkening tolerance
- [x] Test: the same seed gives identical episode files
- [x] The truth record stores the expected SNR and the transit count of each episode
- [x] Episode mix: 30 percent null, 70 percent injected. Periods 0.5 to 20 days. Depths 100 ppm to 1 percent
- [x] An injected episode with no transit in the data is scored as null
- [ ] One small episode set (20 episodes) committed to git
- [ ] Set s98_v2 with periods to 60 days, so single and double transits are common

Acceptance: `tests/test_inject.py` and `tests/test_episodes.py` pass.

### 1c Scorer
- [x] Period credit with aliases 1/2, 2, 1/3 and 3 at partial credit
- [x] Epoch check modulo the true period, within half a duration
- [x] Null handling: "none" on a null episode scores 1.0, a false alarm scores 0.0, a miss scores 0.0
- [x] Set-level metrics: completeness by SNR bin, completeness by transit count, false-alarm rate
- [x] The results file records the git commit, the episode set hash, the config and each episode row

Acceptance: `tests/test_score.py` has one case for each rule and passes.

### 1d Baseline
- [x] Savitzky-Golay detrend with a recorded window
- [x] astropy BLS over a recorded period grid
- [x] Test: the baseline recovers a 1 percent, 3-day injection on synthetic white noise
- [x] SDE threshold chosen on null episodes. The false-alarm rate is below 5 percent (18.63 on s98_v1)
- [x] Baseline run on the real episode set. The completeness table is in the LOGBOOK

Acceptance: a results file in `results/` with the completeness table. The LOGBOOK points to it.

### 1e LLM harness
- [x] An episode is presented to a model as a tool-using task with masked identity
- [x] Tools: summarize, bls_peaks, fold, odd_even, submit_prediction, with JSON Schemas
- [x] Provider-agnostic backend: OpenAI-compatible (OpenRouter, vLLM, Ollama) and Tinker
- [x] The harness uses the same scorer and the same results format
- [ ] One real run: three episodes on an open model through OpenRouter or Tinker, transcript reviewed

Acceptance: a results file for one model on the same episode set as the baseline.

## Phase 2: Tier 2 vetting

- [ ] EB injection: secondary eclipse, odd-even depth difference, V-shaped ingress
- [ ] Blend injection at pixel level, with the target pixel file and a neighbour position
- [ ] Real labelled cases from ExoFOP (CP and KP against FP). The disposition date is recorded
- [ ] Vetting scorer: class accuracy, and a penalty for a confident wrong answer
- [ ] Tools: odd-even test, secondary search, centroid test, Gaia cone search, TRICERATOPS

## Phase 3: Tier 3 agentic environment and RL

- [ ] Gym-style environment: `reset` gives a masked episode, `step` takes a tool call, the episode ends on a report
- [ ] Composite reward with explicit credit for each check and a penalty for a skipped check
- [ ] Tinker RL loop on an open model of the Qwen3 class, LoRA, reward from the scorer
- [ ] Training curves and held-out evaluation on the tier 1 and tier 2 sets
- [ ] Release the adapter as "Star Gazer v0" with the exact episode sets used in training

## Phase 4: Discovery run

- [ ] Run the best system over each 2-minute and FFI target with no TOI in the latest sectors
- [ ] Priority queue: late-K and M dwarfs, single- and double-transit cases, low TRICERATOPS FPP
- [ ] For each survivor: a full vetting report and preregistered predictions for the next sector
- [ ] Submit survivors as community TOIs on ExoFOP

## Open questions

- Does the base model know the TOI catalogue from training? Measure this. Compare masked
  and unmasked accuracy on known TOIs before you trust a tier 2 number.
- Most unvetted stars have only Full Frame Image (FFI) data at 10-minute or 30-minute
  cadence. Add a QLP or TESS-SPOC FFI loader when tier 1 is stable on 2-minute data.
