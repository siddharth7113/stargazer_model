# Plan

Working rule: a checkbox is ticked only when the acceptance criterion is met and the
LOGBOOK has an entry pointing at the evidence (a results file, a test run, or a commit).

## Phase 0: Scaffold (today, 2026-10-11)

- [ ] Repo initialised, `pyproject.toml` only, no lock file
- [ ] README, PLAN, LOGBOOK written
- [ ] Package skeleton imports cleanly
- [ ] `pytest` passes offline

Acceptance: a fresh clone with `uv pip install -e ".[dev]"` and `pytest` is green.

## Phase 1: Tier 1 detection benchmark

### 1a Data
- [ ] `stargazer targets` pulls the 2-minute target list for a sector from MAST
- [ ] TOI cross-match excludes any star with a TOI; TOI CSV hash and date recorded
- [ ] Stellar radius, Teff, logg stored per target from the FITS header
- [ ] `stargazer fetch` caches light curves as `.npz` with SHA-256 manifest
- [ ] At least 50 quiet host stars cached for one sector

Acceptance: `stargazer verify` passes on the cache; the targets file lists the TOI CSV hash.

### 1b Injection
- [ ] batman injection with period, epoch, rp/rs, a/rs from stellar density, impact parameter
- [ ] Injected depth at mid-transit matches rp/rs squared within limb-darkening tolerance (test)
- [ ] Same seed produces byte-identical episode set (test)
- [ ] Expected SNR per episode computed and stored in truth (depth / point scatter × sqrt(N in transit))
- [ ] Episode mix: 30 percent null, 70 percent injected; periods 0.5 to 20 days so single- and
      double-transit cases appear; depths spanning 200 ppm to 2 percent

Acceptance: `tests/test_inject.py` green; one committed small episode set (20 episodes).

### 1c Scorer
- [ ] Period match with aliases 1/2, 2, 1/3, 3 at partial credit
- [ ] Epoch match modulo period within half a duration
- [ ] Null handling: "none" on null = 1.0, false alarm = 0.0, miss on injected = 0.0
- [ ] Set-level: completeness vs SNR bins, completeness vs number of transits, false-alarm rate
- [ ] Results JSON includes git commit, episode-set hash, config, per-episode rows

Acceptance: `tests/test_score.py` green; a hand-constructed case for every rule.

### 1d Baseline
- [ ] Detrend with wotan biweight (window recorded)
- [ ] astropy BoxLeastSquares over a logged period grid
- [ ] Signal detection efficiency threshold chosen on null episodes to give under 5 percent false alarms
- [ ] Baseline recovers a 1 percent, 3-day injection on synthetic white noise (offline test)
- [ ] Baseline run on the real episode set, completeness curve in LOGBOOK

Acceptance: a results file in `results/` with the completeness table, referenced from LOGBOOK.

### 1e LLM harness
- [ ] Episode presented to a model as a tool-using task with masked identity
- [ ] Tools: load, detrend, bls, fold, plot-to-text summary
- [ ] Runs against Claude via the API and against a local open model
- [ ] Scored with the same scorer, same results format

Acceptance: a results file for at least one model on the same episode set as the baseline.

## Phase 2: Tier 2 vetting

- [ ] EB injection: secondary eclipse, odd-even depth difference, V-shaped ingress
- [ ] Blend injection at pixel level using the target pixel file and a neighbour's position
- [ ] Real labelled cases from ExoFOP (CP, KP vs FP) with disposition date recorded
- [ ] Vetting scorer: class accuracy, and penalty for confident wrong answers
- [ ] Tools added: odd-even test, secondary search, centroid test, Gaia cone search, TRICERATOPS

## Phase 3: Tier 3 agentic environment and RL

- [ ] Gym-style env: reset gives a masked TIC, step takes a tool call, episode ends on a report
- [ ] Composite reward with explicit per-check credit and a penalty for skipped checks
- [ ] Tinker RL loop on an open model (Qwen3 class), LoRA, reward from the scorer
- [ ] Training curves and held-out evaluation against the tier 1 and tier 2 sets
- [ ] Release the adapter as "Star Gazer v0" with the exact episode sets it was trained on

## Phase 4: Discovery run

- [ ] Run the best system over every 2-minute and FFI target in the latest sectors with no TOI
- [ ] Priority queue: late-K and M dwarfs, single- and double-transit cases, low TRICERATOPS FPP
- [ ] For each survivor: full vetting report, preregistered predictions for the next sector
- [ ] Submit survivors as community TOIs on ExoFOP

## Open questions

- Does the base model have the TOI catalogue memorised? Measure by comparing masked and
  unmasked accuracy on known TOIs before trusting any tier 2 number.
- FFI (10-minute and 30-minute cadence) data is where most unvetted stars are. Add a QLP or
  TESS-SPOC FFI loader once tier 1 is stable on 2-minute data.
