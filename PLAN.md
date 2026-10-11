# Plan

## Goal

Build a model that works like an exoplanet astronomer: it has the classical tools the
field already trusts for detection and vetting, it runs them in the right order on a
light curve, it weighs the evidence, and it reports a candidate or says "none". The
model does this at machine speed over thousands of stars, so that stars the mission
pipelines left under-vetted get the same scrutiny a careful human gives one target.

The model is trained with reinforcement learning against a verifiable reward, saved as
a checkpoint on Tinker, and run on light curves from any survey through one interface:
time, flux, error. The release is the checkpoint, the toolbox, the benchmark with its
baseline, and the logbook that shows how each number was produced.

Discovery is the measure. Not "beats BLS on a benchmark", but a candidate with a
vetting report and preregistered predictions that a follow-up observation confirms.

## Principles

- The toolbox does the signal processing; the model does the judgement. A model that
  tries to be the detector is worse than BLS at the easy part and no better at the
  hard part. Every tool result we read so far says the same thing.
- One interface for every survey. A tool takes three arrays. A survey adds a loader.
- Every number is reproducible: seeds, manifests, episode-set hashes, results with
  provenance, a logbook entry with the command.
- Do not overfit to one model family. The comparison set stays at three families, and
  tools are judged across them.

Tick a checkbox only when two conditions are true. The acceptance criterion is met. The
LOGBOOK has an entry that points to the evidence: a results file, a test run or a commit.

## Phase 0: Scaffold (done 2026-10-11)

- [x] Repository, `pyproject.toml` only, pre-commit with ruff, name check and the test suite
- [x] README, PLAN, LOGBOOK, STYLE; Sphinx docs with Mermaid diagrams
- [x] SDK design: `Detector` protocol, `EpisodeSet`, `evaluate`, `Results`; thin CLI
- [x] Test suite: one file for each module, one behaviour for each test, integration tests apart

## Phase 1: Tier 1 detection benchmark (done, being extended)

- [x] Host selection with TOI exclusion recorded by hash; 41 hosts in sector 98; light curves cached with a manifest
- [x] batman injection with geometry from the TIC; truth separated from episode files
- [x] Scorer with alias credit, explicit tolerances, zero-transit injections scored as null
- [x] BLS baseline; threshold chosen from null episodes (18.6 on s98_v1 for a 5 percent false-alarm rate)
- [x] Episode sets: s98_v1 (periods to 20 days) and s98_v2 (periods to 60 days, single transits common)
- [x] Baseline on s98_v2 recorded in the LOGBOOK
- [ ] One small episode set committed to git for tests and demonstrations
- [ ] Repeat each comparison run with three seeds and report mean and spread

## Phase 2: The agent harness (done, being measured)

- [x] Provider-agnostic loop: `ToolAgentDetector` over a `ChatBackend`; OpenAI-compatible and Tinker backends
- [x] Tinker backend uses each model family's native tool format through the cookbook renderer
- [x] Tools defined once with JSON Schemas; every call validated; errors returned to the model with the expected arguments
- [x] Budget in every tool result; nudge on a turn without a tool call; truncation detected; generous safety caps
- [x] Sampling defaults from the model cards, seed recorded; cost estimated from the Tinker price table
- [x] Full trajectory saved for each episode; `stargazer view` and `stargazer show`; `stargazer compare`
- [x] Parallel evaluation with a worker pool; models as separate processes
- [x] First full comparison on s98_v1: Inkling-Small and Qwen3.8-27B level with thresholded BLS (LOGBOOK)
- [x] Nemotron-3.5-Lightning full run on s98_v1 recorded (11 percent; loops to the turn cap)
- [x] Same models on s98_v1 with the masked search and single_events: precision rose (Inkling 36 percent at 5 percent false alarm, above BLS); faint-regime completeness did not
- [x] Inkling-Small on s98_v2: 32 percent at 11 percent false alarm; zero single-transit hits because the scorer requires a period
- [ ] Scorer: a single-event submission form (event time, duration, depth, no period) credited against injected transit times; set-level metrics for it
- [ ] Prompt rule: a peak near the noise level with no second line of evidence is "none" (wrong periods were noise peaks)
- [ ] Larger models on the same sets: Kimi K2.6, Nemotron 3 Ultra, Inkling, Qwen3.5-397B-A17B

## Phase 3: The toolbox, in the order an astronomer uses it

Each tool is a function on three arrays with a JSON Schema, a numpydoc Notes section
with its equation and a reference, a unit test on synthetic data, and a tool
description written for the model. The agent gets the tool only when the test passes.

### 3a Detection
- [x] `summarize`: cadence, scatter, flux range, gaps
- [x] `bls_peaks`: Box Least Squares with distinct peaks, alias periods, event counts, a detrend window, and masks
- [x] `single_events`: single-event statistic over box widths, the regime BLS cannot see
- [ ] Iterative multi-signal search: find, mask, repeat, until the next peak is noise (the pipeline's "multiple planet search")
- [ ] Transit Least Squares as a second search with a physical transit shape, for shallow signals
- [ ] Transit-shape fit with batman: depth, duration, impact parameter and their uncertainties; box versus V shape
- [ ] Duration consistency: does the fitted duration match the stellar density and period? (Seager and Mallén-Ornelas)

### 3b Vetting against false positives
- [x] `odd_even`: per-event depths with a verdict word
- [ ] Secondary eclipse search at phase 0.5 and at the period's half
- [ ] Ephemeris match: is the period and epoch a known eclipsing binary, TOI or nearby variable?
- [ ] Centroid motion from the target pixel file: does the dip come from the target or a neighbour?
- [ ] Difference imaging: in-transit minus out-of-transit image, where is the signal?
- [ ] Gaia neighbours within the aperture and the dilution they imply
- [ ] Systematics: momentum dumps, scattered light, and the Earth-and-Moon windows in TESS
- [ ] Stellar context: rotation period, flares, variability class from the light curve
- [ ] TRICERATOPS or an equivalent false-positive probability as the final number

### 3c Reporting
- [ ] `vetting_report`: one structured document from the tool results, in the form ExoFOP expects
- [ ] Preregistered predictions: transit times and depths for the next sector, with frozen code and hashes

## Phase 4: Data beyond one TESS sector

The trained model must not be specific to one survey. The tool interface is a light
curve as three arrays, so each survey needs only a loader and a target list.

- [ ] TESS Full Frame Image light curves (QLP or TESS-SPOC), 10- and 30-minute cadence; this is where unvetted stars are
- [ ] Multi-sector stitching for stars observed in several sectors
- [ ] Kepler long cadence: four-year baselines, the deepest injection-recovery reference
- [ ] K2: different systematics, the transfer test
- [ ] One ground-based survey (NGTS or WASP public data): gaps, airmass trends, red noise
- [ ] Injection into each survey with the same `BuildConfig`; one episode set for each survey
- [ ] Real confirmed planets mixed into the sets, so a policy cannot learn injection artefacts
- [ ] Vetting labels from ExoFOP dispositions with the disposition date recorded

## Phase 5: Reinforcement learning on Tinker

- [ ] Tier 3 environment in the cookbook's RL interface: the episode is the observation, tool calls are actions, `score_episode` is the reward
- [ ] Reward shaping decided and recorded: false-alarm penalty, credit for checks performed, cost per tool call
- [ ] Training on a mixture of surveys; one survey held out
- [ ] Training curves and held-out evaluation at matched false-alarm rate against the BLS baseline and the untrained models
- [ ] Checkpoint saved on Tinker; evaluated through `TinkerBackend(model_path=...)` with the same tools
- [ ] Release: checkpoint, toolbox, episode sets by hash, baseline, logbook

## Phase 6: Discovery run

- [ ] Target list: stars with no TOI in the latest sectors, FFI first, late-K and M dwarfs first
- [ ] Run the best system over the list with the full toolbox; keep every trajectory
- [ ] Survivors: vetting report, false-positive probability, preregistered predictions for the next sector
- [ ] Independent re-analysis of each survivor from the public data with a different model
- [ ] Submit as community TOIs on ExoFOP; request follow-up where warranted

## Handoff: state at the end of 2026-10-11

Read LOGBOOK.md first; it has every number and the command that produced it.

All runs of the day are finished and recorded: `results/full/` (old tools, three models on s98_v1), `results/full_v2tools/` (new tools: Qwen3.8-27B and Inkling-Small on s98_v1, Inkling-Small on s98_v2), baselines and sweeps for both sets. The repository is public at github.com/siddharth7113/stargazer_model; Read the Docs import is pending on the owner's account.

First steps for the next session:
1. Scorer: the single-event submission form, so single-transit episodes can score (LOGBOOK, benchmark finding).
2. Phase 3a: iterative multi-signal search and the transit-shape fit, with tests.
3. Phase 3b: secondary eclipse search and the ephemeris match, the two cheapest vetting tools.
4. Then the larger models on s98_v1 and s98_v2, three seeds each.

## Open questions

- Does the base model know the TOI catalogue from training? Compare masked and unmasked
  accuracy on known TOIs before trusting a tier 2 number.
- GLM-5.3's tokenizer fails with the pinned transformers in the cookbook; retry after an update.
- Nemotron explores to the turn cap on hard episodes; decide whether that is a prompt problem or a cost to accept.
