# Plan

## Goal

Build a model that works like an exoplanet astronomer. The model has the classical
detection and vetting tools that the field trusts. It runs them in the right order on a
light curve, weighs the evidence and reports a candidate or "none". The model does this
at machine speed over thousands of stars. Stars that the mission pipelines left
under-vetted then get the same scrutiny that a careful human gives one target.

Reinforcement learning trains the model against a verifiable reward. The trained model
is saved as a checkpoint on Tinker. It runs on light curves from any survey through one
interface of three arrays: time, flux and flux error. The release contains the
checkpoint, the toolbox, the benchmark with its baseline, and the logbook that shows how
each number was produced.

Discovery is the measure of success. A discovery is a candidate with a vetting report
and preregistered predictions that a follow-up observation confirms. A higher benchmark
score than BLS is not a discovery.

## Principles

- The toolbox does the signal processing; the model does the judgement. A model that
  tries to be the detector is worse than BLS at the easy part and no better at the
  hard part. Every tool result read so far agrees with this.
- One interface for every survey. A tool takes three arrays. A survey adds a loader.
- Every number is reproducible through seeds, manifests, episode-set hashes, results
  with provenance and a logbook entry with the command.
- Do not overfit to one model family. The comparison set stays at three families, and
  tools are judged across them.

Tick a checkbox only when two conditions are true. The acceptance criterion is met. The
LOGBOOK has an entry that points to the evidence: a results file, a test run or a commit.

## Phase 0: Scaffold (done 2026-10-11)

- [x] Repository, `pyproject.toml` only, pre-commit with ruff, name check and the test suite
- [x] README, PLAN, LOGBOOK; Sphinx docs with Mermaid diagrams
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
- [x] Scorer: a single-event submission form (event time, duration, depth, no period) credited against injected transit times; set-level metrics for it (LOGBOOK 2026-10-11, single-event form)
- [ ] Prompt rule: a peak near the noise level with no second line of evidence is "none" (wrong periods were noise peaks)
- [ ] Larger models on the same sets: Kimi K2.6, Nemotron 3 Ultra, Inkling, Qwen3.5-397B-A17B

## Phase 3: The toolbox, in the order an astronomer uses it

Each tool is a function of three arrays. Each tool has a JSON Schema, a numpydoc Notes
section with its equation and a reference, a unit test on synthetic data, and a tool
description written for the model. The agent gets the tool only when the test passes.

### 3a Detection
- [x] `summarize`: cadence, scatter, flux range, gaps
- [x] `bls_peaks`: Box Least Squares with distinct peaks, alias periods, event counts, a detrend window, and masks
- [x] `single_events`: single-event statistic over box widths, the regime BLS cannot see; window tied to the box width and noise at the box timescale (LOGBOOK 2026-10-11)
- [x] Iterative multi-signal search: find, mask, repeat, until the next peak is noise (the pipeline's "multiple planet search"); `search_signals` (LOGBOOK 2026-10-11)
- [ ] Transit Least Squares as a second search with a physical transit shape, for shallow signals
- [x] Transit-shape fit with batman: depth, duration, impact parameter and their uncertainties; box versus V shape; `fit_transit` (LOGBOOK 2026-10-11)
- [x] Duration consistency: `fit_transit` reports the host density from the period and a/R* with a verdict for a dwarf host (Seager and Mallén-Ornelas); the comparison with the TIC density needs the host parameters, which a tool does not see (LOGBOOK 2026-10-11)

### 3b Vetting against false positives
- [x] `odd_even`: per-event depths with a verdict word
- [x] Secondary eclipse search at phase 0.5 and at any other phase; `secondary_eclipse` (LOGBOOK 2026-10-11)
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

### 3d Course notebooks
- [ ] `examples/`: one Jupyter notebook for each tool, for a human reader, in the STYLE.md register, with plots. Each notebook shows one real episode, the tool's equation, the tool's output and what the plot shows. Added as each tool is finished. Done so far: the episode and summarize, injection, the period search, vetting a candidate (fold, fit_transit, odd_even, secondary_eclipse), single events (LOGBOOK 2026-10-11). To come: scoring; the agent loop.

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

The day added the single-event submission form, four tools (search_signals,
fit_transit, secondary_eclipse, and the rebuilt single_events with a window tied to
the box width, noise at the box timescale and a segment-edge guard), and three full
Inkling-Small runs on s98_v2 that measure each step: `results/full_v3form/` (the
form, then the edge guard), `results/full_v4search/` (search_signals and the prompt
rules; mean score 0.538 at an 11 percent false-alarm rate, the best so far). The
`full_v5fit` run, with fit_transit and secondary_eclipse, gives 0.583 at a 3 percent
false-alarm rate (LOGBOOK), the best so far; it also showed that a lone transit paired
with an epoch in a gap was reported as two events, which is fixed. Inkling-Small needs
`--renderer tml_v0`. The course notebooks (Phase 3d) exist for every tool and the scorer.

The prompt is split into `base` (the task, the scoring, the units) and `rules` (the
procedure and decision rules of 2026-10-11, frozen). The owner wants the model to
learn the rules, so runs use `--prompt base` from here and the rules prompt is a
comparison row. The `full_v6count` run is read in the LOGBOOK; the `full_v7base`
run with the base prompt is the next number.

First steps for the next session:
1. Read `results/full_v7base/agent_inkling_small_s98_v2.json` against the v6count
   row. The gap is the value of the rules block. Then the same with the rules prompt
   and the three-way shape verdict if the gap is large, so both rows share the tools.
2. Phase 3b: the ephemeris match against known eclipsing binaries and TOIs, the last
   vetting tool that needs only the light curve's coordinates and the catalogues.
3. The larger models on s98_v1 and s98_v2, three seeds each, and the course notebooks
   (Phase 3d) for the tools that exist.

## Open questions

- Does the base model know the TOI catalogue from training? Compare masked and unmasked
  accuracy on known TOIs before trusting a tier 2 number.
- GLM-5.3's tokenizer fails with the pinned transformers in the cookbook; retry after an update.
- Nemotron explores to the turn cap on hard episodes; decide whether that is a prompt problem or a cost to accept.
