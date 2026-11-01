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
- [x] Host selection: exclude stars in the TESS eclipsing-binary catalogue and Gaia ECL class, as TOI hosts are excluded; s98_v3 built on the 36 clean hosts, BLS baseline recorded (LOGBOOK 2026-10-16)
- [ ] One small episode set committed to git for tests and demonstrations
- [x] A held-out episode set from a second sector: s92_v1, 28 hosts, hash `facd9778...`, BLS 0.32 at 5 percent false alarm (LOGBOOK 2026-10-25). No tool and no prompt is changed from reading its failures; it only confirms a change that s98_v3 showed, and every tool is measured on both before the toolbox freezes
- [x] Repeat each comparison run with three seeds and report mean and spread: Inkling-Small on s98_v3, both prompts (LOGBOOK 2026-10-16); the four large models with the base prompt (LOGBOOK 2026-10-25); the rules prompt on the large models is at two seeds, the rest running on 2026-10-25

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
- [x] Prompt rule: a peak near the noise level with no second line of evidence is "none"; the rules prompt cuts wrong periods from about 30 to 3 to 10 a run and the false-alarm rate from 31 to 2 percent on Inkling-Small (LOGBOOK 2026-10-16), and by 27 and 18 points on Inkling and Nemotron 3 Ultra (LOGBOOK 2026-10-25)
- [x] Larger models on the same sets: Kimi K2.6, Nemotron 3 Ultra, Inkling, Qwen3.5-397B-A17B, three seeds each with the base prompt on s98_v3 (LOGBOOK 2026-10-25); Kimi K2.6 is the only one above the small model's seed spread
- [x] Harness check from the traces: argument coercion, the agent reads the coerced submission, `stargazer rescore` (LOGBOOK 2026-10-25)

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
- [x] Ephemeris match: is the period and epoch a known eclipsing binary, TOI or nearby variable? `ephemeris_match` on the episode context (LOGBOOK 2026-10-11)
- [ ] Centroid motion from the target pixel file: does the dip come from the target or a neighbour?
- [ ] Difference imaging: in-transit minus out-of-transit image, where is the signal?
- [x] Gaia neighbours within the aperture and the dilution they imply; `nearby_stars` (LOGBOOK 2026-10-11)
- [x] Systematics, first the segment edges: `edge_verdict` on each peak of `bls_peaks` and `search_signals`, `at_segment_edges` when every event with data lies within half a day of the start or the end of a data segment, where the scattered-light ramps are; a replay over 20 runs showed it names 71 of 116 periodic false alarms and 46 of 47 single events on no transit, and no findable planet with two or more events. Null on Inkling-Small with the rules prompt (false alarms 2.3 to 0.6 percent, completeness 39.2 to 38.1, score 0.573 to 0.568 on s98_v3; 7.0 to 0.0 percent and 0.575 to 0.590 on s92_v1), because that model was not submitting the ramps; kept for the large-model pass. The first measurement was of a defective version (edges computed after the masks) and is on file as such (LOGBOOK 2026-10-31). Still open: momentum dumps and the Earth-and-Moon windows
- [x] Orbit shape verdict, `orbit_shape`: flat, or variable at the period as a pulsating or rotating star is, from a long-window fold with a period contrast; names the 0.946-day pulsator on TIC 55270064 and the 0.92-day rotation of TIC 32816814, reads all 96 injected planets with five orbits as flat. As a field of `fold` it cost Inkling-Small 3 points of completeness, because the model stopped at the first `variable` word, so it is a separate tool; no gain measured on the small model, its test is the large-model pass and s92_v1 (LOGBOOK 2026-10-25)
- [ ] Stellar context: rotation period, flares, variability class from the light curve. Not built before the freeze: the spotted-host misses are red-noise limited or need a transit-shaped search, and a rotation-harmonic match is unsafe as a verdict (LOGBOOK 2026-10-31)
- [ ] TRICERATOPS or an equivalent false-positive probability as the final number

### 3c Reporting
- [ ] `vetting_report`: one structured document from the tool results, in the form ExoFOP expects
- [ ] Preregistered predictions: transit times and depths for the next sector, with frozen code and hashes

### 3d Course notebooks
- [x] `examples/`: one Jupyter notebook for each tool, for a human reader, in the STYLE.md register, with plots. Each notebook shows one real episode, the tool's equation, the tool's output and what the plot shows. Seven notebooks: the episode and summarize, injection, the period search, vetting a candidate (fold, fit_transit, odd_even, secondary_eclipse), single events, scoring, reading a run (LOGBOOK 2026-10-11; commits 6f9b513, 8a188de; the owner rewrote the text of 01 and 02 and added the figures)
- [ ] Notebooks for the catalogue tools `ephemeris_match` and `nearby_stars`, and one for each new tool as it is finished

## Phase 4: Data beyond one TESS sector

The trained model must not be specific to one survey. The tool interface is a light
curve as three arrays, so each survey needs only a loader and a target list.

- [ ] TESS Full Frame Image light curves (QLP or TESS-SPOC), 10- and 30-minute cadence; this is where unvetted stars are
- [ ] Multi-sector stitching for stars observed in several sectors
- [ ] Kepler long cadence: four-year baselines, the deepest injection-recovery reference
- [ ] K2: different systematics, the transfer test
- [ ] One ground-based survey (NGTS or WASP public data): gaps, airmass trends, red noise
- [ ] Injection into each survey with the same `BuildConfig`; one episode set for each survey
- [x] Real confirmed planets: `stargazer targets --real`, `build --real` (each catalogue ephemeris verified against the data: confirmed, shifted or absent), `context --mask-tois`; the set s98_real_v2 (41 confirmed single-planet hosts of sector 98 with the flux as observed, 40 nulls). Inkling-Small with the rules prompt finds 71 percent at 2.2 percent false alarms, all 20 planets deeper than 5,000 ppm and 3 of 10 under 1,000 ppm; BLS at the same false-alarm rate 66 percent (LOGBOOK 2026-11-01)
- [x] A copy of the TESS eclipsing-binary catalogue on disk (`data/catalogues/`, the MAST CSV) and a Vizier mirror option, so a target cut or a context never waits on Strasbourg; the masked real-planet context leaves out the catalogue's entry for the hidden planet itself (LOGBOOK 2026-11-01)
- [ ] Vetting labels from ExoFOP dispositions with the disposition date recorded

## Phase 5: Reinforcement learning on Tinker

- [x] Tier 3 environment in the cookbook's RL interface: the episode is the observation, tool calls are actions, `score_episode` is the reward (`stargazer/environment.py`, `stargazer train`; the three-step cost test on Inkling-Small, about 90 seconds and about one dollar a step of 32 episodes, LOGBOOK 2026-11-01)
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

## Handoff: state on 2026-11-01

Read LOGBOOK.md first; it has every number and the command that produced it.

The session of 2026-11-01, in order. The parity sentence went into the rules prompt
(step 3: two or more events of one parity are still a period; the decision rule
accepts `insufficient_data` in that case, which the first three-episode check showed
was needed). Measured with three seeds on s98_v3, one on s92_v1 and one on
s98_real_v2 (`results/full_v15parity/`, $46): null on the injected sets by the rule
of 2026-10-25 (completeness up 0.4 on s98_v3 inside the seed spread, down 0.8 on
s92_v1), and on the real set the false alarm gone and completeness 71.4 to 74.3
percent, TOI-2184.01 and TOI-220.01 found and the host of TOI-1338 declined from the
tools alone. The sentence stays because the old one was false; the reference rows for
the frozen toolbox with the current prompt are the `full_v15parity` runs on all
three sets (LOGBOOK 2026-11-01). The prompt hash is `a475afde...`; every earlier rules
run carries `265fd9ab...`.

The eclipsing-binary refill no longer waits on Strasbourg. The TESS catalogue is
committed as a CSV from MAST under `data/catalogues/` and `stargazer context
--eclipsing-binaries PATH` reads it; `--vizier-server vizier.cfa.harvard.edu` routes
the Vizier queries to the Harvard mirror, which answered the small catalogues in a
second and then timed out on two refill attempts, so the Gaia classes of the real
hosts are still empty; a masked set no longer downloads the TOI list. Two defects
fixed on the way: the eclipse epoch of both catalogue copies is BTJD and was stored as
BJD (never exercised, no context had an entry); and nine of the 35 real planets sit in
the eclipsing-binary catalogue at their own period, so the masked writer now leaves
out an entry at the hidden planet's period and counts it in the source record. The
small model with the rules prompt never calls `ephemeris_match`, so the leak measured
zero on it (`agent_..._ebfilled.json`, identical numbers); it would act on a model that
calls the tool. The run on the masked refill is `agent_..._ebmasked.json` in
`results/s98_real_v2/`: 77.1 percent at no false alarm, one episode different from
the parity run, the seed's noise. The set's hash is now `6ba662d9...` and that run is
its reference row.

Phase 5 reading, done against the installed cookbook 0.5.7 and tinker 0.33.1. An
environment implements `initial_observation` and `step`, which returns a
`StepResult` with the reward, and `EnvGroupBuilder.make_envs` builds a group;
`tinker_cookbook.tool_use.build_agent_tool_env` wraps a list of `Tool` objects and a
`reward_fn` over the message history into such an environment, parses responses with
the renderer and takes the tool declarations from
`renderer.create_conversation_prefix_with_tools`, which is the format the backend
already uses. `rl.train.Config` takes `model_name`, `learning_rate`, `lora_rank`,
`max_tokens`, `renderer_name` and a `dataset_builder` whose batches are groups of
environments; the recipe to copy is `recipes/search_tool/`; the rollout preset for
Inkling models is `agentic` (10 turns, 30 tool calls). No installed file says which
models train: `ServiceClient.get_server_capabilities()` returns `trainable` for each
model, and that call is the first step. `get_lr` is calibrated for Qwen only; the
weights README suggests LoRA learning rates near 5e-4.

Phase 5 steps 1 to 3 are done the same day (LOGBOOK 2026-11-01): every sampling
model trains; the environment is `stargazer/environment.py` with `stargazer train`
(nine offline tests with a stub model; the detector's caps set on the runner and the
environment through an explicit `RolloutConfig`, because the cookbook's Inkling
preset cut rollouts at 10 turns unscored; a run stops on the first failed rollout;
`record.json` with the sets by hash; a run with `--save-every 0` discards its final
checkpoint). The cost test: a step of 8 groups of 4 on Inkling-Small is about 90
seconds and about a dollar. The group-signal measurement under the right caps
(`results/train/group_signal2_*`): at temperature 1.0 and group 4 no group of the
eight stars is mixed; group 8 and 16 add mixed groups on a planet and on the null
stars; temperature 1.3 is the lever that reaches a hard planet (found in 1 of 4
rollouts), at 6 percent false alarms and entropy 0.48. The owner's standing rule
since 2026-11-01: small-model runs need no ask; say so before a big-model run. The
Tinker free checkpoint store held five final checkpoints (about 170 GB) and refused
a sixth training client; it is empty now.

Nothing is running.

First three steps of the next session:

1. Decide the checkpoint store with the owner (asked on 2026-11-01, answer open):
   paid storage at $0.10 per GB a month and a checkpoint every 10 steps, or the free
   store with a checkpoint every 25 steps and the training state discarded after the
   run, the sampler weights kept. Write the choice here and in the run script.
2. Decide the reward shaping and record it in PLAN.md: the plain score (today), or
   partial credit for the procedure's steps on a hard planet, or a false-alarm
   penalty below zero. Write the chosen reward as a class beside `EpisodeReward`
   with its own tests, so the plain score stays available as the control.
3. The first real run: Inkling-Small, rules prompt, s98_v3 as the training set and
   s92_v1 as the test set (`--test-episodes`, `--eval-every 10`), group 8 at
   temperature 1.2, learning rate 5e-5, 30 to 50 steps (about $3 and 6 minutes a
   step), checkpoints by the decision of step 1; then evaluate the checkpoint
   through `TinkerBackend(model_path=...)` with `stargazer agent` on s92_v1 and
   s98_real_v2, against the `full_v15parity` rows.

Still open from before: the `variable` stop on active hosts (Phase 5, above); the
Gaia classes of the 43 real hosts (rerun the refill with `--vizier-server` when a
mirror holds, or without it when Strasbourg answers); the Full Frame Image loader;
TIC 55270064's neighbour at 6.7 arcseconds; the small committed episode set for tests.

## Open questions

- Does the base model know the TOI catalogue from training? Compare masked and unmasked
  accuracy on known TOIs before trusting a tier 2 number.
- GLM-5.3's tokenizer fails with the pinned transformers in the cookbook; retry after an update.
- Nemotron explores to the turn cap on hard episodes; decide whether that is a prompt problem or a cost to accept.
- Closed 2026-10-31: TIC 190310047 in s92_v1, the 12.84-day dip submitted in all five
  of its null episodes. Its one event sits 0.33 days from a gap edge, the segment-edge
  verdict names it, and the owner's own check reads it as instrumental.
- Three null hosts of s98_v3 carry signals that every model submits (TIC 55270064 at
  0.946 days, TIC 146420104 at 15.4 days, TIC 33831807 at 14.4 days). Until each is
  explained, the false-alarm rate on s98_v3 measures these three hosts as much as the
  models.
