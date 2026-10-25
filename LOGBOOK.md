# Logbook

Newest entry at the bottom. Each entry has: date, what was done, exact command, evidence.

## 2026-10-11

**Context.** Project started after reading about an independent researcher's AI-assisted
candidate around TIC 4206066 (3.18 day, about 500 ppm, Zenodo preprint, not confirmed).
Decision: build a general injection-recovery benchmark. The project does not follow up
that one candidate.

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
- Limb darkening is a replaceable table with a binary search. It replaces an `if` chain.

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
file is finite. Size 62 MB, not in git; the hash is the reference. Sector 98 light
curves span 57 days. Earlier TESS sectors span 27 days; TESS now observes longer sectors. The injection period
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

**Reading.** The baseline is weak at SNR 7 to 50 with few transits, and a reasoning
system can gain in that regime. Nearly every episode has six or more transits because sector 98 is
57 days long and the period range stops at 20 days. Set s98_v2 should extend periods to
60 days so single and double transits are common.

**Manifest incident.** `stargazer verify data/episodes/s98_v1` reported hash mismatches
on `config.json` and `truth.json`. Cause: the episode set had been added to git by a
`git add -A`, and the end-of-file pre-commit hook appended a newline to both JSON
files. Fix: the trailing newline was removed, verify passes, and the recorded hash
`1ed1abbb...` is unchanged. The set is untracked and `data/episodes/` is ignored.
Pre-commit hooks now exclude `data/` and `results/`. Lesson: artifacts with a manifest
must never pass through a formatter. The 62 MB set is still in the history of commits
`6a2814a` to `5f92408`; rewriting history would remove it, which is a separate decision.

**Secrets.** `.env.example` lists the key variables. The CLI loads `.env` through
python-dotenv at start. Git ignores `.env`. LOGBOOK.md, PLAN.md and STYLE.md are
untracked through `.git/info/exclude` at the owner's request.

**First agent run, Qwen3-8B on Tinker.**
```
stargazer agent --episodes data/episodes/s98_v1 --backend tinker --model Qwen/Qwen3-8B --limit 3 \
    --out results/agent_tinker_qwen3_8b_s98_v1_first3.json
```
Episode set hash `1ed1abbb...`, tinker 0.33.1, renderer `qwen3_disable_thinking`,
temperature 0. Three episodes in 80 seconds wall time, 81,000 tokens in total, cost not
reported by Tinker. Results and transcripts: `results/agent_tinker_qwen3_8b_s98_v1_first3.json`.

| episode | truth | SNR | agent | outcome | tool calls | tokens |
|---|---|---|---|---|---|---|
| ep_0000 | transit, P = 0.851 d | 188 | P = 0.8509 d, t0 ok | hit, 1.0 | 6 | 14,335 |
| ep_0001 | transit, P = 2.262 d | 8.7 | P = 14.40 d | wrong_period, 0.0 | 7 | 17,426 |
| ep_0002 | transit, P = 9.164 d | 4.7 | no submission | miss, 0.0 | 12 (cap) | 49,636 |

The loop, the Qwen3 tool-call text format, the renderer and the scorer all work with a
real model. The run also exposed three defects, listed by severity.

1. *The odd_even error model is wrong on active stars.* On ep_0000 the host varies by
   17 percent peak to peak. The tool reported a 16 sigma odd-even difference for a real
   injected planet, because its standard error uses the point scatter divided by
   sqrt(n), which ignores residual variability after detrending. The error must come
   from the scatter of the per-event depths. This is a tool bug. The model did not
   cause it.
2. *The model inverted the odd_even rule.* Its reasoning on ep_0000 reads "15.97 sigma,
   suggesting the signal is not due to an eclipsing binary". The tool description says
   the opposite. An 8B model at temperature 0 misreads a numeric rule; the tool should
   return an explicit verdict field such as `consistent_with_planet: false` so the
   model does not have to apply the threshold itself. The hit on ep_0000 was right for
   the wrong reason.
3. *No submission at the cap.* On ep_0002 the model folded five candidate periods, got
   null odd-even results for periods with one event, and ran out of tool calls without
   calling submit_prediction. The system prompt should say: after the budget is spent,
   submit detected=false. The scorer treated the non-submission as a miss, which on
   this SNR 4.7 episode is the same score "none" would get, so no number changed.

Also: on ep_0001 the true 2.26-day signal at SNR 8.7 was not among the five BLS peaks on
a star with 4,300 ppm scatter and 14 percent variability. The 0.75-day detrend window is
too wide for that star. The BLS baseline fails there for the same reason, so this is a
limitation of the shared tool, to be addressed by a detrend-window argument on
`bls_peaks`.

**Git history.** The 62 MB episode set was removed from history with git-filter-repo
(`--path data/episodes --invert-paths`), reflog expired, aggressive gc. `.git` went from
78 MB to 352 KB; nine commits remain; `.git/info/exclude` was restored afterwards.

**Qwen3.8-27B on Tinker, three episodes, three runs.**

Run 1, before the fixes (`results/agent_tinker_qwen3_8_27b_s98_v1_first3.json`): all
three episodes ended at the tool-call cap of 12 with no submission, 55,000 tokens each.
The transcripts show correct tool use and a careful alias check; the model had no
room to finish. The cap was a CLI default of 12 that overrode the class default.

Fixes applied (commit after this entry): every tool result carries `calls_remaining`;
the system prompt states the scoring rules, a procedure and a stop rule; `odd_even` uses
the scatter of per-event depths for its error and returns a `verdict` word; `fold`
returns 40 integer-ppm bins instead of 50 float bins with per-bin counts; each BLS peak
carries `half_period`, `double_period` and `n_events_in_data`; validation errors return
the expected arguments; the call cap is now a safety stop of 40. It does not act as a
budget.

Run 2, after the fixes, renderer `qwen3_8_disable_thinking`
(`results/agent_tinker_qwen3_8_27b_s98_v1_first3_v2.json`):

| episode | truth | SNR | agent | outcome | calls | tokens |
|---|---|---|---|---|---|---|
| ep_0000 | transit, P = 0.851 d | 188 | P = 0.8509 d | hit, 1.0 | 6 | 15,214 |
| ep_0001 | transit, P = 2.262 d | 8.7 | detected=false | miss, 0.0 | 9 | 28,075 |
| ep_0002 | transit, P = 9.164 d | 4.7 | detected=false | miss, 0.0 | 9 | 28,177 |

The two misses are reasoned refusals. The loop did not fail. On ep_0001 the model
rejected the 14.4-day peak because its odd and even events disagree and the full-orbit
fold shows a second dip; that is the right call for that peak. The injected 2.26-day,
1,257 ppm signal never appears among the five BLS peaks at any detrend window from
0.75 down to 0.15 days, because a single deep 1.5 percent dip at day 9.65 dominates the
power spectrum at every window. The toolbox needs a tool that masks the strongest
feature and searches again, as the SPOC pipeline does. Added to PLAN 1e.

Run 3, renderer `qwen3_8_medium_reasoning`
(`results/agent_tinker_qwen3_8_27b_reasoning_s98_v1_first3.json`): identical answers
and token counts. A direct probe shows the renderer opens a `<think>` block and the
model closes it; on this task at medium effort the block was empty on every turn, so
there is no trace to read. The trajectory capture works (checked on a direct probe
where the model did think). `qwen3_8_xhigh_reasoning` is the next thing to try for a
readable trace.

**Trajectories.** Every results file now holds, for each episode, the full message list
the backend saw, the model's text, the per-turn thinking when present, and the tool
transcript. `stargazer show --results <file> --episode ep_0001` prints it.

**Qwen3.8-27B at `qwen3_8_xhigh_reasoning`, two episodes**
(`results/agent_tinker_qwen3_8_27b_xhigh_s98_v1_first2.json`). First attempt failed on
both episodes with `TypeError: NoneType - int`: the CLI passed `--max-tool-calls` None
straight into the detector. Fixed in commit `e110947`: None now selects the class
default, the progress line prints a recorded error, and `tests/test_cli.py` checks the
argument defaults. Second attempt: ep_0000 hit in 6 calls; ep_0001 detected=false in 9
calls. The model now chose `detrend_window_days` 0.3 on its own after reading the
flux range in summarize, which is the behaviour the new tool description asks for.

Thinking trace: still empty on every turn, with 537 output tokens over 9 turns. A
direct probe shows Qwen3.8-27B does reason inside `<think>` on a plain question, but on
tool-calling turns at temperature 0 it closes the block at once, even at xhigh effort.
The capture code is correct; this model does not think before tool calls. To read a
reasoning trace, use a model whose template reasons before each call, for example
Nemotron 3.5 with `nemotron3` or GLM-5.3 once its tokenizer loads, or ask for a short
written rationale in the assistant text before each call. Added to PLAN.


**Correction.** The two "reasoning" runs above (`qwen3_8_medium_reasoning` and
`qwen3_8_xhigh_reasoning`) did not use those renderers. The CLI parsed `--renderer`
but never passed it to the backend, so both ran with the default
`qwen3_disable_thinking` template. The conclusion "Qwen3.8 emits no thinking on tool
turns" is withdrawn. Fixed in the next commit, with a CLI test that checks the flags
reach the backend. Lesson: a parsed-but-unused flag is the worst kind of bug in an
experiment harness. The results file records the intended setting, and the real
setting stays hidden. From now on the backend config in the results file is the only
source of truth for what ran. For those two files it says
`renderer: qwen3_disable_thinking`.

**Sampling defaults.** Temperature 0 was my choice for determinism; it was wrong on
two counts. The Qwen3 model cards discourage greedy decoding because it degrades
quality and can loop, and recommend temperature 0.7, top_p 0.8, top_k 20 in
non-thinking mode and 0.6, 0.95, 20 in thinking mode. The Tinker cookbook uses 0.7
for its tool-use benchmark. Also, temperature 0 does not make a hosted model
deterministic. New defaults: the model-card values keyed on the renderer's mode,
a recorded seed (default 0), and the rule that variance is measured by repeating a run
with other seeds. Earlier runs in this logbook used temperature 0 and are not
comparable with later ones.

**Three-model smoke test and the native tool format.** First smoke test with
Qwen3.8-27B, Nemotron-3.5-Lightning-30B-A3B and Inkling-Small on three episodes each,
three workers, in parallel (`results/smoke/`). Nemotron ended every episode as
`model_stopped`: its transcripts show a correct `submit_prediction` written three times
in a `<tool_call>` block without the closing tag, which my regex required. Inkling
wrote its own tool-call tokens (`<|content_invoke_tool_json|>`) as text, because the
prompt demanded Qwen's `<tool_call>` format from every model. Both were my fault. The
backend parsed text by hand and imposed one family's wire format on all families.

Fix (commit `6bb361e`): the Tinker backend now declares tools through the cookbook
renderer's `create_conversation_prefix_with_tools`, converts the conversation to the
cookbook `Message` type with `ToolCall` objects and tool results carrying both the call
id and the function name, and parses replies with the renderer's `parse_response`,
which returns parsed `tool_calls`, the text and any thinking part. The system prompt
no longer mentions a tool-call syntax. The plain-text `<tool_call>` parser remains
only as a fallback. The RL phase needs the same design, because a policy is trained
and evaluated in its own template.

**Parallel evaluation.** `evaluate` takes `workers`; episodes run in a thread pool and
results keep episode order. The Tinker and OpenAI backends guard their lazy setup
with a lock. `stargazer agent --workers 8` is the default. Separate models run as
separate processes.

**Viewer.** `stargazer view --results <file>` opens a Textual terminal viewer: episode
table on the left with kind, SNR, outcome, score, calls and stop reason; the
trajectory on the right with thinking, tool calls, shortened results and nudges;
``t`` toggles thinking, ``r`` toggles full results. `stargazer show` prints the same
rendering for one episode.

**Cost reporting.** Tinker returns no cost per call, so earlier runs printed
`usd=0.0000`, which was a placeholder presented as a number. Now a Tinker run estimates
cost from the published sampling price per million tokens (table in
`stargazer.backends`, read 2026-10-11) on prompt plus sampled tokens, and a backend
with no known price reports no cost field at all. The Tinker console is the bill; the
estimate is for choosing experiments. Observed in the full runs: Inkling-Small about
20,000 tokens an episode, Qwen3.8-27B about 45,000, Nemotron-3.5-Lightning about
200,000 because it explores to the 30-turn cap on hard episodes.

**Full runs launched** at 07:47 on `s98_v1`, all 200 episodes, eight workers each,
three processes in parallel: Qwen3.8-27B (`qwen3_8_disable_thinking`),
Nemotron-3.5-Lightning-30B-A3B (`nemotron3_disable_thinking`), Inkling-Small
(`tml_v0`), model-card sampling defaults, seed 0. Results will be
`results/full/agent_<tag>_s98_v1.json`. Progress printing was in episode order, which
made a slow episode look like a stall; it now prints as episodes complete.

**Inkling-Small, full s98_v1 run** (`results/full/agent_inkling_small_s98_v1.json`,
renderer `tml_v0`, temperature 0.6, seed 0, eight workers, no errors): completeness
0.40 exact at a false-alarm rate of 0.19, mean score 0.52, about 20,000 tokens an
episode, estimated $6. For comparison the BLS baseline on the same set gives 0.32 at a
false-alarm rate of 0.03 (threshold 18.6) or 0.39 at 0.07 (threshold 13). The agent
trades false alarms for completeness; a fair comparison needs the SNR-binned table from
`stargazer compare` once the other two runs finish. The run's log lost its last lines
but the results file holds all 200 episodes.

**Tools for discovery** (commit `c673f8d`). `bls_peaks` takes `masks`, a list of
`{period, t0, duration}` signals to remove before the search, so a dominant dip cannot
hide a weaker periodic one (the ep_0001 failure; a test shows a 0.2 percent, 7-day
signal appearing once a 1 percent, 3-day signal is masked). New tool `single_events`
scores boxes of several widths at every time by the single-event statistic
``depth / (sigma / sqrt(n))``, skips boxes across gaps, and returns the strongest
non-overlapping dips; a lone 1 percent event at day 20 is found at SNR above 20 and
pure noise stays below 8. The prompt procedure now has steps for masking and for
single events. The validator accepts array and object arguments.

**Episode set s98_v2.** Periods 0.5 to 60 days, seed 2, otherwise as v1, so single and
double transits are common. Build and baseline running.

**Full-run comparison on s98_v1, old tools** (runs launched 07:47, before the masked
search and single_events existed; prompt and tools as of commit `2b4ec62`).

| system | false alarm | completeness | mean score | tokens/episode | s/episode |
|---|---|---|---|---|---|
| BLS, SDE 7 | 69% | 42% | 0.393 | - | 3 |
| BLS, SDE 11 | 28% | 41% | 0.502 | - | 3 |
| BLS, SDE 13 | 7% | 39% | 0.545 | - | 3 |
| Inkling-Small | 19% | 40% | 0.522 | 21,283 | 29 |
| Qwen3.8-27B | 24% | 43% | 0.527 | 56,808 | 35 |

Completeness by expected SNR (7-10 / 10-20 / 20-50 / 50+): BLS at SDE 11: 11 / 48 / 44 /
86 percent; Inkling-Small: 11 / 33 / 44 / 91; Qwen3.8-27B: 22 / 38 / 52 / 91.

Reading: at a matched false-alarm rate the agents are level with the thresholded BLS
baseline. Qwen3.8-27B answered a wrong period 37 times out of 142
injections, Inkling 22 times. Both models used the same BLS tool the baseline uses, so
they could only choose among the peaks it offered; the result measures the tool, as
predicted. Qwen hit the turn cap on 6 episodes. Estimated cost: Inkling about $6,
Qwen about $60 (the runs predate the price table, so their files record no cost).

Next: the same two models on the same set with the new tools (masked re-search and
single_events), launched 08:06 into `results/full_v2tools/`. That is the direct test
of whether the tools were the ceiling.

**Wrong-period answers are not aliases.** Of Qwen3.8-27B's 37 wrong periods on
s98_v1, only one is within 2 percent of a simple ratio (1.5x, 2/3x, 4x, 1/4x) of the
truth; the ratio of predicted to true period spans 0.07 to 8.5 with a median of 0.67.
The median expected SNR of these episodes is 7.6, and 15 of the 37 have five or fewer
transits. So the model picked an unrelated BLS peak on a faint injection. These
answers are false alarms that happen to fall on transit episodes. The real false-alarm
behaviour of the agents is therefore worse than the 19 to 24 percent measured on nulls
alone. No extra alias credit is warranted. The fix belongs in the tools and the
decision rule: the masked re-search, the single-event search, and a stricter rule in
the prompt that a peak with SDE near the noise level and no second line of evidence is
"none".

**Nemotron-3.5-Lightning-30B-A3B, full s98_v1 run, old tools**
(`results/full/agent_nemotron35_lightning_s98_v1.json`, renderer
`nemotron3_disable_thinking`, temperature 0.7, seed 0): completeness 0.11 at a
false-alarm rate of 0.26, mean score 0.29, 239,000 tokens an episode, 73 seconds an
episode. It reached the 30-turn cap on most episodes and so never submitted; the few
submissions it made were often right. The problem is behaviour. A lack of capability
does not explain the result. The model keeps folding and re-folding and does not
decide. The price is low at $0.50 per million tokens, but a quarter of a million
tokens for a non-answer is a high cost. To try next: a stricter stop rule in the prompt and a lower safety cap for this model, or
the thinking renderer so it plans before acting.

Three-family comparison on s98_v1 with the old tools, at the agents' own false-alarm
rates: Qwen3.8-27B 43 percent at 24 percent, Inkling-Small 40 percent at 19 percent,
Nemotron-3.5-Lightning 11 percent at 26 percent, BLS 41 percent at 28 percent.

**Episode set s98_v2 and its baseline.** Built with
`stargazer build --targets data/targets_s98.json --n 200 --seed 2 --max-period 60 --out data/episodes/s98_v2`,
hash `d03935c59b8706b8715c3363c917eef30d09c65ebdba46c654079a86b453d731`, verify OK.
127 injections with at least one transit: 18 with one, 14 with two, 22 with three to
five, 73 with six or more. BLS baseline (`results/baseline_bls_s98_v2_sde7.json`,
sweep in `results/calibrate_bls_s98_v2.json`): completeness 0.25 at the 5 percent
false-alarm threshold of 18.63, 0.29 at threshold 13 (false alarm 0.21). By transit
count at the default threshold: one transit 0 percent, two 14 percent, three to five
27 percent, six or more 51 percent. On this set `single_events` must show its value,
because BLS cannot find a single transit by construction.

**New tools on s98_v1: masked re-search and single_events** (runs launched 08:06,
commit `c673f8d` tools and prompt, `results/full_v2tools/`).

| system | tools | false alarm | completeness | mean score | wrong period | false alarms | tokens/ep | cost |
|---|---|---|---|---|---|---|---|---|
| Inkling-Small | old | 19% | 40% | 0.522 | 22 | 11 | 21,283 | ~$6 |
| Inkling-Small | new | 5% | 36% | 0.535 | 9 | 3 | 31,295 | $9 |
| Qwen3.8-27B | old | 24% | 43% | 0.527 | 37 | 14 | 56,808 | ~$60 |
| Qwen3.8-27B | new | 21% | 41% | 0.528 | 24 | 12 | 67,736 | $76 |
| BLS at 5% false alarm (SDE 18.6) | - | 3% | 32% | 0.505 | - | - | - | - |
| BLS at SDE 11 | - | 28% | 41% | 0.502 | - | - | - | - |

Reading. The new tools raised precision. Inkling-Small's wrong periods fell from 22
to 9 and its false alarms from 11 to 3, so at a 5 percent false-alarm rate it now finds
36 percent against the baseline's 32 percent at the same rate. This is the first agent
result above the thresholded BLS at a matched false-alarm rate.
Qwen3.8-27B moved less and hit the 30-turn cap on 16 episodes, up from 6; more tools
gave it more to explore. Both models used the masks heavily (177 and 352 masked
searches) and called single_events on most episodes. Completeness did not improve,
in the SNR 7 to 20 bins or overall. The masked search removes wrong answers, but it does
not surface the faint true signal. The faint regime therefore stays open.

**Inkling-Small on s98_v2 with the new tools**
(`results/full_v2tools/agent_inkling_small_s98_v2.json`): completeness 0.32 at a
false-alarm rate of 0.11, mean score 0.53, $9. The BLS baseline on the same set gives
0.29 at 0.21 (SDE 13) and 0.25 at 0.05. By transit count: one transit 0 of 18, two
1 of 14, three to five 18 percent, six or more 52 percent.

**Benchmark finding.** single_events was called on 169 of 200 episodes and found the
lone events, but no single-transit episode scored, because a submission needs a period
and a lone event has none. The scorer cannot reward the result that the tool finds. A
single-transit candidate is a real astronomical result (the mission pipelines publish
them as such), so the scorer needs a second submission form: `detected=true` with an
event time, duration and depth and no period, credited when the event time falls on an
injected transit. Without it, every system scores zero on the regime the project cares
most about. This is the first item for the next session, ahead of more tools.

**Single-event submission form** (scorer, submit tool, prompt). A prediction with
`detected=true` and no period now claims one event at `t0`. The scorer folds `t0` on the
true period with the same epoch check as the periodic form and sets `event_ok`. A
credited event scores 1.0 if the episode has one transit in the data and
`single_event_credit` (default 0.5, the alias credit) if it has two or more, because
then a period was there to find. An event off every transit is `wrong_event`, score 0;
on a null episode the form is a false alarm. `completeness_with_aliases` and the SNR and
transit-count bins now count every episode with a score above zero, so a credited event
counts as found; `completeness_exact` adds an event hit with score 1.0 to the period
hits. No earlier results file changes under the new definitions, because no earlier
submission used the form. `evaluate` records `single_event_credit` in the results
config.

The submit tool checks the form: a periodic detection needs `t0`; a single-event
detection needs `t0`, `duration` and `depth`; an incomplete submission returns an error
and the episode continues. The `single_events` description and the prompt (scoring
line, step 6, decision rules) now tell the model to submit a lone event with snr above
about 10 and no partner, where before they told it to stay at `detected=false`. This
is a prompt and tool change: agent runs from here are not comparable with
`results/full/` and `results/full_v2tools/`; the next Inkling-Small run on s98_v2 goes
to `results/full_v3form/` and is the baseline of the new form. Tests: six scorer tests,
three submit-tool tests, one viewer test, one evaluate test; 248 tests pass offline.

**Three-episode checks of the single-event form, Inkling-Small on s98_v2** (renderer
`tml_v0`, model-card sampling, seed 0).

```
stargazer agent --episodes data/episodes/s98_v2 --backend tinker --model thinkingmachines/Inkling-Small \
    --renderer tml_v0 --limit 3 --out results/smoke/agent_inkling_small_s98_v2_form3.json
```
A first attempt without `--renderer` failed on every episode with `Expected single
token for <|im_end|>` because the default renderer is the Qwen3 one; the flag is part
of the command for this model. With the flag: ep_0001 hit (P 0.7764 d, 9 calls), ep_0000
and ep_0002 true null, no errors, $0.13. None of the three has a single transit, so a
second run took the first three one-transit episodes through `evaluate` with
`ids=["ep_0004", "ep_0007", "ep_0012"]`, same backend and settings
(`results/smoke/agent_inkling_small_s98_v2_form3_single.json`, $0.14, log beside it).

| episode | snr | depth ppm | duration h | single_events (model's window) | submission | outcome |
|---|---|---|---|---|---|---|
| ep_0004 | 15.6 | 2323 | 5.7 | top event t 0.16 d snr 9.8, true event absent (window 0.3) | detected=false | miss |
| ep_0007 | 25.5 | 4994 | 3.3 | t 57.08 d snr 17.9 (window 0.75) | single event t0 57.083 d | event_hit, 1.0 |
| ep_0012 | 35.3 | 4681 | 5.8 | top event t 32.44 d snr 8.1, true event snr 5.1 (window 0.3) | detected=false | miss |

The form works end to end: the model called single_events, found the one event on
ep_0007, submitted it with t0, duration and depth and no period, the tool accepted the
form, the scorer credited it and the viewer shows `single event t0=57.083 d`. This is
the first scored single-transit detection in the project.

**Finding: the detrending window erases long single events.** On ep_0004 and ep_0012
the model chose `detrend_window_days` 0.3 for an active star, as the bls_peaks
description suggests. A 0.3-day (7.2-hour) running window removes most of a 5.7-hour
dip. Offline, `single_events` on ep_0012 gives the true event snr 5.1 at window 0.3,
13.7 at 0.75 and 53 at 1.5 (where stellar variability also enters); on ep_0004 the
true event is absent at 0.3, snr 5.4 at 0.75 and 10.0 at 1.5. The single-event search
therefore needs a detrending that does not see the box it tests: a window tied to each
box width (three times the width or more) or a fit that masks the box. This is a tool
change for the next session, before the full s98_v2 run, since 9 of the 18 one-transit
injections in s98_v2 are longer than 4 hours.

**single_events: detrending tied to the box width, noise at the box timescale.**
Each box width ``d`` now searches its own detrended series with window
``max(detrend_window_days, 6 d)``, and the statistic divides by the noise at the box
timescale, 1.4826 times the median absolute deviation of the box means, in place of
``sigma / sqrt(n)``. Each event reports that noise as ``noise_ppm``. The reasons, from
the offline scan on the three one-transit episodes (true-event snr at window 0.3):

| factor, noise | ep_0004 (5.7 h) | ep_0007 (3.3 h) | ep_0012 (5.8 h) | top spurious event |
|---|---|---|---|---|
| 3, point scatter | 5.4 | 11.9 | 12.7 | 53 (12-hour box on ep_0012) |
| 6, point scatter | 10.0 | 17.9 | 13.7 | 240 (12-hour box on ep_0012) |
| 3, box timescale | 8.3 | 18.9 | 8.4 | 18 |
| 6, box timescale | 10.4 | 18.5 | 8.5 | 12 |

A wider window keeps more of a long dip (a Savitzky-Golay fit at three widths still
halves a 6-hour dip on a real star), but with the point scatter as the noise, a wide
window on an active star turns slow variability into events with an snr in the
hundreds. The box-timescale noise removes those and keeps the true events, so the
tool now gives the same events whichever window the model picks. On ep_0012 the true
event sits at 8.5 because the star's variability at the 6-hour timescale is real
noise; the earlier 13.7 was an underestimate of that noise. Test:
``test_single_events_keeps_a_long_event_under_a_short_window`` (a 6-hour, 2000 ppm
event under a 0.3-day window, where the plain filter keeps under a quarter of the
depth). A tool change: runs from here are the `full_v3form` series.

**Three one-transit episodes again with the fixed tool**
(`results/smoke/agent_inkling_small_s98_v2_form3_single_fix.json`, $0.14). ep_0004:
single_events now lists the true event first (snr 10.4, 1455 ppm); the model
declined, citing an active star and the snr near the rule's 10. ep_0007: event hit
again, score 1.0. ep_0012: the true event is third at snr 8.5; declined. One event
hit of three, as before, but now the tool surfaces the signals and the decision is
the model's. The full run is next.

**Inkling-Small, full s98_v2 run with the single-event form and the first
single_events fix** (`results/full_v3form/agent_inkling_small_s98_v2.json`, log
beside it; renderer `tml_v0`, temperature 0.6, seed 0, eight workers, no errors,
925 s, $10.15, 35,249 tokens an episode).

```
stargazer agent --episodes data/episodes/s98_v2 --backend tinker --model thinkingmachines/Inkling-Small \
    --renderer tml_v0 --limit 0 --workers 8 --out results/full_v3form/agent_inkling_small_s98_v2.json
```

| run | false alarm | exact | with aliases | mean score | one transit | two | 3-5 | 6+ |
|---|---|---|---|---|---|---|---|---|
| v2tools (period form only) | 0.11 | 0.32 | 0.34 | 0.533 | 0/18 | 1/14 | 18% | 52% |
| v3form | 0.33 | 0.31 | 0.32 | 0.445 | 2/18 | 2/14 | 27% | 42% |

The form did what it was built for: two one-transit episodes scored (ep_0029 and
ep_0079, events at snr 138 and 370), and the two-transit and three-to-five bins rose.
It also opened a false-alarm channel. The model made 22 single-event submissions: 2
hits, 7 on transit episodes but off the transit (`wrong_event`), 13 on null
episodes. The 13 null false alarms and the 7 wrong events are the same six event
times repeated across episodes of the same host star (t = 0.04, 0.27, 24.05, 14.57,
15.07, 21.18 days), each within 0.14 day of the start or the end of a data segment.
The Savitzky-Golay filter fits a one-sided window at a segment edge and leaves a
dip-shaped residual, which the box search then reports with an snr of 12 to 33. The
periodic form also moved: false alarms 8 to 11, wrong periods 6 to 12, and the
six-plus bin fell from 52 to 42 percent; with one seed the share of variance in that
is unknown.

**single_events: segment-edge guard.** A box now has to lie inside one segment (a
gap longer than five cadences splits segments), at least three box widths from the
segment's start and end, in a segment at least two detrending windows long. The
margin came from a scan over one to four widths on the seven repeated false events
and the five true events above: at three widths five of the seven false events were
gone and every true event stayed; the box-in-one-segment rule removed the sixth
(a 12-hour box across a 0.2-day gap); the segment-length rule removed the last (a
one-hour box in a 7.5-hour segment). After the guard the true event is the top
event on ep_0004 (snr 11.4), ep_0007 (15.5) and ep_0012 (9.0), with the same list
at windows 0.3 and 0.75. Test: `test_single_events_skips_the_edge_of_a_segment`
(an exponential ramp at the data start and after the gap is reported without the
guard and not with it). Model check on ep_0032, ep_0038, ep_0042 (the three
repeated null false alarms) and ep_0004
(`results/smoke/agent_inkling_small_s98_v2_edge_fix.json`, $0.18): two true nulls, one
periodic false alarm at a 12.08-day noise peak (the prompt-rule item in the PLAN),
ep_0004 declined again at snr 11.4. The full run with the guard is
`results/full_v3form/agent_inkling_small_s98_v2_edge.json`, launched after the check.

**Prompt: depth and variability rules for a single event.** Offline, the strongest
single event on the 69 null episodes of s98_v2 with the guarded tool: median snr 5.1,
90th percentile 10.4, six above 20. Those six are 18 to 44 percent deep on stars with
a flux range of 20 to 55 percent (TIC 287841093, 231822771 and 9380768), which are
stellar eclipses or large-amplitude variables. The true event of each one-transit
injection above expected snr 15 sits at tool snr 9 to 18 (ep_0079 and ep_0029 at 376
and 161). An snr rule alone cannot separate the two groups, so the prompt's decision
rules now add: a single event needs a depth below 5 percent, and on a star whose flux
range exceeds 5 percent a lone dip needs a partner. Prompt change; the comparable
series stays `full_v3form`, which the edge-guard run already belongs to with the
older prompt (its config records the prompt hash).

**Tool: `search_signals`, the multiple-planet search.** One call runs bls_peaks, takes
the strongest peak, masks it and searches again, until the strongest remaining peak
has sde below `min_sde` (default 10, where noise peaks on real stars end) or
`max_signals` (default 3) are found. It returns the signals in order with the
bls_peaks fields, `residual_sde` (the strongest peak left, so the model can see a
third signal waiting near the threshold) and `stop`. Reference: Jenkins et al. 2010,
the Kepler transiting planet search. The prompt procedure now calls it at step 2 in
place of bls_peaks and drops the manual mask step; bls_peaks stays available. Tests: a
1 percent 3-day and a 0.2 percent 7-day signal come back as ranks 1 and 2; noise
returns no signal with the residual below 10; `max_signals` 1 stops after one. The
viewer summarises the result. Tool change: the next full runs are the `full_v4search`
series.

**Three-episode check of search_signals**
(`results/smoke/agent_inkling_small_s98_v2_search3.json`, Inkling-Small, `tml_v0`,
$0.11). ep_0001 hit (one signal at 0.776 days, sde 23.6, residual 8.9, folded and
odd_even consistent, 8 calls); ep_0000 true null (one signal at sde 10.9, folded to
a noisy dip, declined); ep_0002 true null (zero signals, residual 7.7, 6 calls). The
model used search_signals at step 2 on every episode and still called bls_peaks once
for its own check. No errors. The full run is `results/full_v4search/`.

**Inkling-Small, full s98_v2 run with the segment-edge guard**
(`results/full_v3form/agent_inkling_small_s98_v2_edge.json`, same command as the
v3form run with `_edge` in the output name; no errors, 879 s, $9.65, 33,513 tokens
an episode).

| run | false alarm | exact | with aliases | mean score | one transit | two | 3-5 | 6+ | event submissions |
|---|---|---|---|---|---|---|---|---|---|
| v2tools (period form only) | 0.11 | 0.32 | 0.34 | 0.533 | 0/18 | 1/14 | 4/22 | 38/73 | 0 |
| v3form | 0.33 | 0.31 | 0.32 | 0.445 | 2/18 | 2/14 | 6/22 | 31/73 | 22: 2 hits, 7 wrong, 13 false |
| v3form + edge guard | 0.12 | 0.28 | 0.34 | 0.515 | 2/18 | 2/14 | 6/22 | 33/73 | 9: 6 hits, 2 wrong, 1 false |
| BLS, SDE 18.6 | 0.05 | 0.25 | - | - | 0/18 | 2/14 | 27% | 51% | - |

The guard closed the channel: event-form false alarms fell from 13 to 1 and wrong
events from 7 to 2. The one left (ep_0086, t = 33.49 days) is the same feature that
gave the wrong event on ep_0060, a real dip of one host star that is not at a
segment edge; only a vetting tool can reject it. Six event hits: two one-transit
episodes (ep_0029, ep_0079), two with two transits, one with three and one with
five, where the periodic search had missed. The false-alarm rate is back at the
v2tools level with the same alias-inclusive completeness, and the one-transit bin
went from 0 to 2 of 18. Exact completeness is lower (0.28 against 0.32) because an
event hit on a multi-transit episode scores 0.5 and does not count as exact, and
because the period form found 34 hits against 41; with one seed the second
difference is within what a seed can move. The 16 one-transit misses are the faint
regime: the model declines events near snr 10 on active stars, which is the rule
the prompt gives it.

**Tool: `fit_transit`, the physical transit fit.** Folds the detrended flux at a
period and epoch, keeps 2.5 durations around the transit, bins to 5 minutes and
fits the batman model of the injector by least squares with the epoch offset, the
radius ratio, a/R* and the impact parameter free; limb darkening fixed at the Claret
TESS values for 5500 K, because the tool sees only a light curve. Returns the depth
with an error from the covariance, the duration, a refined t0, a/R*, the impact
parameter, the host density from the period and a/R* (Seager and Mallén-Ornelas
2003) with a verdict (plausible for 0.1 to 20 g/cc), the shape (flat_bottom for
b + k below 0.9, else v_shaped) and the significance over a flat line. The shape
bound sits below 1 because the fit of a near-grazing dip trades b against k: a
b=0.98, k=0.10 injection fitted to b=0.915, k=0.065. Tests: a 1 percent b=0.3
injection returns depth, duration and a/R* within 20 percent, flat_bottom and a
plausible density; b=0.98 returns v_shaped; noise gives a significance below 3
sigma; the density of the Sun-like test host comes back near 1.5 g/cc. The prompt
puts fit_transit between the folds and odd_even and requires flat_bottom with a
plausible density. The viewer summarises the result. Tool and prompt change, so the
next full runs after the `full_v4search` series are `full_v5fit`.

**Three-episode check of fit_transit**
(`results/smoke/agent_inkling_small_s98_v2_fit3.json`, Inkling-Small, `tml_v0`,
$0.14, no errors). ep_0000 true null: the one signal at sde 11 fitted v_shaped with a
density of 0.013 g/cc, verdict too_low_for_a_dwarf, and the model declined it for
that reason. ep_0001 hit: flat_bottom, density plausible, 33 sigma, odd_even
consistent, submitted at 0.7764 days. ep_0002 true null: no signal. The model called
fit_transit between the fold and odd_even on each candidate, as the procedure says.

**Tool: `secondary_eclipse`, the Robovetter's secondary test.** Folds the detrended
flux at a period and epoch, measures the box-mean depth at phase 0.5 and at 200
trial phases, and takes the noise as 1.4826 times the median absolute deviation of
the trial depths outside the primary window, so stellar variability at the transit
timescale is in it. Returns the primary and secondary depths, the secondary's sigma,
the strongest dip at any other phase with its sigma, and a verdict:
secondary_at_half_phase (3 sigma at phase 0.5, an eclipsing binary),
dip_at_other_phase (5 sigma elsewhere, a wrong period or a second signal), else
no_secondary. Reference: Coughlin et al. 2016. Tests: a primary with a 0.4-depth
eclipse half an orbit later gives the binary verdict; a planet alone gives
no_secondary; a second dip at phase 0.3 gives dip_at_other_phase at that phase. The
prompt adds it after odd_even and requires no_secondary. The viewer summarises it.
Tool and prompt change; the `full_v5fit` series includes it.

**Three-episode check of secondary_eclipse**
(`results/smoke/agent_inkling_small_s98_v2_secondary3.json`, Inkling-Small, `tml_v0`,
$0.14, no errors). ep_0001 hit, verdict no_secondary (phase 0.5 at -0.7 sigma);
ep_0000 and ep_0002 true null. On ep_0000 the tool reported dip_at_other_phase at
phase 0.03 with 18 sigma: that is the detrending residual beside the 16.9-day
candidate's own dip, 0.5 days from it, inside the one-duration exclusion the first
version used. The exclusion around the primary is now three durations on each side,
and the detrending window is at least six durations, as in single_events, because
the model's 0.3-day window on a 6-hour box had flattened the fold to a noise of 33
ppm and made a 370 ppm residual look like 11 sigma. With both changes the noise on
ep_0000 is 149 ppm and the strongest other dip is 745 ppm at phase 0.195, 5.0 sigma,
still dip_at_other_phase at the threshold; on a null episode with three orbits of an
active star that is the honest reading, and the model must not submit it. The tests
are unchanged; the next series carries the corrected tool.

**Inkling-Small, full s98_v2 run with search_signals and the single-event prompt rules**
(`results/full_v4search/agent_inkling_small_s98_v2.json`, log beside it; renderer
`tml_v0`, temperature 0.6, seed 0, eight workers, no errors, 807 s, $7.86, 27,286
tokens and 8.0 tool calls an episode; tools and prompt of commit `2683ebb`, so
without fit_transit and secondary_eclipse).

```
stargazer agent --episodes data/episodes/s98_v2 --backend tinker --model thinkingmachines/Inkling-Small \
    --renderer tml_v0 --limit 0 --workers 8 --out results/full_v4search/agent_inkling_small_s98_v2.json
```

| run | false alarm | exact | with aliases | mean score | one transit | two | 3-5 | 6+ | calls/ep | cost |
|---|---|---|---|---|---|---|---|---|---|---|
| v2tools | 0.11 | 0.32 | 0.34 | 0.533 | 0/18 | 1/14 | 4/22 | 38/73 | 9.7 | $9.16 |
| v3form + edge guard | 0.12 | 0.28 | 0.34 | 0.515 | 2/18 | 2/14 | 6/22 | 33/73 | 9.6 | $9.65 |
| v4search | 0.11 | 0.32 | 0.35 | 0.538 | 2/18 | 2/14 | 5/22 | 36/73 | 8.0 | $7.86 |

Outcomes: 39 hits, 1 alias, 10 wrong periods, 5 event hits, 0 wrong events, 72
misses, 65 true nulls, 8 false alarms. Five single-event submissions, all hits.
Reading. The one-call search and the two prompt rules kept the gains of the event
form (five event hits, the one-transit bin at 2 of 18) and removed its costs: the
false-alarm rate is back at the v2tools level and no event was submitted at a wrong
time. The mean score is the highest of the series, with 17 percent fewer tool calls
and 14 percent less cost an episode. Wrong periods are 10 against 6 and 8; with one
seed that spread is within what a seed moves. The faint regime is unchanged: the 72
misses are the injections below expected snr 20, where the search statistic does
not reach the noise ceiling.

**Course notebooks, first three.** `scripts/build_notebooks.py` generates and
executes the notebooks in `examples/` with nbformat and nbclient, so the text and
code are reviewed as Python and the built files carry their plots. Each notebook
loads ep_0001 of s98_v2 if the set is present and a synthetic light curve with one
injection otherwise. Written: 01 the episode and summarize, 02 injection with the
shape against the impact parameter, 03 the period search with detrending,
bls_peaks, search_signals and the fold. New optional dependency group `notebooks`
(matplotlib, jupyter, nbformat, nbclient, ipykernel).

**Course notebooks 04 and 05.** 04 vets the strongest signal of ep_0001 with fold,
fit_transit (with the fitted model drawn over the binned fold), odd_even and
secondary_eclipse (with the trial depths across the orbit). 05 injects one 3000 ppm
transit at day 25 into the same episode and shows the single-event statistic and
the event in the detrended flux. Two findings from writing them: the fit's depth
error on ep_0001 is 2066 ppm for a 1284 ppm depth, because the covariance of a
near-degenerate b and k pair is wide, so the error is honest and not useful as a
detection statistic; and on a host with a strong periodic signal the noise at the
6-hour box timescale is 420 ppm, so the 6-hour injected event is found by the
1-hour box at snr 9 and not by its own width.

**Inkling-Small, full s98_v2 run with fit_transit and secondary_eclipse**
(`results/full_v5fit/agent_inkling_small_s98_v2.json`, log beside it; renderer
`tml_v0`, temperature 0.6, seed 0, eight workers, no errors, 819 s, $9.68, 33,625
tokens and 8.4 tool calls an episode; tools and prompt of commit `6f9b513`).

```
stargazer agent --episodes data/episodes/s98_v2 --backend tinker --model thinkingmachines/Inkling-Small \
    --renderer tml_v0 --limit 0 --workers 8 --out results/full_v5fit/agent_inkling_small_s98_v2.json
```

| run | false alarm | exact | with aliases | mean score | one transit | two | 3-5 | 6+ | wrong period | calls/ep | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v2tools | 0.11 | 0.32 | 0.34 | 0.533 | 0/18 | 1/14 | 4/22 | 38/73 | 6 | 9.7 | $9.16 |
| v3form + edge guard | 0.12 | 0.28 | 0.34 | 0.515 | 2/18 | 2/14 | 6/22 | 33/73 | 8 | 9.6 | $9.65 |
| v4search | 0.11 | 0.32 | 0.35 | 0.538 | 2/18 | 2/14 | 5/22 | 36/73 | 10 | 8.0 | $7.86 |
| v5fit | 0.03 | 0.35 | 0.37 | 0.583 | 1/18 | 2/14 | 4/22 | 40/73 | 4 | 8.4 | $9.68 |
| BLS at 5 percent false alarm | 0.05 | 0.25 | - | - | 0/18 | 2/14 | 27% | 51% | - | - | - |

Outcomes: 43 hits, 1 alias, 4 wrong periods, 3 event hits, 76 misses, 71 true nulls,
2 false alarms. The model called fit_transit on 154 episodes and secondary_eclipse on
126.

Reading. The two vetting tools did what they are for. False alarms fell from 8 to 2,
wrong periods from 10 to 4, and the mean score is the highest of the series at 0.583,
with exact completeness above the earlier runs and far above the BLS baseline at a
lower false-alarm rate. The two false alarms are one host (TIC in the results file)
with a real 0.946-day signal, fitted flat_bottom and plausible at 7 sigma with no
secondary, on four episodes of that host (two counted as wrong periods); that is a
signal in the star, not a harness error, and a known cost of injection into real
hosts.

Finding. The one-transit bin fell from 2 to 1 because ep_0029 and ep_0079, the two
bright lone transits, were submitted as periods. The search paired each real dip
with an epoch inside a gap: bls_peaks reported `n_events_in_data` 2 from floor
arithmetic, the fold showed one clean dip, fit_transit said flat_bottom and plausible
at 35 sigma, and odd_even said insufficient_data with one event of one parity and
none of the other, which the model did not read as a warning. The count is now the
number of epochs with at least three points in transit, so both episodes report 1,
and the prompt says that a signal with one event is a single event and that
odd_even with one event and none of the other parity means the same. A test covers
an epoch inside the gap. Tool and prompt change: the next series is `full_v6count`.

**Three-episode check of the event count by points in transit**
(`results/smoke/agent_inkling_small_s98_v2_count3.json`, Inkling-Small, `tml_v0`,
ids ep_0029, ep_0079, ep_0004, through `evaluate`). All three are event hits, score
1.0 each. On ep_0029 and ep_0079 the model read n_events_in_data 1 and odd_even's
one event against none, called single_events and submitted the lone dip. On ep_0004,
the faint one-transit injection (expected snr 16) that the model declined in two
earlier checks, it submitted the event at snr 11.4 for the same reasons. The full run
is `results/full_v6count/`.

**Inkling-Small, full s98_v2 run with the event count by points in transit**
(`results/full_v6count/agent_inkling_small_s98_v2.json`, log beside it; renderer
`tml_v0`, temperature 0.6, seed 0, eight workers, no errors, 799 s, $9.60, 33,341
tokens and 8.2 tool calls an episode; tools and prompt of commit `db99a6e`).

```
stargazer agent --episodes data/episodes/s98_v2 --backend tinker --model thinkingmachines/Inkling-Small \
    --renderer tml_v0 --limit 0 --workers 8 --out results/full_v6count/agent_inkling_small_s98_v2.json
```

| run | false alarm | exact | with aliases | mean score | one transit | two | 3-5 | 6+ | wrong period | hits |
|---|---|---|---|---|---|---|---|---|---|---|
| v5fit | 0.03 | 0.35 | 0.37 | 0.583 | 1/18 | 2/14 | 4/22 | 40/73 | 4 | 43 |
| v6count | 0.05 | 0.33 | 0.35 | 0.558 | 3/18 | 2/14 | 4/22 | 36/73 | 2 | 39 |

Outcomes: 39 hits, 3 aliases, 2 wrong periods, 3 event hits, 2 wrong events, 78
misses, 69 true nulls, 4 false alarms. Gained against v5fit: ep_0004, ep_0029 and
ep_0079, the three one-transit episodes of the check, all event hits, so the
one-transit bin is 3 of 18, the best so far, and wrong periods are 2. Lost: five
hits became misses, ep_0021, ep_0046, ep_0061, ep_0072 and ep_0107, with expected snr
16 to 142 and 24 to 66 transits each. The four false alarms are two hosts (TIC
146420104 at 15.37 days, TIC 121711742 at 28.11 days) with the same answer on both
of their episodes, real features of those stars. The two wrong events are an 18
percent eclipsing binary submitted as an event against the depth rule (ep_0145) and
an edge-time t0 on ep_0172.

Finding. The five lost hits share one cause: fit_transit at the true period returned
`v_shaped`, with b + k between 0.90 and 1.00, on transits of 1.2 to 1.7 hours at
periods of 0.5 to 1.6 days, and the model obeyed the rule. Over both runs, 6 of 98
fits at a true period are v_shaped, against 40 of 106 fits at a wrong period or on
a null, so the verdict separates the two groups but its 0.9 bound sits inside the
spread of true planets with short transits, where six 5-minute bins cannot fix b.
Four of the five were hits in v5fit with the same tools, so the model's reading of
a v_shaped verdict varies from seed to seed; the verdict needs a middle word.

**fit_transit: three-way shape verdict.** `shape` is now flat_bottom below b + k of
0.9, grazing_uncertain from 0.9 to 1.0, and v_shaped at 1.0 and above
(`SHAPE_VERDICTS`). In the v6count run the fits at a wrong period or on a null fell
66, 26 and 14 into the three bands, and the fits at a true period 92, 6 and 0, so the
middle band holds both true short transits and wrong answers, and the model must
decide it by odd_even, secondary_eclipse and the depth; the prompt says so and no
longer requires flat_bottom. Offline, the five lost hits now read grazing_uncertain
(ep_0046 at 0.92, ep_0107 at 1.00, ep_0061 at 0.90). The grazing test accepts either
of the two upper verdicts. Tool and prompt change: the next series is `full_v7shape`.

**Two system prompts: base and rules.** The owner asked whether the prompt had become
a hand-written decision agent. It had: 49 lines, a seven-step procedure and six
numeric rules, each from one trajectory of one model. The prompt is now split.
`base` (15 lines) states the task, the scoring, that the tools exist and return
verdict words, the budget and the units, and leaves the method to the model.
`rules` is the full text as it stood. `stargazer agent --prompt base|rules` selects
one, default base, and the results config records the name and the hash. The gap
between the two on one model is the value of the hand-written policy; that is what
the reinforcement-learning phase must learn from the reward, so from here the rules
block is frozen and new knowledge goes into tools as verdicts. The `full_v7base`
series starts with Inkling-Small on s98_v2 with the base prompt.

**Three-episode check of the base prompt**
(`results/smoke/agent_inkling_small_s98_v2_base3.json`, Inkling-Small, `tml_v0`,
$0.15, no errors). With no procedure in the prompt the model called summarize,
bls_peaks, single_events, search_signals, fold, fit_transit, odd_even and
secondary_eclipse on its own, in an order of its choosing. ep_0001 hit (12 calls),
ep_0002 true null (6 calls), ep_0000 false alarm: the 16.9-day one-dip signal
submitted as a period, which the rules prompt's one-event rule had blocked. That is
the expected shape of the gap. The full run is `results/full_v7base/`.

**Inkling-Small, full s98_v2 run with the base prompt**
(`results/full_v7base/agent_inkling_small_s98_v2.json`, log beside it; `--prompt
base`, renderer `tml_v0`, temperature 0.6, seed 0, eight workers, no errors, 851 s,
$9.43, 32,734 tokens and 8.9 tool calls an episode; tools of commit `1c8ee67`, the
same as the rules row below except the three-way shape verdict).

```
stargazer agent --episodes data/episodes/s98_v2 --backend tinker --model thinkingmachines/Inkling-Small \
    --renderer tml_v0 --prompt base --limit 0 --workers 8 --out results/full_v7base/agent_inkling_small_s98_v2.json
```

| prompt | false alarm | exact | with aliases | mean score | one transit | two | 3-5 | 6+ | wrong period | hits |
|---|---|---|---|---|---|---|---|---|---|---|
| rules (v6count) | 0.05 | 0.33 | 0.35 | 0.558 | 3/18 | 2/14 | 4/22 | 36/73 | 2 | 39 |
| base (v7base) | 0.40 | 0.39 | 0.43 | 0.476 | 5/18 | 3/14 | 7/22 | 40/73 | 21 | 44 |

Outcomes with the base prompt: 44 hits, 6 aliases, 21 wrong periods, 5 event hits, 3
wrong events, 48 misses, 44 true nulls, 29 false alarms. The model used every tool
without being told the order (bls_peaks 231 calls, search_signals 148, fit_transit
222, secondary_eclipse 150).

Reading. The gap between the two prompts is precision. Without the rules the model
finds more, in every transit-count bin, and submits seven times as many false
alarms and ten times as many wrong periods, so the mean score falls from 0.558 to
0.476. The rules block is worth 0.08 of mean score on this model and this set, and
all of it is restraint: when to say "none". That is the policy the
reinforcement-learning phase has to learn from the reward, and the base row is its
starting point. The base prompt is the default from here; the rules prompt stays as
the comparison row.

**Catalogue context and two vetting tools.** `stargazer/context.py` holds
`EpisodeContext`: the star's TIC, position, TESS magnitude, the episode's time
offset, its TOI entries (ExoFOP CSV), its TESS eclipsing-binary entries (Prša et al.
2022, Vizier J/ApJS/258/16), its Gaia DR3 variability classes (I/358/vclassre) and
the Gaia sources within 63 arcseconds with their G magnitude difference (I/355),
with the download time and the TOI file's hash. `stargazer context --episodes DIR
--targets FILE` fetches one context for each host and writes `context.json` beside
the episodes, then rewrites the manifest. The context carries nothing about the
injection; the writer reads the truth file only for the TIC, the sector and the time
offset. A detector that declares a `context` keyword on `detect` receives it; the
baseline does not and is unchanged. Tools that need it are marked and return an
error on a set without `context.json`.

Two tools read it. `ephemeris_match(period, t0)` compares the candidate with the
TOI and binary periods at the ratios 1, 2, 1/2, 3 and 1/3 within 1 percent, reports
the epoch offset against the catalogue epoch, and gives the verdict known_toi,
known_eclipsing_binary, gaia_variable or no_match. `nearby_stars()` lists the Gaia
sources within two pixels with their flux fraction and the largest depth their total
eclipse could make on the target, verdict possible_blend or isolated. Both are in
the tool registry; the prompts do not mention them, so the model finds them from
their descriptions. Tests are offline with fixed contexts.

`data/episodes/s98_v2` now has `context.json` for its 200 episodes. The episode
files are identical; the manifest hash moved from
`d03935c59b8706b8715c3363c917eef30d09c65ebdba46c654079a86b453d731` to
`f73a513c1687c338c3e1b8d60ed9e32883ff68694f7946ec2fb39d0ecb85a490` because the
manifest lists the new file. Results files before this point record the old hash.

**Three-episode check of the catalogue tools**
(`results/smoke/agent_inkling_small_s98_v2_catalogue3.json`, Inkling-Small,
`tml_v0`, base prompt, ids ep_0025, ep_0054, ep_0005, hosts in the TESS
eclipsing-binary catalogue). The prompts do not name the two tools; the model
called ephemeris_match and nearby_stars on every episode from their descriptions.
ep_0025 true null: the 1.007-day peak matched the catalogued binary at 2.015 days
at ratio 0.5, and the model declined, naming the match. ep_0054 true null: the
17.707-day eclipse matched the catalogued 17.703-day binary. ep_0005 miss: the
injected 1.62-day planet sits on a host whose catalogued binary period is 1.622
days, so the injection fell on a real 25 percent eclipse, and the model declined the
whole signal as the binary.

Finding. 15 of the 200 episodes of s98_v2 sit on hosts in the TESS eclipsing-binary
catalogue, and 5 more carry the Gaia class ECL. The host selection excluded TOI
hosts and not binaries. An injection on such a host is unfindable by design, and
the hosts' own eclipses are the deep false alarms seen all day (TIC 287841093,
231822771, 9380768). The three remaining false-alarm hosts (TIC 146420104,
121711742, 55270064) have no catalogue entry and no bright neighbour within two
pixels, so their signals are either unknown or systematics; the pixel tools would
say which. PLAN gains the exclusion as a Phase 1 item for s98_v3.

## 2026-10-16

**Host selection excludes known eclipsing binaries.** The 2026-10-11 check found three
of the 39 hosts of s98_v2 in the TESS eclipsing-binary catalogue (TIC 9380768,
231822771, 287841093; the second also carries the Gaia DR3 class ECL). Their eclipses
were the deep false alarms of every run, and an injection on one of them cannot be told
from the star's own signal. `CatalogueFetcher.binary_evidence` in `stargazer/context.py`
asks the two catalogues for one star and returns the reasons (`tess_eb`, `gaia_ECL`).
`select_targets` applies it as the last cut, after the dwarf cuts, so the pool draw of a
seed is unchanged and the Vizier query runs only for stars that pass; the targets file
gains `rejected.binary`, `removed_binaries` and `binary_catalogues`. `remove_binaries`
applies the cut alone to an existing targets file and `stargazer remove-binaries` wraps
it. A new draw with `stargazer targets` would also follow the TOI list and the MAST
sector list, which have moved since 2026-10-11, so the kept hosts would not be the same
stars; the filter keeps them identical.

```
stargazer remove-binaries --targets data/targets_s98.json --out data/targets_s98_v3.json
```

Kept 38 of 41, removed the three TICs above with the evidence named, 11 seconds. The
targets file records `derived_from` (the input's creation time) and the catalogue
identifiers.

**Episode set s98_v3 and its baseline.** Built with

```
stargazer build --targets data/targets_s98_v3.json --n 200 --seed 3 --max-period 60 --out data/episodes/s98_v3
stargazer verify data/episodes/s98_v3
stargazer context --episodes data/episodes/s98_v3 --targets data/targets_s98_v3.json
```

36 hosts (two of the 38 lack TIC parameters, as before). Hash
`cf66edd19511d9f3a9782b8c4cf68bc98c4a40b0240c929e08267eac719621c1` after the build,
`745c6cb5750307bcc440cb15dc76cab474a4c40a14407fae6842272ec7c6de79` with
`context.json`; verify OK. The context of every episode has no TOI, no eclipsing-binary
entry and no ECL class. 58 null episodes and 142 injections, of which one has no
transit in the data and is scored as null; 141 with at least one transit: 17 with one,
15 with two, 30 with three to five, 79 with six or more. The null share is lower than
the 73 of s98_v2 by the seed, so false-alarm rates on this set carry wider error bars.

BLS baseline (`results/baseline_bls_s98_v3_sde7.json`, sweep in
`results/calibrate_bls_s98_v3.json`, same commands as for s98_v2 with the set name
changed): completeness 0.383 at the default threshold 7 with a 0.61 false-alarm rate;
at the 5 percent false-alarm threshold of 14.39, completeness 0.34. By transit count at
the default threshold: one transit 0 percent, two 0 percent, three to five 27 percent,
six or more 58 percent.

| SDE threshold | false-alarm rate | completeness exact | mean score |
|---|---|---|---|
| 11 | 0.254 | 0.355 | 0.470 |
| 13 | 0.085 | 0.348 | 0.515 |
| 15 | 0.000 | 0.333 | 0.530 |
| 18 | 0.000 | 0.291 | 0.500 |

Reading. On s98_v2 the false-alarm rate was still 0.07 at threshold 18 and the 5
percent threshold was 18.63; on s98_v3 it is zero from threshold 15 and the 5 percent
threshold is 14.39. The deep false alarms of s98_v2 were the binaries, as the check
said. With the lower threshold BLS keeps more of the real signals: completeness at 5
percent false alarm rises from 0.25 to 0.34, while the six-or-more bin at the default
threshold rises from 51 to 58 percent, in part because 79 instead of 73 injections fall
in it. This set is the reference for the seed runs; the Inkling-Small series on s98_v2
stays as the record of the toolbox and is not comparable with s98_v3 numbers.

**Three-episode check of Inkling-Small on s98_v3**
(`results/smoke/agent_inkling_small_s98_v3_base3.json`, `tml_v0`, base prompt, seed
0, $0.21, 43 seconds). The recorded config lists all eleven tools, the base prompt
hash and no errors. ep_0000 true null. ep_0002 hit on a 0.862-day planet with 46
transits. ep_0001 miss: the injected 8.1-day planet has an expected signal-to-noise
ratio of 1.6 at 154 ppm on a star with 2,234 ppm scatter, so it is unfindable; the
model ran eleven tools, saw a 16.7-day peak with two events fail vetting, and declined.
Harness clean; the six seed runs follow.

**Six seed runs launched** (`results/full_v8seeds/run_seeds.sh`, started
2026-10-16 about 19:40 UTC): Inkling-Small on s98_v3, `tml_v0`, `--workers 8`, seeds
1, 2, 3 with `--prompt base`, then seeds 1, 2, 3 with `--prompt rules`, one after
another. Results land as `results/full_v8seeds/agent_inkling_small_s98_v3_<prompt>_seed<n>.json`,
with `progress.log` beside them. The numbers go in the next entry.

**Six seed runs of Inkling-Small on s98_v3** (`results/full_v8seeds/`, hash
`745c6cb5...`, `tml_v0`, full toolbox, `--workers 8`, seeds 1, 2, 3 for each prompt,
$12 to $13 a run, $73.64 in all). The chain `run_seeds.sh` ran base seed 1 alone in
14 minutes; the owner asked for the rest side by side, so `run_parallel.sh` started
base seed 3 and the three rules seeds while base seed 2 was still running, five
processes at eight workers each. Tinker took it: 23 to 25 minutes a run, no errors,
200 episodes each. Every results file records its seed, its prompt hash and the set
hash; 22 of the 200 episodes differ in outcome between rules seeds 1 and 2, so the
seeds did change the sampling.

| prompt | seed | false alarm | completeness | with aliases | mean score | one-transit hits |
|---|---|---|---|---|---|---|
| base | 1 | 25.4% | 43.3% | 44.0% | 0.525 | 4 of 17 |
| base | 2 | 32.2% | 41.1% | 42.6% | 0.494 | 1 of 17 |
| base | 3 | 35.6% | 36.9% | 39.7% | 0.457 | 3 of 17 |
| base | mean (sd) | 31.1% (5.2) | 40.4% (3.3) | 42.1% (2.2) | 0.492 (0.034) | |
| rules | 1 | 3.4% | 39.7% | 41.8% | 0.573 | 2 of 17 |
| rules | 2 | 3.4% | 39.7% | 41.8% | 0.573 | 3 of 17 |
| rules | 3 | 0.0% | 38.3% | 40.4% | 0.573 | 3 of 17 |
| rules | mean (sd) | 2.3% (2.0) | 39.2% (0.8) | 41.4% (0.8) | 0.573 (0.000) | |

The three rules runs share the mean score 0.5725 by coincidence of totals: seed 3
has two more true nulls and two fewer hits than seed 1, and the partial credits add
to the same 114.5 points; the outcome counts differ (54, 53, 51 hits; 2, 2, 0 false
alarms; 3, 9, 10 wrong periods).

Reading. The spread of one run is now known. With the base prompt the mean score
moves by 0.07 between seeds and the false-alarm rate by 10 points; with the rules
prompt the score does not move and the false-alarm rate stays within 3.4 points. A
later change on this model has to move the base-prompt score by more than about 0.07,
or the rules-prompt false-alarm rate by more than 3 points, before it counts. The gap
between the prompts is 0.08 of mean score, the same as on s98_v2 (0.558 against
0.476), and again it is restraint: completeness is level at 39 to 40 percent while
the false-alarm rate is 31 percent against 2 percent. The base prompt submits ten
times as many wrong periods (30, 34, 27 against 3, 9, 10). The base-prompt wobble is
larger because the model with no procedure decides on its own when to stop, and that
decision varies with the sample. Against BLS on this set (completeness 0.34 at 5
percent false alarm, 0.38 at the default threshold with 61 percent), the rules
prompt finds more planets at a lower false-alarm rate than any BLS threshold, and
the base prompt finds as many as BLS at the default threshold with half its false
alarms. Both prompts find three of the 17 one-transit injections on average, where
BLS finds none.

**Three-episode checks of the large models on s98_v3** (`results/smoke/check_large.sh`,
base prompt, thinking renderers, five models side by side, 20:28 to 20:34 UTC).
Inkling (`tml_v0`, $0.77), Kimi K2.6 (`kimi_k26`, $1.11), Qwen3.5-397B-A17B
(`qwen3_5`, $1.54) and Nemotron 3 Ultra (`nemotron3_ultra`, $1.36): no errors, every
episode ended in a submission, no turn truncated. All four hit ep_0002 and declared
ep_0000 a true null. On ep_0001 (signal-to-noise ratio 1.6, unfindable) Inkling, Kimi
and Qwen submitted the 14.7-day, 16.7-day or 28.9-day noise peak with two events as a
planet; Nemotron and Inkling-Small declined. Kimi makes several tool calls in one turn
(9 calls in 4 turns); the loop handled it. GLM-5.3 (`glm5_3_high_reasoning`) failed on
every episode with `Tokenizer class TokenizersBackend does not exist`, the same
tokenizer failure as on 2026-10-11; it needs a newer transformers than the cookbook
pins. Full runs of the four working models launched at 20:36 UTC
(`results/full_v9large/run_large.sh`, seed 1, `--workers 8`, four processes at once).

**How much the models think.** `--max-tokens` added to `stargazer agent`: the output
cap for each turn, thinking included, None by default so the model-card value (8192
for a thinking renderer) applies and is recorded as `backend_max_tokens`. Inkling-Small
on the same three episodes with `--max-tokens 32768`
(`results/smoke/agent_inkling_small_s98_v3_base3_think32k.json`, $0.19): identical
outcomes and the same 500 output tokens an episode, because the cap never binds.
Measured over the checks, thinking text for each turn:

| model | output tokens an episode | turns with thinking | mean chars a turn | longest turn |
|---|---|---|---|---|
| Inkling-Small | about 500 | 1 of 11 | 6 | 71 |
| Inkling | 400 to 685 | 0 | 0 | 0 |
| Kimi K2.6 | 1,600 to 14,900 | all | 370 to 5,900 | 22,656 |
| Qwen3.5-397B | 2,000 to 7,300 | all | 320 to 1,450 | 10,014 |
| Nemotron 3 Ultra | 3,200 to 4,100 | all | 690 to 740 | 2,777 |

The longest thinking turn, 22,656 characters, is about 6,000 tokens, under the 8,192
cap. No model is cut off; the two Inkling models write no thinking at all in `tml_v0`
and act from the tool results alone. The cap is therefore not the lever for more
reasoning. The renderers in use are already the full-thinking variant of each family
(the cookbook offers `_disable_thinking` and, for Nemotron Ultra, a `_medium_thinking`
variant below them). What remains is the prompt, which would break the comparison and
would be a third prompt variant, or the reward.

**First seed of the four large models, base prompt** (`results/full_v9large/`, seed 1,
`--workers 8`, four processes at once, 20:36 to 21:23 UTC; the twenty remaining runs,
base seeds 2 and 3 and rules seeds 1 to 3 for each model, started at 21:17 UTC from
`run_seeds_large.sh` as eight parallel chains, and their numbers go in the next entry).

| model | renderer | false alarm | completeness | with aliases | mean score | one-transit hits | output tokens an episode | cost | minutes |
|---|---|---|---|---|---|---|---|---|---|
| Inkling-Small, mean of 3 seeds | tml_v0 | 31.1% | 40.4% | 42.1% | 0.492 | 3 of 17 | about 500 | $13 | 25 |
| Inkling | tml_v0 | 40.7% | 43.3% | 44.0% | 0.480 | 1 of 17 | 567 | $53 | 21 |
| Nemotron 3 Ultra | nemotron3_ultra | 28.8% | 38.3% | 40.4% | 0.486 | 4 of 17 | 3,236 | $66 | 23 |
| Qwen3.5-397B-A17B | qwen3_5 | 25.4% | 39.7% | 42.6% | 0.505 | 1 of 17 | 5,028 | $119 | 48 |
| Kimi K2.6 | kimi_k26 | 20.3% | 45.4% | 46.1% | 0.554 | 2 of 17 | 10,370 | $88 | 36 |

No errors in any run. Inkling and Nemotron Ultra land inside the small model's seed
spread (score within 0.034). Qwen is just above it on score, with the gain on the
false-alarm side. Kimi is outside it on both: more planets and fewer false alarms. The
order of the four by score is their order by output tokens an episode.

Reading the traces (Kimi, Qwen and Inkling-Small base seed 1, Inkling-Small rules
seed 1).

- The wrong periods are false alarms in disguise. The median expected
  signal-to-noise ratio of the injected planet on a wrong-period episode is 5.1 for
  Kimi, 5.3 for Qwen and 9.4 for Inkling-Small base, and 15 of Kimi's 24 are below 7.
  The planet was unfindable and the model submitted a noise peak with an SDE of 10
  to 12 and two events, which is the same act as a false alarm on a null episode.
  With the rules prompt the three wrong periods have median signal-to-noise 19.9: the
  rules remove the noise-peak submissions and leave real mistakes. The ratios of
  submitted to true period are mostly unrelated (not aliases), which fits.
- Every model misses the same findable planets. ep_0042, ep_0043, ep_0049 and
  ep_0059 (signal-to-noise 12 to 18, three to five transits, 1,300 to 4,000 ppm) are
  missed by Inkling-Small with both prompts, by Kimi and by Qwen. ep_0042, ep_0043
  and ep_0059 sit on two hosts of Gaia class RS (spotted rotators, TIC 21505902 and
  156836091). The strongest peak there is a 17.3-day V-shaped signal from the
  variability; the model rejects it as an eclipsing binary or as rotation and
  declines the whole star, without looking under it. The detrending window of 0.3
  days leaves the rotation in. This is the Phase 3 item "stellar context: rotation
  period, flares, variability class from the light curve", now with four named
  episodes as the test.
- Kimi's false alarms are argued, not careless: a single event above snr 10 with a
  flat-bottom fit on an isolated star, submitted under the single-event rule of the
  `single_events` tool description, or a two-event peak at 14 days whose double-period
  alias it worked out correctly before submitting the wrong one of the two.

Two harness defects found in the traces, both fixed after the runs in flight had
loaded the old code, so the twenty runs of this batch carry them:

1. Qwen3.5 writes `"detected": "False"` as text. The validator rejected it as not a
   boolean and the model resubmitted; 186 of its 200 episodes needed two or more
   submit calls (527 submit calls in all), one wasted turn each. `ToolSpec.coerce`
   now converts the words true and false and a text that parses as a number to the
   type the schema allows before validation; other text is still rejected.
2. Kimi K2.6 hit a cap in 9 of 200 episodes: `max_turns` (30) in 5, `max_tokens`
   (8,192 in one turn) in 3, and stopped without a tool call once. Two of the capped
   episodes held findable planets (ep_0028 at snr 35, ep_0063 at snr 26), scored as
   misses. The statement in the previous entry that no model is cut off was true of
   the three-episode checks and false of the full run. `stargazer agent --max-turns`
   now exists beside `--max-tool-calls` and `--max-tokens`; a Kimi rerun with
   `--max-turns 60 --max-tokens 32768` is the next check, after this batch.

## 2026-10-25

**The eight overnight runs, and a harness defect that the rescore repairs.** The eight
runs that were in flight at the close of 2026-10-16 (base seeds 2 and 3 of Kimi K2.6
and Qwen3.5-397B, rules seeds 1 and 2 of Inkling and Nemotron 3 Ultra) ended between
21:50 and 22:20 UTC with no errors, $853 in all, and were committed in `dfbd908`
without an entry. The handoff said that every run of the batch loaded the code before
the two fixes. That is true of the base-prompt runs, which started at 21:17 UTC, and
false of the four rules runs, which started after 21:50 UTC as the second step of each
chain, when the argument coercion was in the working tree (commit `6736617`, 21:48
UTC). The `git_commit` field of a results file is read when the file is saved, so it
does not say which code a run loaded; the tool results in the trajectories do.

The coercion fix had a defect of its own. `call_tool` converted the text `"False"` to
the boolean, accepted the submission and answered `ok`, but the agent loop built the
`Prediction` from the arguments as the model wrote them, and `bool("False")` is True.
Nemotron 3 Ultra writes every boolean as text. In its two rules runs 104 and 102
of the 200 submissions carried the text `"False"`; each became a detection with no
period, scored as a false alarm on a null episode or a wrong event on a transit
episode, and the files report false-alarm rates of 83 and 78 percent and scores of
0.333 and 0.319. The other six overnight runs loaded the old code, which rejected the
text and cost the model a turn, as the 2026-10-16 entry describes: Qwen3.5 resubmitted
in 186 of 200 episodes, and Nemotron 3 Ultra with the base prompt in 177 to 179 of 200, so
the base-prompt numbers of both models stand but each paid about one turn an episode.

Fix: `coerce_arguments` in `stargazer.tools` is the one place that reads a call's
arguments as the tool runs them; the agent loop calls it before the tool and builds
the prediction from its output; the transcript keeps the arguments as the model wrote
them. `stargazer rescore --results FILE --episodes SET` rebuilds each prediction of
an agent results file from its last accepted `submit_prediction` call, through the
current coercion, scores the file again with the tolerances in its config, and writes
`FILE_rescored.json` with `rescored_from` in the config. Tests:
`test_text_false_in_a_submission_is_not_a_detection`,
`test_rescore_rebuilds_each_prediction_from_its_accepted_submission`,
`test_coerce_arguments_uses_the_named_tool_schema_and_leaves_an_unknown_tool_alone`,
`test_rescore_command_defaults`. Check: every one of the 22 seed-run files on s98_v3
rescored to the same summary and the same 200 score rows as its source, except the
two Nemotron rules files, where 104 and 102 episodes change. The tables below use
`agent_nemotron3_ultra_s98_v3_rules_seed{1,2}_rescored.json`; the source files stay
as the record of what the harness did.

**Every seed run of the large models on s98_v3** (`results/full_v9large/`, hash
`745c6cb5...`, `--workers 8`, two chains for each model at once; the large models
cost $1,178 in all). Each row is one run of 200 episodes.

| model | prompt | seed | false alarm | completeness | with aliases | mean score | one-transit hits | hits / false alarms / wrong periods | capped episodes | output tokens an episode | cost | minutes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Inkling | base | 1 | 40.7% | 43.3% | 44.0% | 0.480 | 1 of 17 | 60 / 24 / 41 | 0 | 567 | $53 | 21 |
| Inkling | base | 2 | 35.6% | 44.0% | 44.7% | 0.502 | 2 of 17 | 61 / 21 / 40 | 0 | 554 | $53 | 41 |
| Inkling | base | 3 | 44.1% | 42.6% | 42.6% | 0.465 | 0 of 17 | 60 / 26 / 39 | 0 | 523 | $48 | 36 |
| Inkling | rules | 1 | 8.5% | 41.8% | 44.7% | 0.575 | 2 of 17 | 57 / 5 / 19 | 1 | 520 | $56 | 25 |
| Inkling | rules | 2 | 18.6% | 43.3% | 45.4% | 0.552 | 2 of 17 | 59 / 11 / 22 | 1 | 496 | $56 | 27 |
| Nemotron 3 Ultra | base | 1 | 28.8% | 38.3% | 40.4% | 0.486 | 4 of 17 | 51 / 17 / 19 | 0 | 3,236 | $66 | 23 |
| Nemotron 3 Ultra | base | 2 | 22.0% | 35.5% | 36.2% | 0.482 | 4 of 17 | 46 / 13 / 20 | 0 | 3,357 | $66 | 36 |
| Nemotron 3 Ultra | base | 3 | 28.8% | 37.6% | 39.7% | 0.478 | 3 of 17 | 52 / 17 / 18 | 0 | 3,377 | $65 | 36 |
| Nemotron 3 Ultra | rules, rescored | 1 | 6.8% | 39.0% | 41.1% | 0.557 | 4 of 17 | 51 / 4 / 12 | 0 | 2,672 | $54 | 30 |
| Nemotron 3 Ultra | rules, rescored | 2 | 10.2% | 34.8% | 37.6% | 0.518 | 2 of 17 | 48 / 6 / 7 | 0 | 2,872 | $54 | 29 |
| Qwen3.5-397B-A17B | base | 1 | 25.4% | 39.7% | 42.6% | 0.505 | 1 of 17 | 56 / 15 / 30 | 0 | 5,028 | $119 | 48 |
| Qwen3.5-397B-A17B | base | 2 | 33.9% | 39.7% | 42.6% | 0.481 | 2 of 17 | 55 / 20 / 24 | 1 | 4,784 | $123 | 51 |
| Qwen3.5-397B-A17B | base | 3 | 27.1% | 41.1% | 42.6% | 0.507 | 3 of 17 | 56 / 16 / 27 | 1 | 4,626 | $115 | 52 |
| Kimi K2.6 | base | 1 | 20.3% | 45.4% | 46.1% | 0.554 | 2 of 17 | 64 / 12 / 24 | 9 | 10,370 | $88 | 36 |
| Kimi K2.6 | base | 2 | 20.3% | 45.4% | 45.4% | 0.552 | 3 of 17 | 63 / 12 / 30 | 3 | 9,775 | $82 | 64 |
| Kimi K2.6 | base | 3 | 22.0% | 43.3% | 44.7% | 0.538 | 2 of 17 | 59 / 13 / 32 | 2 | 9,320 | $81 | 63 |

Mean over seeds with the standard deviation in brackets, beside Inkling-Small from the
2026-10-16 entry and BLS at 5 percent false alarm (completeness 34 percent):

| model | prompt | seeds | false alarm | completeness | with aliases | mean score | one-transit hits | output tokens an episode | cost a run |
|---|---|---|---|---|---|---|---|---|---|
| Inkling-Small | rules | 3 | 2.3% (2.0) | 39.2% (0.8) | 41.4% (0.8) | 0.573 (0.000) | 2.7 of 17 | 395 | $12 |
| Inkling | rules | 2 | 13.6% (7.2) | 42.6% (1.0) | 45.0% (0.5) | 0.564 (0.016) | 2.0 of 17 | 508 | $56 |
| Nemotron 3 Ultra | rules | 2 | 8.5% (2.4) | 36.9% (3.0) | 39.4% (2.5) | 0.538 (0.028) | 3.0 of 17 | 2,772 | $54 |
| Kimi K2.6 | base | 3 | 20.9% (1.0) | 44.7% (1.2) | 45.4% (0.7) | 0.548 (0.009) | 2.3 of 17 | 9,822 | $84 |
| Qwen3.5-397B-A17B | base | 3 | 28.8% (4.5) | 40.2% (0.8) | 42.6% (0.0) | 0.498 (0.014) | 2.0 of 17 | 4,813 | $119 |
| Inkling-Small | base | 3 | 31.1% (5.2) | 40.4% (3.3) | 42.1% (2.2) | 0.492 (0.034) | 2.7 of 17 | 471 | $13 |
| Inkling | base | 3 | 40.1% (4.3) | 43.3% (0.7) | 43.7% (1.1) | 0.482 (0.018) | 1.0 of 17 | 548 | $51 |
| Nemotron 3 Ultra | base | 3 | 26.6% (3.9) | 37.1% (1.5) | 38.8% (2.3) | 0.482 (0.004) | 3.7 of 17 | 3,324 | $65 |

Reading.

- The seed spread of the large models is smaller than the small model's with the base
  prompt (standard deviation of the score 0.004 to 0.018 against 0.034), so the
  ranking of 2026-10-16 holds with three seeds: Kimi K2.6 is above every other
  base-prompt run by more than any spread, and Inkling, Nemotron 3 Ultra and Qwen3.5
  are within the small model's spread of each other. Kimi's rank comes from the
  false-alarm side (21 percent against 27 to 40) with the highest completeness.
- The rules prompt does for Inkling and Nemotron 3 Ultra what it does for
  Inkling-Small: completeness stays where it was and the false-alarm rate falls, by
  27 points for Inkling and 18 for Nemotron. Neither reaches the small model's 2
  percent. Inkling with the rules submits a different set of null hosts in each seed:
  seed 1 the 0.946-day signal on TIC 55270064 twice and single events on TIC
  117747153 and TIC 121711742; seed 2 eleven false alarms on seven other hosts, among
  them the 14.4-day signal on TIC 33831807 in all three of its null episodes, and
  none on TIC 55270064. Inkling-Small with the rules submits two false alarms or
  none. The two Inkling rules seeds differ by 10 points of false-alarm rate, the
  largest spread in the table, so a third seed is needed before the gap to
  Inkling-Small counts.
- The false alarms are on the hosts, not spread over the sample. Across the 22 seed
  runs on s98_v3 (16 large-model runs and 6 of Inkling-Small) there are 291 false
  alarms on 58 null episodes of 27 hosts, and 128 of them are on three hosts: TIC
  146420104 (47 of 66 chances, a 15.37-day signal in 27 of them), TIC 33831807 (45 of
  66, 14.44 days in 28) and TIC 55270064 (36 of 44, 0.946 days in all 36). The same
  period from runs of five models with two prompts means the signal is in the light
  curve. The 0.946-day dip on TIC 55270064 passes every tool in the box in every run
  that reaches it: flat bottom, host density 3.4 g/cc, odd and even depths the same,
  no secondary, no TOI, no eclipsing-binary entry, no Gaia class, and
  `ephemeris_match` answers `no_match`. By the benchmark it is a false alarm; by the
  tools it is a candidate. The 14.4-day and 15.4-day signals, and the 13.9- to
  14.4-day peaks that lead the false alarms on TIC 279511563, TIC 32816814, TIC
  146535972 and TIC 117747153, sit at the length of one TESS orbit (half the sector),
  where the gap and the scattered-light ramps at the orbit ends make a two-event box
  signal; these are systematics, and a tool that knows the orbit boundaries would name
  them. Both findings go to the handoff.
- Capped episodes: Kimi K2.6 hit a cap in 9, 3 and 2 episodes (seed 1 with the
  `max_tokens` and `model_stopped` cases of the 2026-10-16 entry; seeds 2 and 3 only
  `max_turns`). Inkling with the rules looped on `fold` and `fit_transit` for 30
  turns on ep_0049 (seed 1) and ep_0063 (seed 2), both findable planets scored as
  misses, about $1 each. Qwen3.5 hit `max_turns` once in each of seeds 2 and 3.
- Cost: Qwen3.5 is the most expensive model at $119 a run for the score of the small
  model at $13; Kimi K2.6 at $84 buys 0.06 of score over the small model's base
  prompt and the same score as the rules prompt at seven times the price. With the
  rules prompt the cost of a large model is $54 to $56 a run.

The eight rules runs that were not launched (rules seed 3 of Inkling and Nemotron 3
Ultra, rules seeds 1 to 3 of Kimi K2.6 and Qwen3.5-397B) stay unlaunched; about $700
at the costs above, which needs the owner's yes.
