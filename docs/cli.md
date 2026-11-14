# Command line

The `stargazer` command wraps the library. Each subcommand prints a summary and writes
a JSON file. The command loads `.env` from the current directory at start.

```bash
stargazer <command> --help
```

## Episode sets

### `stargazer targets`

Select host stars in one sector. See {func}`~stargazer.targets.select_targets`.

| Option | Default | Meaning |
|---|---|---|
| `--sector` | required | TESS sector |
| `--n` | 60 | number of stars to draw before the cuts |
| `--seed` | 1 | random seed |
| `--tmag-max` | 12.0 | faintest TESS magnitude |
| `--out` | `data/targets_sNN.json` | output file |
| `--eclipsing-binaries` | | CSV copy of the TESS eclipsing-binary catalogue; the binary cut then makes no Vizier query for it |
| `--vizier-server` | Strasbourg | Vizier mirror for the Gaia variability classes |

### `stargazer remove-binaries`

Remove the known eclipsing binaries from a targets file that was selected before the
binary cut existed, and write a new file. See
{func}`~stargazer.targets.remove_binaries`.

| Option | Meaning |
|---|---|
| `--targets` | input targets file, left unchanged |
| `--out` | output targets file |

### `stargazer fetch`

Download and cache the light curves of a targets file.

| Option | Meaning |
|---|---|
| `--targets` | targets file |
| `--cache` | cache directory, default `data/cache` |

### `stargazer build`

Inject transits and write an episode set. See {doc}`episode_sets`.

| Option | Default | Meaning |
|---|---|---|
| `--targets` | required | targets file |
| `--cache` | `data/cache` | cache directory |
| `--out` | required | output directory |
| `--n` | 200 | number of episodes |
| `--seed` | 1 | random seed |
| `--null-frac` | 0.3 | fraction of null episodes |
| `--min-period`, `--max-period` | 0.5, 20.0 | period range in days |
| `--min-depth`, `--max-depth` | 1e-4, 1e-2 | depth range |

### `stargazer context`

Fetch the catalogue rows of each host and write `context.json` into an episode set:
the star's TOI entries, its TESS eclipsing-binary entries, its Gaia variability
classes and the Gaia sources within three pixels. The tools `ephemeris_match` and
`nearby_stars` read it. See {doc}`episode_sets`.

| Option | Default | Meaning |
|---|---|---|
| `--episodes` | required | episode set |
| `--targets` | required | one or more targets files that together name every host |
| `--mask-tois` | off | leave out the hosts' own TOI entries and any eclipsing-binary entry at the hidden planet's period (real-planet sets); the TOI list is not downloaded |
| `--gaia-source` | `vizier` | `vizier` or `mast`; MAST holds a copy of Gaia DR3 for the sources, not the classes |
| `--eclipsing-binaries` | | a CSV copy of the TESS eclipsing-binary catalogue from MAST, `data/catalogues/`; with it no Vizier query runs for the binaries |
| `--vizier-server` | Strasbourg | a Vizier mirror, for example `vizier.cfa.harvard.edu`, for every Vizier query |
| `--reuse` | | a `context.json` of another set whose stars are copied instead of fetched |

Each field of a context records where it came from, so a reader of a results file
sees which catalogues the model could have read.

### `stargazer verify`

```bash
stargazer verify data/episodes/s98_v1
```

Recompute the hash of each file and report each mismatch with the manifest.

## Detectors

### `stargazer baseline`

Run a baseline detector: {class}`~stargazer.baseline.BLSDetector` (see {doc}`baseline`) or
{class}`~stargazer.scripted.ScriptedRulesDetector`, the procedure of the rules prompt as fixed code.

| Option | Default | Meaning |
|---|---|---|
| `--episodes` | required | episode set |
| `--detector` | `bls` | `bls` or `scripted` |
| `--config` | | JSON overrides for the detector, for example `'{"sde_threshold": 18.6}'` |
| `--workers` | 1 | episodes evaluated at the same time; 8 gives about six times the speed |
| `--limit` | 0 | number of episodes; 0 runs all |
| `--period-rtol` | 0.01 | period tolerance of the scorer |
| `--out` | timestamped file in `results/` | output file |

### `stargazer agent`

Run {class}`~stargazer.agent.ToolAgentDetector`. See {doc}`agent`.

| Option | Default | Meaning |
|---|---|---|
| `--episodes` | required | episode set |
| `--backend` | `openai` | `openai` or `tinker` |
| `--model` | required | model identifier of the provider, or the Tinker base model |
| `--base-url` | OpenRouter | server of the `openai` backend |
| `--api-key-env` | `OPENROUTER_API_KEY` | environment variable that holds the key |
| `--model-path` | | `tinker://` path of a trained checkpoint |
| `--renderer` | | Tinker cookbook renderer |
| `--prompt` | `base` | `base` states the task, the scoring and the units; `rules` adds the procedure and the decision rules |
| `--temperature` | model card value | sampling temperature |
| `--max-tokens` | model card value | output cap for each turn, thinking included; 8192 for a thinking renderer |
| `--seed` | 0 | sampling seed |
| `--max-tool-calls` | 40 | tool-call limit for each episode |
| `--max-turns` | 30 | model-turn limit for each episode; a family that thinks at length needs more |
| `--limit` | 3 | number of episodes; 0 runs all |
| `--workers` | 8 | episodes run at the same time |
| `--period-rtol` | 0.01 | period tolerance of the scorer |
| `--out` | timestamped file in `results/` | output file |

### `stargazer score`

Score predictions made outside Python. The file maps each episode identifier to a
prediction:

```json
{"ep_0000": {"detected": true, "period": 3.18, "t0": 1.2}, "ep_0001": {"detected": false}}
```

| Option | Meaning |
|---|---|
| `--episodes` | episode set |
| `--preds` | predictions file |
| `--system` | name written to the results file |
| `--period-rtol` | period tolerance |
| `--out` | output file |

### `stargazer rescore`

```bash
stargazer rescore --results results/full_v9large/agent_nemotron3_ultra_s98_v3_rules_seed1.json \
                  --episodes data/episodes/s98_v3
```

Rebuild the predictions of an agent results file from the trajectories it holds and
score them again. The prediction of an episode is its last accepted `submit_prediction`
call, read with the argument coercion of the current code. The command repairs a
results file that an earlier harness misread, for example one where the text
`"False"` counted as a detection. The new file names the source under
`rescored_from` in its config. The trajectories, the usage and the cost stay as they
were.

| Option | Meaning |
|---|---|
| `--results` | the results file of an agent run |
| `--episodes` | the episode set it ran on; the hash must match |
| `--out` | output file; default is the results path with `_rescored` before `.json` |

## Training

### `stargazer train`

Train a LoRA policy on an episode set through the Tinker cookbook's
reinforcement-learning loop. The episode is the observation, each tool call an action,
the benchmark score of the submission the reward. See {doc}`architecture` and
{class}`~stargazer.environment.TrainingRun`.

```bash
stargazer train --episodes data/episodes/s98_v3 --test-episodes data/episodes/s92_v1 \
    --model thinkingmachines/Inkling-Small --renderer tml_v0 --prompt rules \
    --batch-size 8 --group-size 4 --steps 30 --learning-rate 5e-5 \
    --log-path data/train/<run>
```

| Option | Default | Meaning |
|---|---|---|
| `--episodes` | required | the training episode set |
| `--test-episodes` | | a held-out set evaluated every `--eval-every` steps, one rollout for each episode |
| `--model` | required | Tinker base model |
| `--renderer` | required | Tinker cookbook renderer of the model family |
| `--prompt` | `rules` | the system prompt, as for `stargazer agent` |
| `--reward` | `score` | `score` pays the benchmark score; `penalised` pays a false alarm -1 and the score otherwise. The decision of 2026-11-01 trains the first run on `score` |
| `--batch-size` | 8 | episode groups in one training step |
| `--group-size` | 4 | rollouts of the same episode in a group; the loss centres each reward on its group's mean |
| `--learning-rate` | required | LoRA learning rate; the cookbook has no calibrated value for Inkling |
| `--lora-rank` | 32 | LoRA rank |
| `--max-tokens` | 4096 | output cap for each sampled turn |
| `--max-turns`, `--max-tool-calls` | 30, 40 | the detector's safety stops, applied by the cookbook |
| `--max-trajectory-tokens` | 60 000 | conversation length at which the episode ends |
| `--temperature` | 1.0 | sampling temperature during training; the cookbook's default, hotter than the model card |
| `--steps` | one pass | training steps |
| `--eval-every`, `--save-every` | 0, 20 | steps between evaluations and between checkpoints; 0 means never, and a run without saving discards the final checkpoint the cookbook writes. With saving on, a pruner beside the loop deletes the training state of every checkpoint but the newest and keeps every sampler checkpoint, so a run stays within the free store (100 GB) |
| `--seed` | 0 | order of the episodes |
| `--log-path` | required | directory for the run's `record.json`, the cookbook's `metrics.jsonl`, `config.json` and checkpoints; `data/train/` is not committed |

A rollout that raises stops the run, so a run never trains on fewer groups than its
record says. The checkpoint is evaluated with `stargazer agent --backend tinker
--model-path tinker://...` on the same tools.

## Analysis

### `stargazer calibrate`

Choose a detection threshold from the null episodes of a results file. See
{ref}`threshold-choice`.

| Option | Default | Meaning |
|---|---|---|
| `--results` | required | results file |
| `--stat` | `sde` | statistic in `Prediction.extra` |
| `--thresholds` | `5,6,7,8,9,10,12,15` | thresholds at which to report the metrics |
| `--max-false-alarm` | 0.05 | target false-alarm rate |
| `--out` | | output file |

### `stargazer compare`

```bash
stargazer compare results/a.json results/b.json ...
```

Print one table for several results files from the same episode set.

### `stargazer view` and `stargazer show`

Read the trajectories of an agent run. See {doc}`trajectories`.
