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

### `stargazer verify`

```bash
stargazer verify data/episodes/s98_v1
```

Recompute the hash of each file and report each mismatch with the manifest.

## Detectors

### `stargazer baseline`

Run {class}`~stargazer.baseline.BLSDetector`. See {doc}`baseline`.

| Option | Default | Meaning |
|---|---|---|
| `--episodes` | required | episode set |
| `--config` | | JSON overrides for the detector, for example `'{"sde_threshold": 18.6}'` |
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
| `--temperature` | model card value | sampling temperature |
| `--max-tokens` | model card value | output cap for each turn, thinking included; 8192 for a thinking renderer |
| `--seed` | 0 | sampling seed |
| `--max-tool-calls` | 40 | tool-call limit for each episode |
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
