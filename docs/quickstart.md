# Quickstart

This page runs the full workflow on TESS sector 98: select host stars, build an
episode set, run the baseline and an agent, then read and compare the results.

## Build an episode set

```bash
stargazer targets --sector 98 --n 60 --seed 1
stargazer fetch   --targets data/targets_s98.json
stargazer build   --targets data/targets_s98.json --n 200 --seed 1 --out data/episodes/s98_v1
stargazer verify  data/episodes/s98_v1
```

- `targets` draws 60 stars observed at 2-minute cadence in sector 98 and removes every
  star with a TESS Object of Interest (TOI).
- `fetch` downloads their light curves into `data/cache/`.
- `build` writes 200 episodes, about 30 percent of them without a transit.
- `verify` recomputes the hash of every file against the manifest.

See {doc}`episode_sets` for the selection cuts and the injection model.

## Run the baseline

```bash
stargazer baseline --episodes data/episodes/s98_v1
```

The command writes `results/baseline_bls_s98_v1_<timestamp>.json`, or the path given to `--out`, and prints the summary. The
default threshold gives a high false-alarm rate on 57-day light curves; choose a
threshold from the null episodes as described in {ref}`threshold-choice`.

## Run an agent

```bash
stargazer agent --episodes data/episodes/s98_v1 --backend tinker \
    --model Qwen/Qwen3.8-27B --renderer qwen3_8_disable_thinking --limit 3
```

`--limit 3` runs three episodes. Read their trajectories before a larger run, then use
`--limit 0 --workers 8` for the full set.

## Read the trajectories

```bash
stargazer view --results results/<run>.json
stargazer show --results results/<run>.json --episode ep_0001
```

## Compare systems

```bash
stargazer compare results/baseline_bls_s98_v1_sde7.json results/full/agent_*.json
```

The output has one row for each system, with the false-alarm rate, the completeness and
the mean score, and a second table of completeness by expected signal-to-noise ratio.

## Use the library

The command line calls the same functions:

```python
from stargazer import BLSDetector, EpisodeSet, ToolAgentDetector, evaluate
from stargazer.backends import TinkerBackend

episodes = EpisodeSet("data/episodes/s98_v1")
baseline = evaluate(BLSDetector(sde_threshold=18.6), episodes, workers=8)

backend = TinkerBackend("Qwen/Qwen3.8-27B", renderer="qwen3_8_disable_thinking")
agent = evaluate(ToolAgentDetector(backend), episodes, workers=8)
agent.save("results/my_run.json")
print(agent.summary["completeness_exact"], agent.summary["false_alarm_rate"])
```
