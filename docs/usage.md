# Usage

## Install

```bash
uv venv
uv pip install -e ".[dev,docs]"
pre-commit install
pytest
```

## Build an episode set

```bash
stargazer targets --sector 98 --n 60 --seed 1
stargazer fetch   --targets data/targets_s98.json
stargazer build   --targets data/targets_s98.json --n 200 --seed 1 --out data/episodes/s98_v1
stargazer verify  data/episodes/s98_v1
```

## Run the baseline

```bash
stargazer baseline --episodes data/episodes/s98_v1
```

The command writes a results file under `results/` and prints the summary.

## Write a detector

Implement the protocol. Keep `config` complete, so a results file can reproduce a run.

```python
import numpy as np

from stargazer import Prediction


class MyDetector:
    name = "my_detector"

    def __init__(self, threshold: float = 8.0):
        self._threshold = threshold

    @property
    def config(self) -> dict:
        return {"threshold": self._threshold}

    def detect(self, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray) -> Prediction:
        ...
        return Prediction(detected=True, period=3.18, t0=1.2, depth=5e-4, extra={"stat": 9.1})
```

Then evaluate it on the same episode set as the baseline.

```python
from stargazer import EpisodeSet, evaluate

episodes = EpisodeSet("data/episodes/s98_v1")
results = evaluate(MyDetector(), episodes)
results.save("results/my_detector_s98_v1.json")
```

## Run a language-model agent

Put the provider key in a `.env` file. Copy `.env.example` to `.env` and fill in the
values.

```bash
cp .env.example .env   # then edit
uv pip install -e ".[openai]"
stargazer agent --episodes data/episodes/s98_v1 --model qwen/qwen3-32b --limit 3
```

The sampling parameters default to the values in the model card for the renderer:
temperature 0.7, top_p 0.8 and top_k 20 in non-thinking mode, and 0.6, 0.95 and 20 in
thinking mode. Greedy decoding is discouraged by the model cards and does not make a
hosted model deterministic, so temperature 0 is not a default. The seed is recorded in
the results file; to measure run-to-run variance, repeat the run with `--seed 1` and
`--seed 2` and report the mean and spread.

The default `--limit 3` runs three episodes. Read their transcripts before a larger
run. The Tinker backend needs the `tinker` extra.

```bash
uv pip install -e ".[tinker]"
stargazer agent --episodes data/episodes/s98_v1 --backend tinker --model Qwen/Qwen3-8B --limit 3
```

## Read the trajectories

```bash
stargazer view --results results/full/agent_qwen38_27b_s98_v1.json   # terminal viewer
stargazer show --results results/full/agent_qwen38_27b_s98_v1.json --episode ep_0001
```

The viewer has three panes. The left pane lists every episode with its expected
signal-to-noise ratio (SNR), outcome and score. The centre pane shows the trajectory turn by turn: a header with the tool calls,
the thinking, the model's text, one summary line per tool result and any nudge. The
right pane shows the truth next to the prediction, and the stop reason, tokens and cost.

| Key                        | Action                                                              |
| -------------------------- | ------------------------------------------------------------------- |
| `j`, `k` or the arrow keys | next or previous episode                                            |
| `n`, `p`                   | next or previous turn, or search match                              |
| `enter`                    | expand or collapse the results of the current turn                  |
| `r`                        | expand or collapse all results as tables                            |
| `t`                        | expand or collapse all thinking                                     |
| `f`                        | cycle the filter: all, hit, miss, false_alarm, wrong_period, errors |
| `/`, `escape`              | search the trajectory, clear the search                             |
| `q`                        | quit                                                                |

`stargazer show` prints the same turns to the console. Add `--full` to expand each result.

## Compare systems

```bash
stargazer compare results/baseline_bls_s98_v1_sde7.json results/full/agent_*.json
```

The command accepts only files scored on the same episode set. It prints the
false-alarm rate, the completeness and the mean score of each system, then a second
table of completeness by SNR bin. An agent has no detection statistic to sweep, so its
completeness is read at the false-alarm rate it produced.

## Score external predictions

If a system runs outside Python, write its predictions as JSON keyed by episode id and
score the file.

```bash
stargazer score --episodes data/episodes/s98_v1 --preds preds.json --system my_system
```

## Build the documentation

```bash
sphinx-build -b html docs docs/_build/html
```
