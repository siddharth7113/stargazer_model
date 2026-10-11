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
values. The CLI loads it at start; git ignores it; the key is never written into a
results file. An exported variable works as well.

```bash
cp .env.example .env   # then edit
uv pip install -e ".[openai]"
stargazer agent --episodes data/episodes/s98_v1 --model qwen/qwen3-32b --limit 3
```

The default `--limit 3` runs three episodes. Read the transcript in the results file
before you run more. For Tinker:

```bash
uv pip install -e ".[tinker]"
stargazer agent --episodes data/episodes/s98_v1 --backend tinker --model Qwen/Qwen3-8B --limit 3
```

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
