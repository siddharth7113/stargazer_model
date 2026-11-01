# Installation

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/) or pip
- Network access to MAST and ExoFOP to build an episode set, and to Vizier (or the
  catalogue copy under `data/catalogues/`) for the catalogue context. The test suite
  and the scoring of an existing set run offline.

## Install from source

Star Gazer is not on PyPI. Clone the repository and install it in editable mode:

```bash
git clone https://github.com/siddharth7113/stargazer_model.git star_finder
cd star_finder
uv venv
uv pip install -e ".[dev,docs,tinker,view]"
```

## Optional extras

| Extra | Installs | Needed for |
|---|---|---|
| `dev` | pytest, ruff, pre-commit | tests and the commit hooks |
| `docs` | Sphinx, MyST, furo, sphinx-design, sphinxcontrib-mermaid | building this documentation |
| `openai` | openai | `stargazer agent --backend openai` (OpenRouter, vLLM, Ollama) |
| `tinker` | tinker, tinker-cookbook | `stargazer agent --backend tinker` and `stargazer train` |
| `view` | textual | `stargazer view` |

The core dependencies are numpy, astropy, lightkurve, batman-package, PyYAML,
python-dotenv and rich.

The cookbook's training loop imports `datasets`, `transformers` and
`huggingface-hub`, whose releases constrain each other. The set that works on
2026-11-01 is transformers 5.5.4, huggingface-hub 1.33 and datasets 5.1; an older
`datasets` fails to import against pyarrow 26, and an older `transformers` refuses
huggingface-hub 2. When `stargazer train` fails at import, resolve the three
together:

```bash
uv pip install -U "transformers>=4.57.6,<=5.5.4" datasets tokenizers huggingface-hub
```

## API keys

The agent backends read their keys from environment variables. The command line loads
a `.env` file from the project root at start, and git ignores that file.

```bash
cp .env.example .env
```

| Variable | Backend |
|---|---|
| `TINKER_API_KEY` | Tinker |
| `OPENROUTER_API_KEY` | OpenRouter, the default server of the `openai` backend |
| any name given to `--api-key-env` | another OpenAI-compatible server |

A key is never written to a results file.

## Check the installation

```bash
pytest                 # offline, about 20 seconds
stargazer --help
```

To install the commit hooks:

```bash
pre-commit install
```
