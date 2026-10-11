# Installation

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/) or pip
- Network access to MAST and ExoFOP to build an episode set. The test suite and the
  scoring of an existing set run offline.

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
| `tinker` | tinker, tinker-cookbook | `stargazer agent --backend tinker` and RL training |
| `view` | textual | `stargazer view` |

The core dependencies are numpy, astropy, lightkurve, batman-package, PyYAML,
python-dotenv and rich.

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
