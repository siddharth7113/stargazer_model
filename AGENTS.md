# AGENTS.md

This file gives guidance to a coding agent that works in this repository. The owner
reviews every change closely. Before a change, read this file, then STYLE.md if it is
present, then the Handoff section at the end of PLAN.md, then the last entries of
LOGBOOK.md. STYLE.md is the owner's personal style file. It is not tracked by git, so
a clone does not have it. The rules below are the parts of it that apply to code.

## Project

Star Gazer is a verifiable benchmark and reinforcement-learning environment for
exoplanet detection. The owner states the goal as follows. The model acts as an
astronomer with the classical detection and vetting tools of the field, works at
machine speed and makes discoveries. PLAN.md states the goal, the principles and the
roadmap. LOGBOOK.md has every number produced so far with the command that produced it.

## Fixed rules

- Follow STYLE.md. It requires Simplified Technical English in docs and docstrings;
  numpydoc with `Notes` for every equation and numbered `References` with DOIs;
  absolute imports only; no single-letter variable names; a lookup table for a chain
  of `if` statements; one class for one responsibility, with no loose helper
  functions; maintained standard packages only (numpy, scipy, astropy, lightkurve,
  batman, textual).
- Every docstring must stand alone for a first-time reader. Define "episode",
  "detector", "truth" where they appear.
- Tests: one file for each module under `tests/`, one behaviour for each test, a
  docstring on every test saying what it checks and why, no tautological asserts.
  Cross-module tests go in `tests/integration/`. Style rules are pre-commit hooks in
  `scripts/`. Do not write a test for a style rule.
- Do not add a spend cap, a budget or a cost guard in code. The owner controls spend
  through API keys. Report cost only when the backend knows it; never print a zero
  as a placeholder.
- Agent code must be provider-agnostic. The Tinker backend uses the cookbook
  renderer's native tool format (`create_conversation_prefix_with_tools`,
  `parse_response`); never hand-parse tool calls, never impose one family's syntax.
- Sampling defaults come from the model cards (`stargazer.backends.SAMPLING_DEFAULTS`)
  with a recorded seed. Temperature 0 is not a default.
- Say "identical" or "the same"; never "byte-identical" or "golden output".
- Do not commit `data/episodes/`, `data/cache/`, `.env` or run logs. Results JSON
  files under `results/` are committed; they are evidence.
- Artifacts with a manifest (episode sets) must never pass through a formatter.
  Pre-commit excludes `data/` and `results/` for that reason.

## Commands

```bash
uv venv && uv pip install -e ".[dev,docs,tinker,view]"
.venv/bin/python -m pytest -W ignore -q          # offline, about 20 seconds
.venv/bin/ruff format . && .venv/bin/ruff check .
.venv/bin/python scripts/check_names.py stargazer/*.py
.venv/bin/sphinx-build -q -b html docs docs/_build/html
```

Keys live in `.env` (copy `.env.example`); the CLI loads it. `stargazer agent` runs a
model on an episode set; start with `--limit 3` and read the trajectory with
`stargazer view` before a full run. Full runs use `--workers 8` and one process for
each model.

## Working method

- Before an expensive run, do a three-episode run and fix every harness defect that it
  shows. A parsed-but-unused flag or a cap that cuts a model off makes the results
  file lie about what ran. The results file's `config` is the only truth about a run.
- Record every run in LOGBOOK.md: date, exact command, results file, numbers, what
  they mean. Tick a PLAN.md box only with a LOGBOOK entry as evidence.
- A change to the tools or the prompt breaks the comparison with earlier runs. Record
  the break in the LOGBOOK and rerun the baseline comparison.
- Prefer a tool that gives the model a verdict word over one that makes the model
  apply a threshold. Prefer derived fields (alias periods, event counts) over asking
  the model to do arithmetic.
- End a session by updating the Handoff section of PLAN.md: what is running, where
  results will land, the first three steps for the next session.

## Layout

The module map is in docs/architecture.md. The public API is in `stargazer/__init__.py`.
