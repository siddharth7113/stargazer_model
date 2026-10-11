# Reproducibility

Each number in this project can be traced to a results file, and each results file to
the code, the data and the configuration that produced it.

## Seeds

- Each random choice uses one `numpy.random.Generator` with an explicit integer seed.
- The seed is written to the output: `targets_sNN.json`, the `config.json` of an
  episode set, and the configuration of an agent run.
- The same seed and the same inputs give identical files. `tests/test_episodes.py`
  checks this.
- A hosted model is not deterministic for a given seed. Repeat agent runs with
  different seeds to measure the spread.

## Manifests

{class}`~stargazer.provenance.Manifest` records the SHA-256 of each file in a
directory, and `Manifest.verify` recomputes them. The hash of an episode set is the
SHA-256 of its manifest.

```{mermaid}
flowchart LR
    files[episode files, truth, config] -->|SHA-256 each| manifest[manifest.json]
    manifest -->|SHA-256| hash[episode set hash]
    hash --> results[results.json]
    results --> logbook[LOGBOOK.md]
```

```bash
stargazer verify data/episodes/s98_v1
```

An episode set must not pass through a formatter or a line-ending conversion, which
would change its hashes. The pre-commit configuration excludes `data/` and `results/`
for this reason.

## Separation of the truth

- A detector reads only the episode file: `time`, `flux` and `flux_err`.
- The scorer reads the truth file.
- {meth}`~stargazer.episodes.EpisodeSet.build` writes the two from different
  dictionaries, and `tests/test_episodes.py` checks that an episode file holds only the
  three arrays.

## Contents of a results file

{meth}`~stargazer.evaluate.Results.save` writes:

| Field | Content |
|---|---|
| `created` | UTC time |
| `git_commit` | commit hash, with `-dirty` if the working tree had changes |
| `system` | detector name |
| `config` | full detector configuration, including the backend, model, renderer, sampling parameters, seed, limits and the SHA-256 of the system prompt |
| `scoring` | the tolerances |
| `episode_set` | path and hash |
| `preds` | the prediction of each episode, with the trajectory for an agent |
| `rows` | the score row of each episode |
| `summary` | the set metrics |

{func}`~stargazer.compare.compare` refuses results files from different episode sets.

## Tests

- `pytest` runs without network access, on synthetic white-noise light curves with
  the cadence and gap structure of TESS.
- `tests/` has one file for each module; each test covers one behaviour and has a
  docstring that states what it checks and why.
- `tests/integration/` holds tests that cross module boundaries.
- Style rules, such as the ban on single-letter names, are pre-commit hooks in
  `scripts/`.

## Logbook

Each run whose number appears in a document has an entry in {doc}`logbook` with the
date, the exact command and the results file.
