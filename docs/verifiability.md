# Verifiability

Each number in this project must be reproducible by another person. These are the
mechanisms.

## Seeds

Each random choice uses one `numpy.random.Generator` with an explicit integer seed. The
output records the seed. The same seed and the same inputs give identical output files.
`tests/test_episodes.py` checks this.

## Manifests

`Manifest.of_directory` records the SHA-256 of each file in a directory. `Manifest.verify`
recomputes the hashes and reports each mismatch. The episode-set hash is the SHA-256
of the manifest. A results file records that hash, so a reader can confirm which
episodes produced a number.

```{mermaid}
flowchart LR
    files[episode files, truth, config] -->|sha256 each| manifest[manifest.json]
    manifest -->|sha256| hash[episode set hash]
    hash --> results[results.json]
    results --> logbook[LOGBOOK.md]
```

## Separated truth

The detector reads only the episode file. The scorer reads the truth file. The code
enforces this in `EpisodeSet.build`, which writes the two files from different
dictionaries. `tests/test_episodes.py` checks that an episode file holds only `time`,
`flux` and `flux_err`.

## Provenance in results

`Results.to_dict` records:

- the UTC creation time,
- the git commit, with `-dirty` if the working tree had changes,
- the detector name and full configuration,
- the scoring tolerances,
- the episode set path and hash,
- the prediction and the score row of each episode.

## Offline tests

`pytest` runs without the network. The tests use synthetic white-noise light curves
with the same cadence and gap structure as TESS, so each test knows the exact noise
level and gap position.

The suite follows one rule: one test checks one behaviour of one function, and its
docstring states the behaviour and why it matters. There is one test file for each
module in `tests/`. Tests that cross module boundaries live in `tests/integration/`
with the `integration` marker; they check that the parts fit, and a unit test must fail
first on a unit-level bug. Style rules such as "no single-letter names" are pre-commit
hooks in `scripts/`, not tests. The pre-commit hook runs the suite before each commit.

## Logbook

Each run that gives a number in a document has a LOGBOOK entry. The entry gives the
date, the exact command and the results file. A reader can rerun the command and
compare.
