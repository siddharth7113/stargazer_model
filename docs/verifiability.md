# Verifiability

This page lists the mechanisms that make a number in this project reproducible: the
seeds, the file manifests, the separation of truth from episode data, the provenance
fields in a results file, the offline test suite and the logbook.

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

A detector reads only the episode file and the scorer reads the truth file.
`EpisodeSet.build` writes the two from different dictionaries, so no field of the truth
can reach an episode file. `tests/test_episodes.py` checks that an episode file holds only `time`,
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

Each test covers one behaviour of one function, and its docstring states what is
checked and why. There is one test file for each module in `tests/`, and
tests that cross module boundaries are in `tests/integration/` with the `integration`
marker. Style rules such as "no single-letter names" are pre-commit hooks in
`scripts/`; the pre-commit configuration also runs the test suite before each
commit.

## Logbook

Each run whose number appears in a document has an entry in LOGBOOK.md with the date,
the exact command and the results file, so the command can be rerun and the number
compared.
