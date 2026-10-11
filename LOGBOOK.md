# Logbook

Newest entry at the bottom. Each entry has: date, what was done, exact command, evidence.

## 2026-10-11

**Context.** Project started after reading about an independent researcher's AI-assisted
candidate around TIC 4206066 (3.18 day, about 500 ppm, Zenodo preprint, not confirmed).
Decision: build a general injection-recovery benchmark rather than chase one result.

**Environment.** Linux, Python 3.13.9 via uv venv, RTX 4050 6 GB, 12 cores, 15 GB RAM.
MAST and PyPI reachable.

**Setup.**
- `git init`, `pyproject.toml` with dependencies, no lock file (owner's request).
- `uv pip install -e ".[dev]"`. batman 2.5.1 needs `distutils`, which Python 3.13 removed,
  so `setuptools` was added as a runtime dependency. Verified imports:
  lightkurve 2.6.0, batman 2.5.1, astropy 8.0.1, numpy 2.5.4.

**Target selection, sector 98.**
```
stargazer targets --sector 98 --n 60 --seed 1
```
Output: `data/targets_s98.json`. 13,000 stars have SPOC 2-minute data in sector 98.
435 of them have a TOI and were excluded. From a seeded draw of 240, the cuts rejected
138 for Tmag, 28 for Teff and 33 for radius. 41 of the 60 requested passed. The run
took 4.5 minutes, mostly the MAST observation query. The targets file records the TOI
CSV SHA-256 and download time.

**Fetch.**
```
stargazer fetch --targets data/targets_s98.json
```
41 cached, 0 failed, about 40 seconds for each download. `stargazer verify data/cache`
passes. Log: `results/fetch_s98.log` (not in git).

**Design changes during the day.** Recorded here because they change what a reader
sees in the code.
- SDK first: `Detector` protocol, `EpisodeSet`, `evaluate`, `Results`. The CLI is a thin layer.
- Absolute imports only, enforced by ruff TID252.
- Equations live in numpydoc Notes sections with units and sources.
- wotan removed. Detrending uses `lightkurve.LightCurve.flatten` (scipy Savitzky-Golay).
  Reason: wotan has no recent release; lightkurve is a standard, maintained dependency.
- Provenance is three classes: `Digest`, `Manifest`, `GitState`. Hashing uses
  `hashlib.file_digest` from the standard library. `GitState` exists so a results file
  names the exact code behind its numbers; a version string cannot do that between
  releases.
- Limb darkening is a replaceable table with a binary search, not an `if` chain.

**Episode set s98_v1.**
```
stargazer build --targets data/targets_s98.json --n 200 --seed 1 --out data/episodes/s98_v1
stargazer verify data/episodes/s98_v1
```
Episode set hash `af9db8ecce57d49bc9e54e83c085de9d229166c8f5a3d45bfc9850a8867832e5`.
Depth range 100 ppm to 1 percent (the first build at 200 ppm to 2 percent had a median
SNR of 38, too easy, and was discarded before any run). Composition: 57 null, 11
injected with no transit in the data (scored as null), 132 with at least one transit.
Expected SNR percentiles 10/25/50/75/90: 1.9, 6.5, 20.9, 77.6, 244.4. 36 episodes are
below SNR 7, 28 between 7 and 20, 68 above 20. Two episodes have one transit and two
have two. All 41 hosts are used. Size 62 MB, not in git; the hash is the reference.

**Baseline run, in progress.**
```
stargazer baseline --episodes data/episodes/s98_v1 --out results/baseline_bls_s98_v1_sde7.json
```
Default config: Savitzky-Golay window 0.75 days, 10-minute bins, periods 0.5 days to
half the baseline, SDE threshold 7. About 3 seconds for each episode.
