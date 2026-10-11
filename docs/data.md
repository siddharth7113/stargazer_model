# Data

## Source

The light curves are TESS Science Processing Operations Center (SPOC) 2-minute cadence
products. The flux column is PDCSAP, which the pipeline has corrected for systematics.
`LightCurveCache.fetch` downloads one product with lightkurve, drops flagged and
non-finite points, normalises the flux to a median of one, and saves a `.npz` file.

## Host selection

`select_targets` performs five steps.

1. Get the list of SPOC 120 s targets in a sector from MAST.
2. Draw a seeded random subset.
3. Get the TESS Input Catalog (TIC) parameters of the subset.
4. Apply cuts: Tmag at or below 12, Teff between 3000 K and 6500 K, radius at or below
   1.5 solar radii, luminosity class DWARF or unknown.
5. Remove each star in the ExoFOP TESS Object of Interest (TOI) list.

The output file records each count, the seed, the cuts, the TOI CSV hash and the TOI
download time.

## Injection

`sample_truth` draws one injection for each transit episode.

| Parameter | Distribution | Range |
|-----------|-------------|-------|
| period | log-uniform | 0.5 to 20 days |
| depth | log-uniform | 100 ppm to 1 percent |
| impact parameter | uniform | 0 to 0.9 |
| epoch | uniform | start to start plus one period |

`make_truth` derives the rest of the geometry. The semi-major axis in stellar radii
comes from Kepler's third law and the stellar density. The density comes from `logg`
and the radius in the TIC. The limb-darkening coefficients come from a coarse table by
Teff. The truth record stores each value.

The truth record also stores two derived quantities.

Expected SNR
: `depth / sigma * sqrt(N_in_transit)`. `sigma` is the point-to-point scatter of the
  flux before injection.

Transit count
: The number of transit events with at least three in-transit points. A long period can
  give zero or one.

## Episode files

An episode file holds three arrays: `time`, `flux` and `flux_err`. The time starts at
zero. The file holds no TIC ID, no sector, no coordinates and no stellar parameters.
This prevents a detector from looking up the answer.

```{mermaid}
flowchart TB
    subgraph agent["Visible to the detector"]
        ep["ep_0042.npz<br/>time, flux, flux_err"]
    end
    subgraph hidden["Visible to the scorer only"]
        truth["truth.json<br/>kind, tic, sector, star, snr, n_transits, transit{...}"]
        cfg["config.json<br/>seed, ranges, git commit, targets hash"]
        man["manifest.json<br/>SHA-256 of each file"]
    end
```
