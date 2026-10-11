# Episode sets

An episode set is built in three steps, each with its own command:

| Step | Command | Function | Output |
|---|---|---|---|
| Select host stars | `stargazer targets` | {func}`~stargazer.targets.select_targets` | `data/targets_sNN.json` |
| Download light curves | `stargazer fetch` | {meth}`~stargazer.data.LightCurveCache.fetch` | `data/cache/*.npz` |
| Inject and write | `stargazer build` | {meth}`~stargazer.episodes.EpisodeSet.build` | `data/episodes/<name>/` |

## Light curves

- Source: TESS Science Processing Operations Center (SPOC) 2-minute cadence products
  {cite:p}`jenkins2016`, from MAST through lightkurve.
- Flux column: PDCSAP, corrected for instrumental systematics by the pipeline.
- Cleaning: flagged and non-finite points are removed.
- Normalisation: the flux is divided by its median.

## Host selection

{func}`~stargazer.targets.select_targets` applies these steps:

1. Get the list of SPOC 2-minute targets in the sector from MAST.
2. Draw a seeded random subset of `n` stars.
3. Get their TESS Input Catalog (TIC) parameters {cite:p}`stassun2019`.
4. Apply the cuts below.
5. Remove each star on the ExoFOP TOI list {cite:p}`guerrero2021`.

| Parameter | Cut | Reason |
|---|---|---|
| TESS magnitude | $T_\mathrm{mag} \le 12$ | enough photons for shallow transits |
| Effective temperature | $3000 \le T_\mathrm{eff} \le 6500$ K | covered by the limb-darkening table |
| Radius | $R_\star \le 1.5\,R_\odot$ | dwarfs, where small planets give measurable depths |
| Luminosity class | DWARF or unknown | removes giants |
| Surface gravity and radius | finite | needed for $a/R_\star$ |

The targets file records the count after each step, the seed, the cuts, the hash of the
TOI table and its download time.

## Injected parameters

{func}`~stargazer.inject.sample_truth` draws the free parameters of each transit, in
this order:

| Parameter | Distribution | Default range |
|---|---|---|
| Period $P$ | log-uniform | 0.5 to 20 days |
| Depth $\delta$ | log-uniform | 100 ppm to 1 percent |
| Impact parameter $b$ | uniform | 0 to 0.9 |
| Epoch $t_0$ | uniform | $t_\mathrm{start}$ to $t_\mathrm{start} + P$ |

A log-uniform draw gives equal weight to each decade of period and depth. The epoch is
drawn within the first period, so a long period can leave one transit or none in the
data.

## Derived geometry

{func}`~stargazer.inject.make_truth` derives the rest of the transit from the star.

The planet-to-star radius ratio follows from the depth, because the geometric depth is
the area ratio:

$$
k = \frac{R_p}{R_\star} = \sqrt{\delta}
$$

The stellar mass follows from the TIC surface gravity $g = 10^{\log g}$ and radius,
and Kepler's third law gives the scaled semi-major axis {cite:p}`seager2003`:

$$
M_\star = \frac{g R_\star^2}{G}, \qquad
\frac{a}{R_\star} = \frac{1}{R_\star}\left(\frac{G M_\star P^2}{4\pi^2}\right)^{1/3}
$$ (eq-a-rs)

The inclination follows from the impact parameter of a circular orbit,
$b = (a/R_\star)\cos i$, and the duration from first to fourth contact is

$$
T_{14} = \frac{P}{\pi}\arcsin\!\left(\frac{\sqrt{(1+k)^2 - b^2}}{(a/R_\star)\sin i}\right)
$$ (eq-t14)

For the Sun and Earth, {eq}`eq-a-rs` gives $a/R_\star = 215$ and {eq}`eq-t14` gives
$T_{14} = 13.1$ hours.

## Transit model

The transit is computed with batman {cite:p}`kreidberg2015` on a circular orbit, with
the quadratic limb-darkening law

$$
\frac{I(\mu)}{I(1)} = 1 - u_1(1-\mu) - u_2(1-\mu)^2
$$

where $\mu$ is the cosine of the angle between the line of sight and the surface
normal. The coefficients come from a coarse table by effective temperature, based on
{cite:t}`claret2017`:

| $T_\mathrm{eff}$ (K) | $u_1$ | $u_2$ |
|---|---|---|
| below 4000 | 0.30 | 0.35 |
| 4000 to 5000 | 0.45 | 0.20 |
| 5000 to 6000 | 0.40 | 0.25 |
| 6000 and above | 0.33 | 0.28 |

The injection is multiplicative, $F_\mathrm{out} = F_\mathrm{in} \times m(t)$, so the
noise and the stellar variability of the real light curve are kept, and in transit the
noise scales with the model as it does for a real planet.

## Derived quantities in the truth

Point-to-point scatter
: The white-noise level of the flux before injection,

  $$
  \sigma = \frac{1.4826\,\mathrm{MAD}(\Delta F)}{\sqrt{2}}
  $$

  where $\Delta F$ is the difference of consecutive flux values. The difference removes
  slow variability, the factor 1.4826 converts a median absolute deviation to a
  Gaussian standard deviation, and $\sqrt{2}$ corrects for the variance of a
  difference.

Expected signal-to-noise ratio
: The matched-filter signal-to-noise ratio of a box of the injected depth and duration
  against white noise,

  $$
  \mathrm{SNR} = \frac{\delta}{\sigma}\sqrt{N_\mathrm{in}}
  $$ (eq-snr)

  where $N_\mathrm{in}$ is the number of in-transit points over all transits in the
  data. It is the multiple event statistic of the Kepler pipeline {cite:p}`jenkins2002`
  up to the shape of the transit, and an upper bound for a detector that must also
  find the period.

Transit count
: The number of transits with at least three in-transit points. A transit inside a data
  gap does not count. A long period can give a count of one or zero.

## Files

```
data/episodes/s98_v1/
  episodes/ep_0000.npz ...   time, flux, flux_err
  truth.json                 kind, TIC, sector, star, SNR, transit count, transit parameters
  config.json                seed, ranges, git commit, targets hash
  manifest.json              SHA-256 of each file
```

An episode file holds only `time`, `flux` and `flux_err`, with the time counted from
zero. It holds no TIC ID, sector, coordinates or stellar parameters, so a detector
cannot look up the star.

## Sets built so far

| Set | Seed | Periods | Hosts | Transits in 57 days |
|---|---|---|---|---|
| `s98_v1` | 1 | 0.5 to 20 days | 39 | nearly always six or more |
| `s98_v2` | 2 | 0.5 to 60 days | 39 | single and double transits are common |

The hash of each set is in its manifest and in every results file scored on it. The
LOGBOOK has the build command of each set.
