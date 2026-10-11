# Baseline detector

{class}`~stargazer.baseline.BLSDetector` is a classical transit search: detrend, bin,
search with Box Least Squares {cite:p}`kovacs2002`, and report a detection when the
signal detection efficiency is above a threshold. It sets the score that an agent has
to beat on the same episode set.

## Steps

1. **Detrend.** {func}`~stargazer.baseline.detrend` divides the flux by a
   Savitzky-Golay fit {cite:p}`savitzky1964` through `lightkurve.LightCurve.flatten`,
   with gaps split and transits clipped from the fit.
2. **Bin.** {func}`~stargazer.baseline.bin_lc` averages the flux into 10-minute bins.
3. **Search.** `astropy.timeseries.BoxLeastSquares` computes the power on a grid
   uniform in frequency, for six trial durations, with the log-likelihood objective.
4. **Decide.** The detector reports the best peak and compares its SDE with the
   threshold.

## Signal detection efficiency

The SDE measures how far the best peak stands above the typical peak of the
periodogram:

$$
\mathrm{SDE} = \frac{\max(p) - \langle p \rangle}{\sigma_p}
$$

where $p$ is the BLS power over the period grid, $\langle p \rangle$ its mean and
$\sigma_p$ its standard deviation. Pure white noise gives an SDE of about 5 to 7 on a
grid of this size.

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `detrend_window_days` | 0.75 | Savitzky-Golay window in days |
| `bin_minutes` | 10.0 | bin width |
| `min_period_days` | 0.5 | shortest trial period |
| `max_period_frac_of_span` | 0.5 | longest trial period as a fraction of the baseline, so two transits fit |
| `durations_days` | 0.03 to 0.25 | six trial durations |
| `frequency_factor` | 2.0 | grid coarseness; larger is faster |
| `sde_threshold` | 7.0 | detection threshold |

Pass overrides as keyword arguments, or as JSON with `stargazer baseline --config`.

## Limitations

- The detrending window must be about three times the transit duration. The default
  keeps transits up to about 6 hours; a longer transit loses depth.
- The longest trial period is half the baseline. A planet with one transit in the data
  cannot be found.
- One search returns one signal. A strong feature hides a weaker periodic signal.

(threshold-choice)=
## Threshold choice

The textbook threshold of 7 is too low for real 57-day TESS light curves, which have
residual systematics. On `s98_v1` it gives a false-alarm rate of 69 percent.
{func}`~stargazer.calibrate.choose_threshold` picks the threshold from the SDE values
recorded on the null episodes:

1. Sort the $n$ null statistics in ascending order.
2. For the $i$-th value as threshold, the false-alarm rate is $(n - i)/n$.
3. Return the first value with a rate at or below the target.

```bash
stargazer calibrate --results results/baseline_bls_s98_v1_sde7.json --max-false-alarm 0.05
```

With $n$ null episodes the smallest non-zero rate is $1/n$, so a target of 5 percent
needs at least 20 null episodes. Because every prediction stores its SDE, a results
file can be rescored at any threshold without a new search.
