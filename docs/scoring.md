# Scoring

## One episode

`score_episode` takes one prediction and one truth record. It returns a row with a
score between 0 and 1 and an outcome label.

```{mermaid}
flowchart TD
    A{truth kind} -->|null| B{detected?}
    B -->|no| C[true_null, 1.0]
    B -->|yes| D[false_alarm, 0.0]
    A -->|transit| E{detected?}
    E -->|no| F[miss, 0.0]
    E -->|yes| G{period matches<br/>true or alias?}
    G -->|no| H[wrong_period, 0.0]
    G -->|yes| I{epoch within<br/>half a duration?}
    I -->|yes| J["credit × 1.0"]
    I -->|no| K["credit × 0.7"]
```

Period credit
: 1.0 at the true period. 0.5 at one half or two times the true period. 0.25 at one
  third or three times the true period. A period matches if
  `|pred / (true × ratio) − 1| < period_rtol`. The default `period_rtol` is 0.01.

Epoch check
: The predicted epoch is folded on the true period. It passes if it is within
  `epoch_frac × duration` of a transit centre. The default `epoch_frac` is 0.5.

Depth
: The scorer flags `depth_ok` if the predicted depth is within a factor of two of the
  truth. The flag does not change the score.

Injected episode with no transit in the data
: If every transit falls in a gap or after the end of the data, `n_transits` is zero.
  The scorer treats the episode as null. "None" scores 1.0. A detection scores 0.0.

## One episode set

`summarize` aggregates the rows.

| Metric | Definition |
|--------|-----------|
| `false_alarm_rate` | fraction of null episodes with a detection |
| `completeness_exact` | fraction of transit episodes with outcome `hit` |
| `completeness_with_aliases` | fraction of transit episodes with period credit above 0 |
| `by_snr` | completeness in expected-SNR bins: 0-5, 5-7, 7-10, 10-20, 20-50, 50+ |
| `by_n_transits` | completeness by transit count: 1, 2, 3-5, 6+ |
| `mean_score` | mean of the episode scores |

## Threshold choice

A detector that reports a statistic, for example the BLS signal detection efficiency
(SDE), records it in `Prediction.extra`. `stargazer.calibrate.choose_threshold`
re-scores a results file at each null-episode statistic and returns the lowest
threshold with a false-alarm rate at or below a target. The LOGBOOK records the results
file and the target, so the threshold is a function of recorded data.

## Use as a reward

`score_episode` has no dependency on files or the network. A training loop can call it
on each rollout. The tolerances are arguments, so a curriculum can tighten them.
