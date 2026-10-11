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
    E -->|yes| L{period given?}
    L -->|yes| G{period matches<br/>true or alias?}
    G -->|no| H[wrong_period, 0.0]
    G -->|yes| I{epoch within<br/>half a duration?}
    I -->|yes| J["credit × 1.0"]
    I -->|no| K["credit × 0.7"]
    L -->|no| M{t0 within half a<br/>duration of a transit?}
    M -->|no| N[wrong_event, 0.0]
    M -->|yes| O{one transit<br/>in the data?}
    O -->|yes| P[event_hit, 1.0]
    O -->|no| Q[event_hit, single_event_credit]
```

Period credit
: 1.0 at the true period. 0.5 at one half or two times the true period. 0.25 at one
  third or three times the true period. A period matches if
  `|pred / (true × ratio) − 1| < period_rtol`. The default `period_rtol` is 0.01.

Epoch check
: The predicted epoch is folded on the true period. It passes if it is within
  `epoch_frac × duration` of a transit centre. The default `epoch_frac` is 0.5.

Single-event form
: A prediction with `detected` true and no period claims one transit-like event at
  time `t0`. The event passes the epoch check above and sets `event_ok`. A credited
  event scores 1.0 if the episode has one transit in the data, because that episode
  has no measurable period and the event is the complete answer. It scores
  `single_event_credit` if the data hold two or more transits, because the period was
  there to find. The default `single_event_credit` is 0.5, the alias credit. An event
  that fails the check has outcome `wrong_event` and scores 0.0. On a null episode the
  form is a false alarm like any detection. The submit tool requires `t0`, `duration`
  and `depth` for this form, so the claim names one specific dip.

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
| `completeness_exact` | fraction of transit episodes with outcome `hit`, or `event_hit` with score 1.0 |
| `completeness_with_aliases` | fraction of transit episodes with a score above 0: a hit, an alias or a credited event |
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

`score_episode` reads no files and uses no network, so a training loop can call it on
each rollout. The tolerances are arguments and a curriculum can tighten them.
