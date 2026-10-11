# Scoring

{func}`~stargazer.score.score_episode` compares one prediction with one truth record
and returns a score between 0 and 1 with an outcome label.
{func}`~stargazer.score.summarize` aggregates the scores of an episode set.

## Outcomes

| Truth | Prediction | Outcome | Score |
|---|---|---|---|
| null | none | `true_null` | 1.0 |
| null | any detection | `false_alarm` | 0.0 |
| transit | none | `miss` | 0.0 |
| transit | periodic, period matches | `hit` or `alias` | {eq}`eq-score` |
| transit | periodic, no period match | `wrong_period` | 0.0 |
| transit | single event, epoch passes | `event_hit` | 1.0 or `single_event_credit` |
| transit | single event, epoch fails | `wrong_event` | 0.0 |

An injected episode whose transits all fall in gaps or after the end of the data has a
transit count of zero. No detector can find it, so it is scored as a null episode.

```{mermaid}
flowchart TD
    A{truth kind} -->|null| B{detected?}
    B -->|no| C[true_null, 1.0]
    B -->|yes| D[false_alarm, 0.0]
    A -->|transit| E{detected?}
    E -->|no| F[miss, 0.0]
    E -->|yes| P{period given?}
    P -->|no| S{epoch check}
    S -->|pass| SH[event_hit]
    S -->|fail| SW[wrong_event, 0.0]
    P -->|yes| G{period matches<br/>true or alias?}
    G -->|no| H[wrong_period, 0.0]
    G -->|yes| I{epoch check}
    I -->|pass| J["credit × 1.0"]
    I -->|fail| K["credit × 0.7"]
```

## Period credit

The predicted period $P'$ matches the true period $P$ at ratio $r$ when

$$
\left| \frac{P'}{r P} - 1 \right| < \epsilon_P
$$

with $\epsilon_P$ = `period_rtol` (default 0.01). The ratios are tried in order and the
first match sets the credit $c$:

| Ratio $r$ | Credit $c$ | Typical cause |
|---|---|---|
| 1 | 1.0 | correct period |
| 1/2, 2 | 0.5 | every second transit missed, or two unrelated dips paired |
| 1/3, 3 | 0.25 | next harmonic |

## Epoch check

The predicted epoch $t_0'$ is folded on the true period:

$$
\phi = \left( (t_0' - t_0 + P/2) \bmod P \right) - P/2
$$

The check passes when $|\phi| < f\,T_{14}$, with $f$ = `epoch_frac` (default 0.5). The
fold uses the true period also for an alias, so an aliased answer must still point at
a real transit.

## Score

For a periodic prediction with credit $c$ and epoch result $e \in \{0, 1\}$:

$$
s = c \, (0.7 + 0.3\,e)
$$ (eq-score)

The weights make a correct period with a wrong epoch (0.7) worth more than a half-period
alias with a correct epoch (0.5).

A single-event prediction passes the same epoch check. It scores 1.0 when the episode
has one transit in the data, and `single_event_credit` (default 0.5) when it has two or
more, because the period was then measurable.

The depth does not change the score. The row records `depth_ok`, true when the
predicted depth is within a factor of two of the truth.

## Set metrics

| Metric | Definition |
|---|---|
| `false_alarm_rate` | fraction of null episodes with a detection |
| `completeness_exact` | fraction of transit episodes with outcome `hit`, or `event_hit` on a one-transit episode |
| `completeness_with_aliases` | fraction of transit episodes with a score above zero |
| `by_snr` | `completeness_with_aliases` in expected-SNR bins: 0–5, 5–7, 7–10, 10–20, 20–50, above 50 |
| `by_n_transits` | `completeness_with_aliases` by transit count: 1, 2, 3–5, 6 or more |
| `mean_score` | mean of the episode scores |

The expected SNR is defined in {eq}`eq-snr`.

## Use as a reward

`score_episode` reads no files and uses no network, so a training loop can call it on
each rollout. The tolerances are arguments, and a curriculum can tighten them.
