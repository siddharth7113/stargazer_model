# Trajectories

The results file of an agent run stores, for each episode, in `preds[<episode>].extra`:

| Key | Content |
|---|---|
| `conversation` | every message that the backend received, in order |
| `thinking` | the reasoning trace of each turn, where the backend exposes one |
| `transcript` | each tool call with its arguments and its result |
| `reasoning` | the `reasoning` argument of the submission |
| `stop` | why the episode ended: `submitted`, `model_stopped`, `max_tool_calls`, `max_turns` or `max_tokens` |
| `usage` | input and output tokens, and cost when known |
| `n_tool_calls`, `n_turns`, `n_nudges` | counts |

## Terminal viewer

```bash
stargazer view --results results/full/agent_qwen38_27b_s98_v1.json
```

The viewer needs the `view` extra. It has three panes:

- **Episodes**: every episode with its expected SNR, outcome and score.
- **Trajectory**: the turns of the selected episode, each with its tool calls, the
  thinking, the model's text, one summary line for each tool result and any reminder.
- **Details**: the truth next to the prediction, the stop reason, the tokens and the
  cost.

```{image} ../assets/viewer.svg
:alt: The trajectory viewer
:width: 100%
```

| Key | Action |
|---|---|
| `j`, `k`, arrow keys | next or previous episode |
| `n`, `p` | next or previous turn, or search match |
| `enter` | expand or collapse the results of the current turn |
| `r` | expand or collapse all results as tables |
| `t` | expand or collapse all thinking |
| `f` | cycle the filter: all, hit, miss, false_alarm, wrong_period, errors |
| `/`, `escape` | search the trajectory, clear the search |
| `q` | quit |

## Console output

```bash
stargazer show --results results/full/agent_qwen38_27b_s98_v1.json --episode ep_0001
```

| Option | Effect |
|---|---|
| `--full` | print tool results in full |
| `--no-thinking` | hide the thinking trace |
| `--width N` | console width in characters, default 120 |
