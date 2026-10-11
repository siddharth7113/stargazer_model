"""A terminal viewer for the trajectories in a results file.

A results file holds, for each episode, the model's conversation, its thinking, the
tool calls with their results, the prediction and the score. Reading that as JSON is
slow. This module renders it two ways: :func:`render_trajectory` builds Rich markup
for one episode, which the ``show`` command prints, and :class:`TrajectoryViewer`
is a Textual application with an episode table on the left and the trajectory on the
right, which the ``view`` command starts.

Keys in the viewer: up and down select an episode; ``t`` toggles thinking; ``r``
toggles full tool results; ``q`` quits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rich.markup import escape

from stargazer.episodes import EpisodeSet
from stargazer.evaluate import Results


@dataclass(frozen=True)
class EpisodeRow:
    """One line of the episode table.

    Parameters
    ----------
    episode_id : str
    kind : str
        ``null`` or ``transit`` from the truth.
    snr : float
        Expected SNR of the injection, 0 for a null.
    outcome : str
        Scorer outcome label.
    score : float
    n_calls : int
        Tool calls the agent made.
    stop : str
        Why the episode ended.
    """

    episode_id: str
    kind: str
    snr: float
    outcome: str
    score: float
    n_calls: int
    stop: str


def episode_rows(results: Results, truth: dict) -> list[EpisodeRow]:
    """Build the table rows for every episode in a results file, in episode order."""
    rows = []
    for episode_id in results.rows:
        prediction = results.preds.get(episode_id) or {}
        record = truth[episode_id]
        rows.append(
            EpisodeRow(
                episode_id=episode_id,
                kind=record["kind"],
                snr=float(record.get("snr", 0.0) or 0.0),
                outcome=results.rows[episode_id]["outcome"],
                score=float(results.rows[episode_id]["score"]),
                n_calls=int(prediction.get("n_tool_calls") or 0),
                stop=str(prediction.get("stop") or prediction.get("error", "")[:20] or ""),
            )
        )
    return rows


def _format_result(content: str, full: bool, width: int = 300) -> str:
    """Pretty-print a tool result; shorten long lists unless ``full``."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return content if full else content[:width]
    if full:
        return json.dumps(data, indent=1)
    short = {}
    for key, value in data.items():
        if isinstance(value, list) and len(value) > 8:
            short[key] = f"[{len(value)} values]"
        else:
            short[key] = value
    text = json.dumps(short)
    return text if len(text) <= width else text[:width] + " ..."


def render_trajectory(
    prediction: dict, truth: dict, row: dict, show_thinking: bool = True, full_results: bool = False
) -> str:
    """Render one episode's trajectory as Rich markup.

    Parameters
    ----------
    prediction : dict
        The stored prediction with its ``conversation``, ``thinking`` and ``transcript``.
    truth : dict
        The truth record.
    row : dict
        The score row.
    show_thinking : bool
        Include the per-turn thinking.
    full_results : bool
        Print tool results in full instead of shortened.
    """
    transit = truth.get("transit", {})
    lines = [
        f"[bold]truth[/bold]  kind={truth['kind']}  snr={float(truth.get('snr', 0) or 0):.1f}  "
        f"n_transits={truth.get('n_transits')}  period={transit.get('period', float('nan')):.4f}  "
        f"t0={transit.get('t0', float('nan')):.3f}  depth={transit.get('depth', float('nan')) * 1e6:.0f} ppm",
        f"[bold]score[/bold]  outcome=[{'green' if row['score'] > 0 else 'red'}]{row['outcome']}[/]  "
        f"score={row['score']}  stop={prediction.get('stop')}  calls={prediction.get('n_tool_calls')}  "
        f"nudges={prediction.get('n_nudges', 0)}  tokens={prediction.get('usage')}",
    ]
    if "error" in prediction:
        lines.append(f"[red]error[/red] {escape(str(prediction['error']))}")
    thinking = prediction.get("thinking") or []
    turn = 0
    for message in prediction.get("conversation", []):
        role = message["role"]
        if role == "assistant":
            turn += 1
            thought = thinking[turn - 1] if turn - 1 < len(thinking) else ""
            if show_thinking and thought:
                lines.append(f"\n[dim][turn {turn} thinking][/dim]\n[dim]{escape(thought)}[/dim]")
            if message.get("content"):
                lines.append(f"\n[turn {turn} says] {escape(message['content'])}")
            for call in message.get("tool_calls", []):
                name = call["function"]["name"]
                arguments = escape(call["function"]["arguments"])
                lines.append(f"[cyan][turn {turn} calls][/cyan] [bold]{name}[/bold]({arguments})")
        elif role == "tool":
            lines.append(
                f"[yellow][result][/yellow] {escape(_format_result(message['content'], full_results))}"
            )
        elif role == "user" and turn > 0:
            lines.append(f"\n[magenta][nudge][/magenta] {escape(message['content'])}")
    lines.append(f"\n[bold]reasoning[/bold] {escape(str(prediction.get('reasoning', '')))}")
    return "\n".join(lines)


def load(results_path: Path) -> tuple[Results, dict]:
    """Read a results file and the truth of its episode set."""
    results = Results.load(results_path)
    return results, EpisodeSet(results.episode_set).truth


def run_viewer(results_path: Path) -> None:
    """Start the Textual viewer on a results file."""
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal
    from textual.widgets import DataTable, Footer, Header, RichLog, Static

    results, truth = load(results_path)
    rows = episode_rows(results, truth)

    class TrajectoryViewer(App):
        CSS = """
        Horizontal { height: 1fr; }
        #episodes { width: 58; }
        #trajectory { width: 1fr; padding: 0 1; }
        """
        BINDINGS = [
            Binding("t", "toggle_thinking", "thinking"),
            Binding("r", "toggle_results", "full results"),
            Binding("q", "quit", "quit"),
        ]

        def __init__(self) -> None:
            super().__init__()
            self.show_thinking = True
            self.full_results = False
            self.current = rows[0].episode_id if rows else None

        def compose(self) -> ComposeResult:
            yield Header()
            with Horizontal():
                yield DataTable(id="episodes", cursor_type="row")
                yield RichLog(id="trajectory", wrap=True, markup=True, auto_scroll=False)
            yield Static(f"{results_path}  system={results.system}  episodes={len(rows)}", id="status")
            yield Footer()

        def on_mount(self) -> None:
            table = self.query_one(DataTable)
            table.add_columns("episode", "kind", "snr", "outcome", "score", "calls", "stop")
            for row in rows:
                table.add_row(
                    row.episode_id,
                    row.kind,
                    f"{row.snr:.1f}",
                    row.outcome,
                    f"{row.score:.2f}",
                    str(row.n_calls),
                    row.stop,
                    key=row.episode_id,
                )
            self.render_current()

        def on_data_table_row_highlighted(self, event) -> None:
            self.current = event.row_key.value
            self.render_current()

        def action_toggle_thinking(self) -> None:
            self.show_thinking = not self.show_thinking
            self.render_current()

        def action_toggle_results(self) -> None:
            self.full_results = not self.full_results
            self.render_current()

        def render_current(self) -> None:
            log = self.query_one(RichLog)
            log.clear()
            if self.current is None:
                return
            log.write(
                render_trajectory(
                    results.preds[self.current],
                    truth[self.current],
                    results.rows[self.current],
                    show_thinking=self.show_thinking,
                    full_results=self.full_results,
                )
            )
            log.scroll_home(animate=False)

    TrajectoryViewer().run()
