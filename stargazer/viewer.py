"""A terminal viewer for the trajectories in a results file.

A results file holds, for each episode, the model's conversation, its thinking, the
tool calls with their results, the prediction and the score. Reading that as JSON is
slow. This module first groups a conversation into turns with :func:`group_turns`.
Small pure functions then format each part of a turn. Two front ends use them:

- :func:`render_trajectory` returns a Rich renderable for one episode. The ``show``
  command prints it.
- :func:`make_viewer` returns a Textual application with three panes: the episode
  list, the trajectory and the details. The ``view`` command starts it.

Each tool result shows as one summary line (:func:`summary_line`). When the reader
expands it, :func:`result_detail` shows it as a table or as aligned rows, not as JSON.

Keys in the viewer:

- ``j`` and ``k`` (or the arrow keys in the list) select the next or previous episode.
- ``n`` and ``p`` jump to the next or previous turn, or search match.
- ``enter`` expands or collapses the results of the current turn.
- ``t`` expands or collapses all thinking. ``r`` expands or collapses all results.
- ``f`` cycles the outcome filter of the episode list.
- ``/`` searches the trajectory. ``escape`` clears the search. ``q`` quits.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rich import box
from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.panel import Panel
from rich.rule import Rule
from rich.segment import Segment
from rich.style import Style
from rich.table import Table
from rich.text import Text

from stargazer.episodes import EpisodeSet
from stargazer.evaluate import Results

if TYPE_CHECKING:
    from textual.app import App

LONG_LIST = 8
"""A numeric list with more values than this is summarised in pretty JSON."""

RESULT_WIDTH = 96
"""Width in characters below which a JSON value stays on one line."""

ARGUMENT_WIDTH = 60
"""Maximum characters of one argument value in a turn header."""

SUMMARY_WIDTH = 110
"""Maximum characters of the generic one-line summary of a tool result."""

SHORT_THINKING = 60
"""Thinking shorter than this many characters shows inline, without a collapsible header."""

SPARK_LEVELS = "▁▂▃▄▅▆▇█"
"""The eight block characters of a sparkline, from the lowest value to the highest."""

OUTCOME_TONE = {
    "hit": "success",
    "event_hit": "success",
    "true_null": "success",
    "alias": "warning",
    "wrong_period": "warning",
    "wrong_event": "warning",
    "miss": "error",
    "false_alarm": "error",
}
"""Tone of each scorer outcome. The Textual viewer maps a tone to a theme variable."""

RICH_TONE = {"success": "green", "warning": "yellow", "error": "red", "muted": "dim"}
"""Rich style of each tone for console output. Named ANSI colours follow the terminal palette."""

QUANTITIES = (
    ("period", "d", 1.0, "{:.5f}"),
    ("t0", "d", 1.0, "{:.4f}"),
    ("depth", "ppm", 1e6, "{:.0f}"),
    ("duration", "h", 24.0, "{:.2f}"),
)
"""Transit quantities in the details pane: key, unit (d is days, h is hours), factor from the
stored unit, format."""

PEAK_COLUMNS = (
    ("period", "period", "{:.5f}"),
    ("t0", "t0", "{:.4f}"),
    ("duration", "dur", "{:.3f}"),
    ("depth_ppm", "depth", "{:,.0f}"),
    ("sde", "sde", "{:.2f}"),
    ("n_events_in_data", "events", "{}"),
    ("half_period", "half", "{:.4f}"),
    ("double_period", "double", "{:.4f}"),
)
"""Columns of the ``bls_peaks`` table: result key, header, format. Times are in days, depth in ppm."""

EVENT_COLUMNS = (
    ("time", "t", "{:.3f}"),
    ("duration", "dur", "{:.3f}"),
    ("depth_ppm", "depth", "{:,.0f}"),
    ("snr", "snr", "{:.1f}"),
    ("n_points", "points", "{:,}"),
)
"""Columns of the ``single_events`` table: result key, header, format. Times are in days, depth in ppm."""

EPISODE_HEADER = f"{'episode':<8} {'snr':>6} {'outcome':<12} {'score':>5} {'calls':>5}"
"""Column header of the episode list. :func:`episode_label` uses the same widths."""


@dataclass(frozen=True)
class EpisodeRow:
    """One line of the episode list.

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
    error : bool
        The run raised an error or ended without a submission.
    """

    episode_id: str
    kind: str
    snr: float
    outcome: str
    score: float
    n_calls: int
    stop: str
    error: bool = False


FILTERS: dict[str, Callable[[EpisodeRow], bool]] = {
    "all": lambda row: True,
    "hit": lambda row: row.outcome == "hit",
    "miss": lambda row: row.outcome == "miss",
    "false_alarm": lambda row: row.outcome == "false_alarm",
    "wrong_period": lambda row: row.outcome == "wrong_period",
    "errors": lambda row: row.error,
}
"""Outcome filters of the episode list, in the order that ``f`` cycles them."""


@dataclass(frozen=True)
class ToolCall:
    """One tool call of a turn with its result.

    Parameters
    ----------
    name : str
        Tool name.
    arguments : str
        Arguments as the model wrote them, usually a JSON object.
    result : str
        Content of the tool message, empty if no result came back.
    """

    name: str
    arguments: str
    result: str = ""


@dataclass(frozen=True)
class Turn:
    """One assistant turn of a conversation and what followed it.

    Parameters
    ----------
    number : int
        Turn number, from 1.
    thinking : str
        The thinking trace of the turn, empty if the model gave none.
    text : str
        The visible text of the reply.
    calls : tuple of ToolCall
        The tool calls of the reply, each with its result.
    nudges : tuple of str
        User messages that the agent loop sent after the turn.
    """

    number: int
    thinking: str = ""
    text: str = ""
    calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    nudges: tuple[str, ...] = field(default_factory=tuple)


def episode_rows(results: Results, truth: dict) -> list[EpisodeRow]:
    """Build the list rows for every episode in a results file, in episode order."""
    rows = []
    for episode_id in results.rows:
        prediction = results.preds.get(episode_id) or {}
        record = truth[episode_id]
        error = str(prediction.get("error", ""))
        stopped = prediction.get("stop")
        stop = str(stopped or error[:20] or "")
        rows.append(
            EpisodeRow(
                episode_id=episode_id,
                kind=record["kind"],
                snr=float(record.get("snr", 0.0) or 0.0),
                outcome=results.rows[episode_id]["outcome"],
                score=float(results.rows[episode_id]["score"]),
                n_calls=int(prediction.get("n_tool_calls") or 0),
                stop=stop,
                error=bool(error) or (stopped is not None and stopped != "submitted"),
            )
        )
    return rows


def filter_rows(rows: Sequence[EpisodeRow], name: str) -> list[EpisodeRow]:
    """Return the rows that the named filter of :data:`FILTERS` keeps."""
    return [row for row in rows if FILTERS[name](row)]


def next_filter(name: str) -> str:
    """Return the filter after ``name`` in :data:`FILTERS`, back to the first after the last."""
    names = list(FILTERS)
    return names[(names.index(name) + 1) % len(names)]


def group_turns(prediction: dict) -> list[Turn]:
    """Group a stored conversation into turns.

    Each assistant message starts a turn. The turn takes the thinking at the same
    position, the tool results that answer its calls, and the user messages (nudges)
    that come before the next assistant message. The user messages before the first
    assistant message are the task prompt and are not part of a turn.

    Parameters
    ----------
    prediction : dict
        The stored prediction with ``conversation`` and ``thinking``.

    Returns
    -------
    list of Turn
    """
    thinking = prediction.get("thinking") or []
    drafts: list[dict] = []
    for message in prediction.get("conversation") or []:
        role = message.get("role")
        if role == "assistant":
            number = len(drafts) + 1
            calls = [
                {
                    "id": call.get("id"),
                    "name": call["function"]["name"],
                    "arguments": call["function"].get("arguments") or "",
                    "result": "",
                }
                for call in message.get("tool_calls") or []
            ]
            drafts.append(
                {
                    "number": number,
                    "thinking": (thinking[number - 1] if number <= len(thinking) else "") or "",
                    "text": message.get("content") or "",
                    "calls": calls,
                    "nudges": [],
                }
            )
        elif role == "tool" and drafts:
            _attach_result(drafts[-1]["calls"], message)
        elif role == "user" and drafts:
            drafts[-1]["nudges"].append(message.get("content") or "")
    return [
        Turn(
            number=draft["number"],
            thinking=draft["thinking"].strip(),
            text=draft["text"].strip(),
            calls=tuple(ToolCall(call["name"], call["arguments"], call["result"]) for call in draft["calls"]),
            nudges=tuple(draft["nudges"]),
        )
        for draft in drafts
    ]


def _attach_result(calls: list[dict], message: dict) -> None:
    """Put a tool message on the call with its id, or else on the first call without a result."""
    content = message.get("content") or ""
    by_id = [call for call in calls if call["id"] is not None and call["id"] == message.get("tool_call_id")]
    open_calls = by_id or [call for call in calls if not call["result"]]
    if open_calls:
        open_calls[0]["result"] = content
    else:
        calls.append({"id": None, "name": "?", "arguments": "", "result": content})


def _parse(text: str) -> object:
    """Return the JSON value of a text, or None if the text is not JSON."""
    try:
        return json.loads(text) if text else None
    except json.JSONDecodeError:
        return None


def format_arguments(arguments: str, width: int = ARGUMENT_WIDTH) -> str:
    """Write tool arguments as ``key=value`` pairs and shorten long values.

    Parameters
    ----------
    arguments : str
        The arguments as a JSON object. Other text is shortened and returned as it is.
    width : int
        Maximum characters of one value.
    """
    data = _parse(arguments) if arguments else {}
    if data is None:
        return _shorten(arguments, width)
    if not isinstance(data, dict):
        return _shorten(json.dumps(data), width)
    return ", ".join(f"{key}={_shorten(json.dumps(value), width)}" for key, value in data.items())


def _shorten(text: str, width: int) -> str:
    """Cut a text to ``width`` characters and mark the cut with an ellipsis."""
    return text if len(text) <= width else text[: width - 1] + "…"


def turn_header(turn: Turn) -> str:
    """Return the header line of a turn: its number and each call with its arguments."""
    calls = "  ·  ".join(f"{call.name}({format_arguments(call.arguments)})" for call in turn.calls)
    flags = "  [nudged]" if turn.nudges else ""
    return f"Turn {turn.number}  {calls or 'no tool call'}{flags}"


def thinking_title(thinking: str) -> str | None:
    """Return the collapsible title of a thinking trace, or None if it is short enough to show inline."""
    if len(thinking) < SHORT_THINKING:
        return None
    return f"thinking · {len(thinking):,} chars"


def _is_number(value: object) -> bool:
    """Return True for an int or a float, but not for a bool."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_scalar(value: object) -> bool:
    """Return True for a value that is not a list or a dict."""
    return not isinstance(value, list | dict)


def _is_flat(value: object) -> bool:
    """Return True for a dict whose values are all scalars."""
    return isinstance(value, dict) and all(_is_scalar(item) for item in value.values())


def _grouped(value: float) -> str:
    """Write a number with thousands separators, six significant digits and a true minus sign."""
    return f"{value:,.6g}".replace("-", "\u2212")


def summarize_numbers(values: Sequence[float]) -> str:
    """Summarise a list of numbers as its count, minimum and maximum."""
    finite = [value for value in values if _is_number(value) and math.isfinite(value)]
    if not finite:
        return f"[{len(values)} values]"
    return f"[{len(values)} values, min {_grouped(min(finite))} … max {_grouped(max(finite))}]"


def sparkline(values: Sequence[float | None]) -> str:
    """Draw values as one line of block characters, lowest value ``▁``, highest ``█``.

    A missing value (an empty bin) is a space, so a gap is visible. If all values are
    the same, each is ``▁``.
    """
    finite = [value for value in values if _is_number(value) and math.isfinite(value)]
    if not finite:
        return " " * len(values)
    low, span = min(finite), max(finite) - min(finite)
    top = len(SPARK_LEVELS) - 1
    characters = []
    for value in values:
        if not (_is_number(value) and math.isfinite(value)):
            characters.append(" ")
        else:
            level = round((value - low) / span * top) if span > 0 else 0
            characters.append(SPARK_LEVELS[level])
    return "".join(characters)


def _inline(value: object) -> str:
    """Write a JSON value on one line, with long numeric lists summarised."""
    if isinstance(value, dict):
        return "{" + ", ".join(f"{json.dumps(key)}: {_inline(item)}" for key, item in value.items()) + "}"
    if isinstance(value, list):
        if len(value) > LONG_LIST and all(_is_number(item) for item in value):
            return summarize_numbers(value)
        return "[" + ", ".join(_inline(item) for item in value) + "]"
    return json.dumps(value, ensure_ascii=False)


def _block(value: object, indent: int, used: int, width: int) -> str:
    """Write a JSON value on one line if it fits, else one item per line."""
    flat = _inline(value)
    fits = indent + used + len(flat) <= width
    if fits or _is_scalar(value) or (isinstance(value, list) and all(_is_scalar(item) for item in value)):
        return flat
    pad = " " * (indent + 2)
    if isinstance(value, dict):
        lines = [
            f"{pad}{json.dumps(key)}: {_block(item, indent + 2, len(json.dumps(key)) + 2, width)}"
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(lines) + "\n" + " " * indent + "}"
    lines = [
        f"{pad}{_inline(item) if _is_flat(item) else _block(item, indent + 2, 0, width)}" for item in value
    ]
    return "[\n" + ",\n".join(lines) + "\n" + " " * indent + "]"


def format_result(content: str, width: int = RESULT_WIDTH) -> str:
    """Pretty-print a tool result as JSON with long numeric lists summarised.

    A value that fits in ``width`` characters stays on one line. A longer dict or list
    gets one item per line. A list of more than :data:`LONG_LIST` numbers becomes its
    count, minimum and maximum.

    Parameters
    ----------
    content : str
        The tool message content. Text that is not JSON is returned as it is.
    width : int
        Line width in characters.
    """
    data = _parse(content)
    if data is None:
        return content
    return _block(data, 0, 0, width)


def _field(record: dict, key: str, pattern: str = "{}") -> str:
    """Format one value of a record, or a question mark if it is missing or of the wrong type."""
    value = record.get(key)
    if value is None:
        return "?"
    try:
        return pattern.format(value)
    except (TypeError, ValueError):
        return str(value)


def _summarize_line(arguments: dict, data: dict) -> str:
    """Summarise a ``summarize`` result: points, span, cadence, scatter, flux range, gaps."""
    return (
        f"n={_field(data, 'n_points', '{:,}')} pts · {_field(data, 'span_days', '{:.1f}')} d · "
        f"{_field(data, 'cadence_minutes', '{:.1f}')} min · "
        f"scatter {_field(data, 'point_scatter_ppm', '{:,.0f}')} ppm · "
        f"flux {_field(data, 'flux_min', '{:.3f}')}\u2013{_field(data, 'flux_max', '{:.3f}')} · "
        f"{len(data.get('gaps_days') or [])} gaps"
    )


def _peaks_line(arguments: dict, data: dict) -> str:
    """Summarise a ``bls_peaks`` result: the count and the top peak."""
    peaks = data.get("peaks") or []
    if not peaks:
        return "0 peaks"
    top = peaks[0]
    events = f" ({top['n_events_in_data']} events)" if "n_events_in_data" in top else ""
    return (
        f"{len(peaks)} peaks · top P={_field(top, 'period', '{:.3f}')} d sde {_field(top, 'sde', '{:.1f}')} "
        f"depth {_field(top, 'depth_ppm', '{:,.0f}')} ppm{events}"
    )


def _signals_line(arguments: dict, data: dict) -> str:
    """Summarise a ``search_signals`` result: the count, the top signal and the residual."""
    signals = data.get("signals") or []
    residual = (
        f" · residual sde {_field(data, 'residual_sde', '{:.1f}')}"
        if data.get("residual_sde") is not None
        else ""
    )
    if not signals:
        return f"0 signals{residual}"
    top = signals[0]
    period, sde = _field(top, "period", "{:.3f}"), _field(top, "sde", "{:.1f}")
    depth = _field(top, "depth_ppm", "{:,.0f}")
    return f"{len(signals)} signals · top P={period} d sde {sde} depth {depth} ppm{residual}"


def _events_line(arguments: dict, data: dict) -> str:
    """Summarise a ``single_events`` result: the count and the first event."""
    events = data.get("events") or []
    if not events:
        return "0 events"
    top = events[0]
    return (
        f"{len(events)} events · top t={_field(top, 'time', '{:.2f}')} d snr {_field(top, 'snr', '{:.1f}')} "
        f"depth {_field(top, 'depth_ppm', '{:,.0f}')} ppm"
    )


def _fold_line(arguments: dict, data: dict) -> str:
    """Summarise a ``fold`` result: the depth, the bins and the empty bins."""
    flux = data.get("flux_ppm") or []
    empty = data.get("n_empty_bins", sum(value is None for value in flux))
    return f"depth {_field(data, 'depth_ppm', '{:,.0f}')} ppm · {len(flux)} bins · {empty} empty"


def _odd_even_line(arguments: dict, data: dict) -> str:
    """Summarise an ``odd_even`` result: the verdict, both depths with their events, the difference."""
    odd_events = data.get("odd_events", data.get("odd_n", "?"))
    even_events = data.get("even_events", data.get("even_n", "?"))
    verdict = f"verdict {data['verdict']} · " if "verdict" in data else ""
    return (
        f"{verdict}odd {_field(data, 'odd_depth_ppm', '{:,.0f}')} ppm ({odd_events} ev) · "
        f"even {_field(data, 'even_depth_ppm', '{:,.0f}')} ppm ({even_events} ev) · "
        f"{_field(data, 'difference_sigma', '{:.1f}')}\u03c3"
    )


def _submit_line(arguments: dict, data: dict) -> str:
    """Summarise a ``submit_prediction`` call from its arguments: detection, period or event, epoch."""
    if not arguments.get("detected"):
        return "detected=False"
    if arguments.get("period") is None:
        return f"detected=True single event t0={_field(arguments, 't0', '{:.3f}')} d"
    return (
        f"detected=True P={_field(arguments, 'period', '{:.4f}')} d t0={_field(arguments, 't0', '{:.3f}')} d"
    )


SUMMARY_LINES: dict[str, Callable[[dict, dict], str]] = {
    "summarize": _summarize_line,
    "bls_peaks": _peaks_line,
    "search_signals": _signals_line,
    "single_events": _events_line,
    "fold": _fold_line,
    "odd_even": _odd_even_line,
    "submit_prediction": _submit_line,
}
"""One-line summary of the result of each tool. Other tools get one line of JSON."""


def result_error(call: ToolCall) -> str | None:
    """Return the error text of a tool result, or None if the call did not fail."""
    data = _parse(call.result)
    if isinstance(data, dict) and "error" in data:
        return str(data["error"])
    return None


def summary_line(call: ToolCall) -> str:
    """Return the one-line summary of a tool call and its result.

    A failed call gives its error text. A tool in :data:`SUMMARY_LINES` gives its own
    summary. Any other tool, or a result with an unexpected layout, gives one line of
    JSON with long lists summarised, cut to :data:`SUMMARY_WIDTH` characters.
    """
    if not call.result:
        return "no result"
    error = result_error(call)
    if error is not None:
        return error
    data = _parse(call.result)
    arguments = _parse(call.arguments)
    summarise = SUMMARY_LINES.get(call.name)
    if summarise is not None and isinstance(data, dict):
        try:
            return summarise(arguments if isinstance(arguments, dict) else {}, data)
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
    text = call.result if data is None else _inline(data)
    return _shorten(" ".join(text.split()), SUMMARY_WIDTH)


def result_sparkline(call: ToolCall) -> str:
    """Return the sparkline of the folded flux of a ``fold`` result, or an empty text for any other call."""
    data = _parse(call.result)
    if call.name != "fold" or not isinstance(data, dict) or not isinstance(data.get("flux_ppm"), list):
        return ""
    return sparkline(data["flux_ppm"])


def _value_text(value: object) -> str:
    """Write one result value for a table cell: numbers grouped, lists summarised."""
    if _is_number(value):
        return f"{value:,}".replace("-", "\u2212") if isinstance(value, int) else _grouped(value)
    if isinstance(value, str):
        return value
    return _inline(value)


def key_value_rows(data: dict, skip: Iterable[str] = ()) -> list[tuple[str, str]]:
    """Return ``(key, value)`` text pairs for the fields of a result, without the keys in ``skip``."""
    skipped = set(skip)
    return [(key, _value_text(value)) for key, value in data.items() if key not in skipped]


def _key_value_table(rows: Sequence[tuple[str, str]]) -> Table:
    """Return the aligned ``key  value`` rows as a borderless grid."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold")
    grid.add_column()
    for key, value in rows:
        grid.add_row(key, value)
    return grid


def record_table(records: Sequence[dict], columns: Sequence[tuple[str, str, str]]) -> Table:
    """Return a table with one row per record and the columns that any record has.

    Parameters
    ----------
    records : sequence of dict
        For example the peaks of a ``bls_peaks`` result.
    columns : sequence of (key, header, format)
        For example :data:`PEAK_COLUMNS`.
    """
    present = [column for column in columns if any(column[0] in record for record in records)]
    table = Table(
        box=box.SIMPLE_HEAD,
        show_edge=False,
        pad_edge=False,
        header_style="bold",
        padding=(0, 1, 0, 0),
        caption="times in days, depth in ppm",
        caption_justify="left",
        caption_style="dim",
    )
    for _, header, _ in present:
        table.add_column(header, justify="right")
    for record in records:
        table.add_row(*(_field(record, key, pattern) for key, _, pattern in present))
    return table


def _records_detail(key: str, columns: Sequence[tuple[str, str, str]]) -> Callable[[dict], RenderableType]:
    """Return a renderer that shows the list ``key`` as a table and the other fields as rows."""

    def render(data: dict) -> RenderableType:
        table = record_table(data.get(key) or [], columns)
        return Group(table, _key_value_table(key_value_rows(data, skip=(key,))))

    return render


def _rows_detail(data: dict) -> RenderableType:
    """Show every field of a result as an aligned ``key  value`` row."""
    return _key_value_table(key_value_rows(data))


def _fold_detail(data: dict) -> RenderableType:
    """Show a fold as a table of phase and flux and the other fields.

    The summary line above the table already holds the sparkline.
    """
    flux = data.get("flux_ppm") or []
    phase = data.get("phase") or [None] * len(flux)
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="bold")
    table.add_column("phase", justify="right")
    table.add_column("flux ppm", justify="right")
    for phase_value, flux_value in zip(phase, flux, strict=False):
        table.add_row(_value_text(phase_value), "—" if flux_value is None else _value_text(flux_value))
    rows = key_value_rows(data, skip=("phase", "flux_ppm", "n"))
    return Group(table, _key_value_table(rows))


RESULT_DETAILS: dict[str, Callable[[dict], RenderableType]] = {
    "bls_peaks": _records_detail("peaks", PEAK_COLUMNS),
    "single_events": _records_detail("events", EVENT_COLUMNS),
    "summarize": _rows_detail,
    "odd_even": _rows_detail,
    "fold": _fold_detail,
}
"""Expanded view of the result of each tool. Other tools get pretty JSON."""


def result_detail(call: ToolCall) -> RenderableType:
    """Return the expanded view of a tool result.

    A failed call shows its error in red and its other fields as rows. A tool in
    :data:`RESULT_DETAILS` gets its table or rows. Any other result is pretty JSON
    with long numeric lists summarised.
    """
    data = _parse(call.result)
    if isinstance(data, dict) and "error" in data:
        rows = key_value_rows(data, skip=("error",))
        return Group(Text(str(data["error"]), style="red"), _key_value_table(rows))
    render = RESULT_DETAILS.get(call.name)
    if render is not None and isinstance(data, dict):
        try:
            return render(data)
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
    return Text(format_result(call.result))


def matching_turns(turns: Sequence[Turn], query: str) -> list[int]:
    """Return the positions of the turns that contain ``query``, without regard to case."""
    needle = query.casefold()
    if not needle:
        return []
    found = []
    for position, turn in enumerate(turns):
        parts = [turn.thinking, turn.text, *turn.nudges]
        parts += [part for call in turn.calls for part in (call.name, call.arguments, call.result)]
        if any(needle in part.casefold() for part in parts):
            found.append(position)
    return found


def adjacent_index(targets: Sequence[int], current: int, direction: int) -> int | None:
    """Return the first target after ``current``, or the last target before it.

    Parameters
    ----------
    targets : sequence of int
        Positions that the cursor can go to, in increasing order.
    current : int
        The cursor position. Use -1 for "before the first turn".
    direction : int
        +1 to go forward, -1 to go back.

    Returns
    -------
    int or None
        The new position, or None if no target is in that direction.
    """
    if direction > 0:
        return next((target for target in targets if target > current), None)
    return next((target for target in reversed(targets) if target < current), None)


def _number(value: object, factor: float, pattern: str) -> str:
    """Format a stored number in display units, or a dash if it is missing."""
    if not _is_number(value) or not math.isfinite(float(value)):
        return "—"
    return pattern.format(float(value) * factor)


def comparison_rows(prediction: dict, truth: dict) -> list[tuple[str, str, str]]:
    """Return ``(quantity, truth, prediction)`` rows for the quantities in :data:`QUANTITIES`.

    A null episode has no true value. A prediction without a detection has no
    predicted value. A missing value is a dash.
    """
    transit = truth.get("transit") or {}
    detected = bool(prediction.get("detected"))
    rows = []
    for key, unit, factor, pattern in QUANTITIES:
        predicted = prediction.get(key) if detected else None
        rows.append(
            (
                f"{key} ({unit})",
                _number(transit.get(key), factor, pattern),
                _number(predicted, factor, pattern),
            )
        )
    return rows


def run_facts(prediction: dict, row: dict) -> list[tuple[str, str]]:
    """Return ``(label, value)`` pairs that describe how an episode ended.

    The pairs are the outcome, the score, the stop reason, the counts of tool calls,
    turns and nudges, and the tokens, cost and time if the prediction holds them.
    """
    usage = prediction.get("usage") or {}
    facts = [
        ("outcome", str(row.get("outcome"))),
        ("score", f"{float(row.get('score') or 0.0):.2f}"),
        ("stop", str(prediction.get("stop") or "—")),
        ("tool calls", str(prediction.get("n_tool_calls", "—"))),
        ("turns", str(prediction.get("n_turns", "—"))),
        ("nudges", str(prediction.get("n_nudges", 0))),
    ]
    if "input_tokens" in usage or "output_tokens" in usage:
        facts.append(("tokens in", f"{int(usage.get('input_tokens') or 0):,}"))
        facts.append(("tokens out", f"{int(usage.get('output_tokens') or 0):,}"))
    if "usd" in usage:
        facts.append(("cost", f"${float(usage['usd']):.4f}"))
    if "seconds" in prediction:
        facts.append(("time", f"{float(prediction['seconds']):.1f} s"))
    return facts


def _two_column_table(*headers: str) -> Table:
    """Return an empty table with the house style: no box, bold header."""
    table = Table(*headers, box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="bold")
    table.columns[0].style = "bold"
    return table


def summary_tables(prediction: dict, truth: dict, row: dict) -> tuple[Table, Table]:
    """Return the truth-against-prediction table and the run facts table of one episode."""
    comparison = _two_column_table("", "truth", "prediction")
    for quantity, true_value, predicted in comparison_rows(prediction, truth):
        comparison.add_row(quantity, true_value, predicted)
    facts = _two_column_table("run", "")
    for label, value in run_facts(prediction, row):
        facts.add_row(label, value)
    return comparison, facts


class LeftBar:
    """A Rich renderable that draws a bar (│) left of every line of another renderable.

    Parameters
    ----------
    renderable : RenderableType
        The content, wrapped to the width that is left after the bar.
    style : str
        Rich style of the bar.
    """

    def __init__(self, renderable: RenderableType, style: str = "dim") -> None:
        self.renderable = renderable
        self.style = style

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        """Render the content two columns narrower and put the bar before each line."""
        inner = options.update(width=max(1, options.max_width - 2))
        bar = Segment("│ ", Style.parse(self.style))
        for line in console.render_lines(self.renderable, inner, pad=False):
            yield bar
            yield from line
            yield Segment.line()


def _indent(renderable: RenderableType, columns: int = 2) -> RenderableType:
    """Indent a renderable by some columns."""
    grid = Table.grid()
    grid.add_column(width=columns)
    grid.add_column()
    grid.add_row("", renderable)
    return grid


def render_trajectory(
    prediction: dict, truth: dict, row: dict, show_thinking: bool = True, full_results: bool = False
) -> RenderableType:
    """Render one episode as Rich renderables for console output.

    The output has the summary tables, any error, then one block per turn: a rule and
    the header line, the thinking in italic behind a dim bar, the text of the reply,
    one summary line per tool result and each nudge in magenta. The reasoning comes last.

    Parameters
    ----------
    prediction : dict
        The stored prediction with its ``conversation`` and ``thinking``.
    truth : dict
        The truth record.
    row : dict
        The score row.
    show_thinking : bool
        Include the per-turn thinking.
    full_results : bool
        Show each tool result expanded under its summary line.
    """
    tone = RICH_TONE[OUTCOME_TONE.get(str(row.get("outcome")), "muted")]
    comparison, facts = summary_tables(prediction, truth, row)
    kind = Text.assemble(
        ("truth ", "bold"),
        f"kind={truth.get('kind')}  snr={float(truth.get('snr', 0) or 0):.1f}  ",
        f"n_transits={truth.get('n_transits', '—')}   ",
        ("outcome ", "bold"),
        (str(row.get("outcome")), f"bold {tone}"),
    )
    tables = Table.grid(padding=(0, 4))
    tables.add_row(comparison, facts)
    parts: list[RenderableType] = [kind, tables]
    if "error" in prediction:
        parts.append(Panel(Text(str(prediction["error"])), title="error", border_style="red"))
    for turn in group_turns(prediction):
        parts.append(Rule(style="cyan"))
        parts.append(Text(turn_header(turn), style="bold cyan"))
        if show_thinking and turn.thinking:
            title = thinking_title(turn.thinking)
            heading = [Text(title, style="dim")] if title else []
            parts.append(_indent(Group(*heading, LeftBar(Text(turn.thinking, style="italic")))))
        if turn.text:
            parts.append(_indent(Text(turn.text)))
        for call in turn.calls:
            failed = result_error(call) is not None
            line = Text.assemble(
                ("▸ ", "dim"), (call.name, "bold"), "  ", (summary_line(call), "red" if failed else "")
            )
            spark = result_sparkline(call)
            parts.append(_indent(Group(line, Text(f"  {spark}")) if spark else line))
            if full_results:
                parts.append(_indent(result_detail(call), 4))
        for nudge in turn.nudges:
            parts.append(_indent(Text(f"⚠ {nudge}", style="magenta")))
    parts.append(Rule(style="dim"))
    parts.append(Text.assemble(("reasoning ", "bold"), str(prediction.get("reasoning", ""))))
    return Group(*parts)


def episode_label(row: EpisodeRow) -> str:
    """Return the Textual markup line of one episode in the list, with the outcome in its tone."""
    tone = OUTCOME_TONE.get(row.outcome, "text-muted")
    mark = "!" if row.error else " "
    outcome = f"[${tone}]{row.outcome:<12}[/]"
    return f"{row.episode_id:<8} {row.snr:>6.1f} {outcome} {row.score:>5.2f} {row.n_calls:>5}{mark}"


def load(results_path: Path) -> tuple[Results, dict]:
    """Read a results file and the truth of its episode set."""
    results = Results.load(results_path)
    return results, EpisodeSet(results.episode_set).truth


VIEWER_CSS = """
Screen { layout: vertical; }
#panes { height: 1fr; }
#list-pane { width: 43; border: round $primary-background; }
#list-pane:focus-within { border: round $accent; }
#list-head { color: $text-muted; text-style: bold; }
#episodes { height: 1fr; border: none; padding: 0; }
#trajectory { width: 1fr; border: round $primary-background; padding: 0 1; }
#trajectory:focus-within { border: round $accent; }
#details { width: 41; border: round $primary-background; padding: 0 1; }
.turn { height: auto; margin: 0 0 1 0; }
.turn-header { background: $boost; color: $accent; text-style: bold; padding: 0 1; }
.turn.current .turn-header { background: $accent 30%; }
.turn.match .turn-header { color: $warning; }
.thought { color: $text-muted; text-style: italic; border-left: solid $foreground 35%; padding: 0 0 0 1;
           margin: 0 0 0 2; }
.says { margin: 1 0 0 2; }
.nudge { color: ansi_magenta; text-style: bold; margin: 0 0 0 2; }
.error { color: $error; text-style: bold; padding: 0 1; }
.empty { color: $text-muted; padding: 1; }
Collapsible { padding: 0; margin: 0; border-top: none; background: transparent; }
Collapsible > Contents { padding: 0 0 0 2; height: auto; }
CollapsibleTitle { width: 1fr; padding: 0 0 0 2; color: $text-muted; }
.thinking > CollapsibleTitle { color: $text-muted; text-style: italic; }
.result > CollapsibleTitle { color: $foreground; }
.result.failed > CollapsibleTitle { color: $error; }
.result Contents { padding: 0 0 1 2; }
#details Static { margin: 0 0 1 0; }
#outcome { text-style: bold; }
#reasoning { color: $text-muted; }
#search { dock: bottom; display: none; }
"""
"""Layout and colours of the viewer. Colours are Textual theme variables, so the theme sets them."""


def make_viewer(results: Results, truth: dict, title: str = "") -> App:
    """Build the Textual viewer for a results file.

    Parameters
    ----------
    results : Results
        The results to browse.
    truth : dict
        The truth of the episode set, keyed by episode id.
    title : str
        Text for the title bar, for example the results path.

    Returns
    -------
    textual.app.App
        The application. Call ``run()`` to start it.
    """
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.widget import Widget
    from textual.widgets import Collapsible, Footer, Header, Input, OptionList, Static
    from textual.widgets.option_list import Option

    rows = episode_rows(results, truth)

    class Section(Collapsible):
        """A collapsible that does not scroll itself into view after a bulk toggle."""

        follow = True

        def scroll_visible(self, *args: object, **kwargs: object) -> None:
            """Scroll into view, unless a bulk toggle cleared ``follow``."""
            if self.follow:
                super().scroll_visible(*args, **kwargs)
            self.follow = True

    class TrajectoryViewer(App):
        """Three panes: the episode list, the trajectory and the details of one episode."""

        CSS = VIEWER_CSS
        BINDINGS = [
            Binding("j", "episode(1)", "next ep"),
            Binding("k", "episode(-1)", "prev ep"),
            Binding("n", "turn(1)", "next turn"),
            Binding("p", "turn(-1)", "prev turn"),
            Binding("enter", "expand_turn", "open turn", show=False),
            Binding("t", "toggle('thinking')", "thinking"),
            Binding("r", "toggle('result')", "results"),
            Binding("f", "cycle_filter", "filter"),
            Binding("slash", "search", "search", key_display="/"),
            Binding("escape", "clear_search", "clear search", show=False),
            Binding("q", "quit", "quit"),
        ]

        def __init__(self) -> None:
            super().__init__()
            self.title = title or "stargazer"
            self.expanded = {"thinking": True, "result": False}
            self.filter_name = "all"
            self.listed: list[EpisodeRow] = list(rows)
            self.current: str | None = None
            self.turns: list[Turn] = []
            self.query_text = ""
            self.matches: list[int] = []
            self.cursor = -1

        def compose(self) -> ComposeResult:
            yield Header()
            with Horizontal(id="panes"):
                with Vertical(id="list-pane"):
                    yield Static(EPISODE_HEADER, id="list-head", markup=False)
                    yield OptionList(id="episodes")
                yield VerticalScroll(id="trajectory")
                with VerticalScroll(id="details"):
                    yield Static(id="outcome")
                    yield Static(id="comparison")
                    yield Static(id="facts")
                    yield Static(id="reasoning", markup=False)
            yield Input(placeholder="search this trajectory, enter to jump, escape to clear", id="search")
            yield Footer()

        def on_mount(self) -> None:
            self.query_one("#list-pane").border_title = "episodes"
            self.query_one("#details").border_title = "details"
            self.fill_list()
            self.query_one(OptionList).focus()

        def fill_list(self) -> None:
            self.listed = filter_rows(rows, self.filter_name)
            options = self.query_one(OptionList)
            options.clear_options()
            options.add_options([Option(episode_label(row), id=row.episode_id) for row in self.listed])
            pane = self.query_one("#list-pane")
            pane.border_subtitle = f"filter: {self.filter_name}  {len(self.listed)}/{len(rows)}"
            if self.listed:
                options.highlighted = 0
            else:
                self.current = None
                self.call_after_refresh(self.show_episode, None)

        async def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
            await self.show_episode(event.option.id)

        def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
            self.action_expand_turn()

        async def show_episode(self, episode_id: str | None) -> None:
            self.current = episode_id
            scroll = self.query_one("#trajectory", VerticalScroll)
            await scroll.remove_children()
            self.cursor = -1
            if episode_id is None:
                self.turns = []
                await scroll.mount(Static("No episode matches this filter. Press f.", classes="empty"))
                return
            prediction = results.preds.get(episode_id) or {}
            self.turns = group_turns(prediction)
            self.matches = matching_turns(self.turns, self.query_text)
            scroll.border_title = f"{episode_id}  ·  {len(self.turns)} turns"
            widgets: list[Widget] = []
            if "error" in prediction:
                widgets.append(Static(f"error: {prediction['error']}", classes="error", markup=False))
            widgets += [self.turn_widget(position, turn) for position, turn in enumerate(self.turns)]
            if not widgets:
                widgets.append(Static("No conversation recorded.", classes="empty"))
            await scroll.mount_all(widgets)
            scroll.scroll_home(animate=False)
            self.show_details(prediction, truth[episode_id], results.rows[episode_id])
            self.update_subtitle()

        def turn_widget(self, position: int, turn: Turn) -> Widget:
            children: list[Widget] = [Static(turn_header(turn), classes="turn-header", markup=False)]
            if turn.thinking:
                thought = Static(turn.thinking, classes="thought", markup=False)
                heading = thinking_title(turn.thinking)
                if heading is None:
                    thought.add_class("inline-thinking")
                    thought.display = self.expanded["thinking"]
                    children.append(thought)
                else:
                    collapsed = not self.expanded["thinking"]
                    children.append(Section(thought, title=heading, collapsed=collapsed, classes="thinking"))
            if turn.text:
                children.append(Static(turn.text, classes="says", markup=False))
            for call in turn.calls:
                spark = result_sparkline(call)
                heading = f"{call.name}  {summary_line(call)}" + (f"\n{spark}" if spark else "")
                classes = "result failed" if result_error(call) is not None else "result"
                detail = Static(result_detail(call))
                collapsed = not self.expanded["result"]
                children.append(Section(detail, title=heading, collapsed=collapsed, classes=classes))
            children += [Static(f"⚠ {nudge}", classes="nudge", markup=False) for nudge in turn.nudges]
            classes = "turn match" if position in self.matches else "turn"
            return Vertical(*children, classes=classes)

        def show_details(self, prediction: dict, record: dict, row: dict) -> None:
            outcome = str(row.get("outcome"))
            tone = OUTCOME_TONE.get(outcome, "text-muted")
            snr = float(record.get("snr", 0) or 0)
            self.query_one("#outcome", Static).update(
                f"[${tone}]{outcome}[/]  score {float(row.get('score') or 0):.2f}\n"
                f"[$text-muted]{record.get('kind')}  snr {snr:.1f}  "
                f"transits {record.get('n_transits', '—')}[/]"
            )
            comparison, facts = summary_tables(prediction, record, row)
            self.query_one("#comparison", Static).update(comparison)
            self.query_one("#facts", Static).update(facts)
            reasoning = str(prediction.get("reasoning") or prediction.get("error") or "")
            self.query_one("#reasoning", Static).update(f"reasoning\n{reasoning}")

        def update_subtitle(self) -> None:
            search = f"search '{self.query_text}': {len(self.matches)} turns" if self.query_text else ""
            self.sub_title = f"{results.system}  ·  filter {self.filter_name}  {search}".strip()

        def action_episode(self, step: int) -> None:
            options = self.query_one(OptionList)
            if step > 0:
                options.action_cursor_down()
            else:
                options.action_cursor_up()

        def action_turn(self, direction: int) -> None:
            widgets = list(self.query(".turn"))
            targets = self.matches if self.query_text else range(len(widgets))
            position = adjacent_index(list(targets), self.cursor, direction)
            if position is None or position >= len(widgets):
                self.bell()
                return
            if 0 <= self.cursor < len(widgets):
                widgets[self.cursor].remove_class("current")
            self.cursor = position
            widgets[position].add_class("current")
            self.scroll_to_cursor()

        def scroll_to_cursor(self) -> None:
            self.scroll_to_turn(self.cursor)

        def action_expand_turn(self) -> None:
            widgets = list(self.query(".turn"))
            if not widgets:
                return
            if self.cursor < 0:
                self.cursor = 0
                widgets[0].add_class("current")
            collapsibles = list(widgets[self.cursor].query("Collapsible.result"))
            collapse = not any(collapsible.collapsed for collapsible in collapsibles)
            for collapsible in collapsibles:
                collapsible.collapsed = collapse

        def action_toggle(self, kind: str) -> None:
            anchor = self.top_turn()
            self.expanded[kind] = not self.expanded[kind]
            for collapsible in self.query(f"Section.{kind}"):
                collapsible.follow = False
                collapsible.collapsed = not self.expanded[kind]
            if kind == "thinking":
                for thought in self.query(".inline-thinking"):
                    thought.display = self.expanded[kind]
            self.call_after_refresh(self.scroll_to_turn, anchor)

        def top_turn(self) -> int:
            """Return the cursor turn, or else the turn at the top of the pane."""
            if self.cursor >= 0:
                return self.cursor
            scroll = self.query_one("#trajectory", VerticalScroll)
            for position, widget in enumerate(self.query(".turn")):
                if widget.virtual_region.bottom > scroll.scroll_y:
                    return position
            return 0

        def scroll_to_turn(self, position: int) -> None:
            widgets = list(self.query(".turn"))
            if 0 <= position < len(widgets):
                scroll = self.query_one("#trajectory", VerticalScroll)
                scroll.scroll_to_widget(widgets[position], top=True, animate=False)

        def action_cycle_filter(self) -> None:
            self.filter_name = next_filter(self.filter_name)
            self.fill_list()
            self.update_subtitle()

        def action_search(self) -> None:
            search = self.query_one("#search", Input)
            search.display = True
            search.value = self.query_text
            search.focus()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            self.query_text = event.value.strip()
            event.input.display = False
            self.apply_search()
            self.cursor = -1
            if self.matches:
                self.action_turn(1)
            elif self.query_text:
                self.notify(f"no turn contains '{self.query_text}'", severity="warning")
            self.query_one(OptionList).focus()

        def action_clear_search(self) -> None:
            self.query_one("#search", Input).display = False
            self.query_text = ""
            self.apply_search()
            self.query_one(OptionList).focus()

        def apply_search(self) -> None:
            self.matches = matching_turns(self.turns, self.query_text)
            for position, widget in enumerate(self.query(".turn")):
                widget.set_class(position in self.matches, "match")
            self.update_subtitle()

    return TrajectoryViewer()


def run_viewer(results_path: Path) -> None:
    """Start the Textual viewer on a results file."""
    results, truth = load(results_path)
    make_viewer(results, truth, title=str(results_path)).run()
