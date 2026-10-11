"""Build and execute the course notebooks in ``examples/``.

Each notebook explains one part of the toolbox to a human reader, with the same
language rules as the documentation and a plot for each result. The notebooks are
generated from this file, so a change to a tool is a change here, and the built
notebooks carry their outputs for a reader without a Python environment.

Run::

    .venv/bin/python scripts/build_notebooks.py            # build and execute all
    .venv/bin/python scripts/build_notebooks.py --no-exec  # build only

A notebook uses the real episode set ``data/episodes/s98_v2`` if it is present and a
synthetic light curve otherwise, so each one runs on a fresh clone.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "examples"

PREAMBLE = '''import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path("..").resolve()))
sys.path.insert(0, str(Path("../tests").resolve()))
plt.rcParams["figure.figsize"] = (11, 3.5)
plt.rcParams["figure.dpi"] = 110

EPISODES = Path("../data/episodes/s98_v2")


def load_episode(episode_id="ep_0001"):
    """Return (time, flux, truth) for one real episode, or a synthetic one if the set is absent."""
    if EPISODES.exists():
        from stargazer.episodes import EpisodeSet

        episode_set = EpisodeSet(EPISODES)
        time, flux, _ = episode_set.load(episode_id)
        return time, flux, episode_set.truth[episode_id]
    from conftest import synthetic_lc

    from stargazer.inject import inject, make_truth

    time, flux, _ = synthetic_lc(days=57.0)
    time = time - time.min()
    star = {"teff": 5500.0, "logg": 4.45, "radius_rsun": 0.95, "tmag": 10.0, "mass_msun": 0.95}
    truth = make_truth(period=0.7764, t0=0.1453, depth=0.0016, b=0.3, star=star)
    transit = {"period": truth.period, "t0": truth.t0, "duration": truth.duration, "depth": truth.depth}
    record = {"kind": "transit", "n_transits": 73, "snr": 60.0, "transit": transit}
    return time, inject(time, flux, truth), record


time, flux, truth = load_episode()
print(f"{len(time)} points over {time.max() - time.min():.1f} days; truth kind {truth['kind']}")'''


@dataclass
class Notebook:
    """One course notebook: a file name, a title and an ordered list of cells.

    Parameters
    ----------
    name : str
        File name without the extension.
    cells : list of tuple
        ``("markdown", text)`` or ``("code", source)`` in order. The first cell is the
        introduction; the preamble that loads the episode follows it.
    """

    name: str
    cells: list[tuple[str, str]] = field(default_factory=list)

    def build(self) -> nbformat.NotebookNode:
        """Return the notebook object with the preamble as the first code cell."""
        notebook = nbformat.v4.new_notebook()
        notebook.metadata["kernelspec"] = {
            "name": "python3",
            "display_name": "Python 3",
            "language": "python",
        }
        intro, rest = self.cells[0], self.cells[1:]
        notebook.cells.append(nbformat.v4.new_markdown_cell(intro[1].strip()))
        notebook.cells.append(nbformat.v4.new_code_cell(PREAMBLE))
        for kind, text in rest:
            cell = nbformat.v4.new_markdown_cell if kind == "markdown" else nbformat.v4.new_code_cell
            notebook.cells.append(cell(text.strip()))
        return notebook

    def write(self, execute: bool) -> Path:
        """Build, optionally execute, and write the notebook. Return its path."""
        notebook = self.build()
        if execute:
            NotebookClient(
                notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(OUT)}}
            ).execute()
        path = OUT / f"{self.name}.ipynb"
        nbformat.write(notebook, path)
        return path


NOTEBOOKS: list[Notebook] = [
    Notebook(
        "01_the_episode",
        [
            (
                "markdown",
                """
# 1. The episode

An episode is one light curve: the brightness of one star against time. This course
uses the TESS 2-minute cadence. The benchmark hides a transit in most episodes and
nothing in the rest. The detection system, a model with tools, sees only the tools'
outputs. This notebook shows what the tools see.

The first cell loads one episode. It uses the real episode set `data/episodes/s98_v2`
if it is present, and a synthetic light curve with one injected planet otherwise.
""",
            ),
            (
                "markdown",
                """
## The raw light curve

Time is in days from the first point. Flux is divided by its median, so 1.0 is the
star's normal brightness and 0.99 is one percent fainter. A transit is a dip below
1.0 that lasts hours and repeats at one period.
""",
            ),
            (
                "code",
                """
fig, axis = plt.subplots()
axis.plot(time, flux, ",", color="0.3")
axis.set_xlabel("time (days)")
axis.set_ylabel("relative flux")
axis.set_title("One TESS sector of one star")
plt.show()
""",
            ),
            (
                "markdown",
                """
## What `summarize` reports

The model's first tool call is `summarize`. It returns the numbers below and
nothing else. The point scatter is the white-noise level, in parts per million
(ppm); a transit must be well deeper than it. The flux range tells whether the star
is active: a range far from 1.0 means spots or pulsation, which a period search
can mistake for a planet. The gaps are the breaks in the data, for example the
downlink in the middle of a sector.
""",
            ),
            (
                "code",
                """
from stargazer.tools import summarize

summary = summarize(time, flux)
for key, value in summary.items():
    print(f"{key:20s} {value}")
""",
            ),
            (
                "markdown",
                """
## The hidden truth

The truth record is hidden from the model. It holds what was injected: the period,
the epoch of one transit centre, the duration and the depth. The plot marks each
injected transit on the light curve, so the reader can compare what the model has
to find with what the eye sees.
""",
            ),
            (
                "code",
                """
fig, axis = plt.subplots()
axis.plot(time, flux, ",", color="0.3")
if truth["kind"] == "transit":
    transit = truth["transit"]
    n_centres = int((time.max() - transit["t0"]) / transit["period"]) + 1
    centres = transit["t0"] + transit["period"] * np.arange(n_centres)
    for centre in centres:
        axis.axvline(centre, color="tab:red", alpha=0.3, lw=0.8)
    depth_ppm = transit["depth"] * 1e6
    axis.set_title(f"Injected transits every {transit['period']:.4f} days, depth {depth_ppm:.0f} ppm")
else:
    axis.set_title("A null episode: nothing was injected")
axis.set_xlabel("time (days)")
axis.set_ylabel("relative flux")
plt.show()
""",
            ),
        ],
    ),
    Notebook(
        "02_injection",
        [
            (
                "markdown",
                """
# 2. Injection

The benchmark makes an episode by multiplying a real light curve by a transit model.
The model is batman, with the geometry from the star's radius and surface gravity.
This notebook builds one transit from its parameters and shows what each parameter
does to the shape.
""",
            ),
            (
                "markdown",
                """
## From depth and period to a shape

`make_truth` takes a period, an epoch, a depth and an impact parameter, and a host
star. It derives the radius ratio from the depth, the scaled semi-major axis from
Kepler's third law, the inclination from the impact parameter, the limb darkening
from the star's temperature, and the duration from all of them. The returned record
is the truth of the episode.
""",
            ),
            (
                "code",
                """
from stargazer.inject import make_truth, transit_model

star = {"teff": 5500.0, "logg": 4.45, "radius_rsun": 0.95, "tmag": 10.0, "mass_msun": 0.95}
example = make_truth(period=5.0, t0=2.0, depth=0.005, b=0.3, star=star)
for name, value in vars(example).items():
    print(f"{name:10s} {value:.5g}")
""",
            ),
            (
                "markdown",
                """
## The shape of one transit

The plot shows the model around one transit for three impact parameters. At b = 0
the planet crosses the centre of the disc: a long, flat-bottomed dip. At b = 0.9 it
crosses near the limb: shorter and rounder. At b above 1 minus the radius ratio the
planet never lies fully on the disc: a V shape with no flat bottom, which is the
shape of a grazing eclipsing binary too.
""",
            ),
            (
                "code",
                """
grid = np.linspace(1.8, 2.2, 2000)
fig, axis = plt.subplots()
for b in (0.0, 0.6, 0.9, 0.97):
    model = make_truth(period=5.0, t0=2.0, depth=0.005, b=b, star=star)
    change_ppm = (transit_model(grid, model) - 1) * 1e6
    axis.plot((grid - 2.0) * 24, change_ppm, label=f"b = {b}, {model.duration * 24:.2f} h")
axis.set_xlabel("hours from mid-transit")
axis.set_ylabel("model flux change (ppm)")
axis.legend()
plt.show()
""",
            ),
            (
                "markdown",
                """
## Injection into noise

`inject` multiplies the flux by the model. The plot shows the same transit in the
real noise of the loaded episode, so the reader sees how much of the shape survives.
The tools work on this.
""",
            ),
            (
                "code",
                """
from stargazer.inject import inject

injected = inject(time, flux, example)
window = np.abs(time - 2.0) < 0.3
fig, axis = plt.subplots()
axis.plot((time[window] - 2.0) * 24, injected[window], ".", ms=3, color="0.3", label="data with injection")
axis.plot((time[window] - 2.0) * 24, transit_model(time[window], example), color="tab:red", label="model")
axis.set_xlabel("hours from mid-transit")
axis.set_ylabel("relative flux")
axis.legend()
plt.show()
""",
            ),
        ],
    ),
    Notebook(
        "03_period_search",
        [
            (
                "markdown",
                """
# 3. The period search

A transit repeats. The Box Least Squares (BLS) search folds the light curve at each
trial period and asks how well a box-shaped dip fits. The tool `bls_peaks` returns
the strongest distinct peaks. The tool `search_signals` runs the search, masks the
strongest signal, and searches again until the rest is noise. This notebook shows
both on the loaded episode.
""",
            ),
            (
                "markdown",
                """
## Detrending first

Both tools detrend the light curve before the search: a Savitzky-Golay filter fits a
polynomial in a sliding window and divides the flux by the fit. The window must be
several times the transit duration or the filter erases the dip. The plot shows the
raw and the detrended flux.
""",
            ),
            (
                "code",
                """
from stargazer.baseline import detrend

flat_time, flat_flux = detrend(time, flux, 0.75)
fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(11, 5))
top.plot(time, flux, ",", color="0.3")
top.set_ylabel("raw flux")
bottom.plot(flat_time, flat_flux, ",", color="0.3")
bottom.set_ylabel("detrended flux")
bottom.set_xlabel("time (days)")
plt.show()
""",
            ),
            (
                "markdown",
                """
## `bls_peaks`

Each peak has a period, an epoch, a duration, a depth and a signal detection
efficiency (SDE): how far the peak stands above the mean of the periodogram, in
standard deviations. On real stars noise alone reaches an SDE of 10 to 14, so a peak
is a hypothesis to check, not a result. The tool also returns the half and double
period of each peak, because a search often lands on an alias of the true period.
""",
            ),
            (
                "code",
                """
from stargazer.tools import bls_peaks

peaks = bls_peaks(time, flux)
for peak in peaks["peaks"]:
    print({key: peak[key] for key in ("period", "t0", "duration", "depth_ppm", "sde", "n_events_in_data")})
""",
            ),
            (
                "markdown",
                """
## `search_signals`

A strong signal dominates the periodogram at every period. The multiple-planet
search of the mission pipelines removes each signal it accepts and searches again.
The tool returns the signals in order and the SDE of the strongest peak left, so
the reader can see where the search stopped.
""",
            ),
            (
                "code",
                """
from stargazer.tools import search_signals

found = search_signals(time, flux)
for signal in found["signals"]:
    print({key: signal[key] for key in ("rank", "period", "t0", "depth_ppm", "sde")})
print("residual sde", found["residual_sde"], "stop", found["stop"])
""",
            ),
            (
                "markdown",
                """
## The fold

A fold places every point at its phase in the orbit of the top signal. A planet
shows one dip at phase 0 and a flat curve elsewhere. The plot folds at the first
signal's period and epoch, binned to 10 minutes, with the truth's period in the
title for comparison.
""",
            ),
            (
                "code",
                """
from stargazer.baseline import bin_lc

top = found["signals"][0] if found["signals"] else peaks["peaks"][0]
phase = ((flat_time - top["t0"] + 0.5 * top["period"]) % top["period"]) - 0.5 * top["period"]
order = np.argsort(phase)
bin_phase, bin_flux, _ = bin_lc(phase[order] + 0.5 * top["period"], flat_flux[order], 10.0)
fig, axis = plt.subplots()
axis.plot(phase * 24, flat_flux, ",", color="0.6")
axis.plot((bin_phase - 0.5 * top["period"]) * 24, bin_flux, "-", color="tab:red", lw=1.5)
axis.set_xlabel("hours from the signal's epoch")
axis.set_ylabel("detrended flux")
true_period = f"{truth['transit']['period']:.4f} days" if truth["kind"] == "transit" else "none"
axis.set_title(f"Folded at {top['period']:.4f} days; the truth is {true_period}")
plt.show()
""",
            ),
        ],
    ),
]


def main() -> None:
    """Build every notebook, executing unless ``--no-exec`` is given."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-exec", action="store_true", help="write the notebooks without running them")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    for notebook in NOTEBOOKS:
        path = notebook.write(execute=not args.no_exec)
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
