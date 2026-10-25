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

- The Box Least Squares (BLS) search folds the light curve at each trial period and
  asks how well a box-shaped dip fits. The tool `bls_peaks` returns the strongest
  distinct peaks.
- The tool `search_signals` runs the search, masks the strongest signal, and searches
  again until the rest is noise.

The output of this notebook is a short list of candidates: a period, an epoch, a
duration and a depth for each, with a strength. Notebook 4 tests them.
""",
            ),
            (
                "markdown",
                """
## Detrending first

Both tools detrend the light curve before the search with a **Savitzky-Golay** filter.
The filter slides a window along the light curve. In each window it fits a polynomial
of order 2 to the flux. The value of the fit at the window centre is the trend at that
point. The detrended flux is the raw flux divided by the trend. Points more than three
standard deviations from the trend are clipped and the fit is repeated, so a deep
transit does not pull the trend down. The light curve is split at gaps longer than
five cadences, so a gap does not distort the fit on either side.

The window is 0.75 days here. The window must be several times the transit duration or
the filter would erase the dip. A window of 0.75 days keeps transits up to about 6
hours. The plot shows the raw and the detrended flux.
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
## How the search works

A fold takes a trial period and places every point at its phase in that orbit: the
time since the last trial epoch, instead of the time since the start. If the trial
period is the true one, every transit lands at the same phase and the faint dips stack
into one clear dip. If the trial period is wrong, the dips land at different phases
and smear into the noise.

On each folded curve the search fits a box: flux at one level everywhere, and a lower
level inside a window of some width and position. It slides the window across the
phase, tries several widths, and keeps the box that lowers the squared residuals the
most. That best box gives the epoch, the duration and the depth. Its improvement in
fit is the power at that trial period. The search repeats the fold and the box fit for
thousands of trial periods. The power against period is the periodogram, and its
highest peak is the candidate.

![A wrong fold, the right fold with its box, and the periodogram](figures/bls_fold.png)

The figure uses a synthetic light curve with one injected transit every 3.2 days.
The smaller peaks at half, double and triple the period are aliases: a fold at a
multiple of the true period still puts some dips at phase 0, so a box fits there too,
with less power.

## `bls_peaks`

Each peak has:
- `period`: the trial period with the most box-fit power, in days.
- `t0`: the epoch, the time of the first transit centre, in days from the start of the
  light curve.
- `duration`: the width of the box, in days. A duration of 0.051 days is 1.2 hours.
- `depth_ppm`: the depth of the box in parts per million. 1000 ppm is 0.1 percent of
  the star's light.
- `sde`: the signal detection efficiency. It says how far the peak stands above the
  mean of the periodogram, in standard deviations of the periodogram.
- `n_events_in_data`: the number of transit epochs with at least three points in
  transit. An epoch that falls in a data gap does not count.

On real stars noise alone reaches an SDE of 10 to 14, so a peak is a hypothesis to
check. The tool also returns the half and double period of each peak, because a search
often lands on an alias of the true period.

Read the output below as one signal, not five. The first row is the planet: a period
of 0.776 days, SDE 23.6, 48 events. The next three rows are at two, three and four
times that period. A fold at a multiple of the true period still puts a dip at phase
0, so the box fits there too, but with fewer events and a lower SDE. The last row is
at two thirds of the period, another alias. Its depth is lower because that fold
spreads the dips over two phases and the box catches only part of them.
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
    Notebook(
        "04_vetting_a_candidate",
        [
            (
                "markdown",
                """
# 4. Vetting a candidate

A peak from the period search is a candidate, not a planet. Four tools test it:
- `fold` shows the shape of the dip,
- `fit_transit` fits a physical model and reports the geometry,
- `odd_even` compares alternate transits, and
- `secondary_eclipse` looks for a second dip half an orbit later.

Each tool returns a verdict word with the numbers behind it. This notebook runs the
four on the strongest signal of the loaded episode.

The tools look for the usual impostors. The most common one is an eclipsing binary:
two stars that orbit each other and take turns to block each other's light. Its dips
can be as shallow as a planet's when the two stars are blended with a third, or when
one star only grazes the edge of the other. Each test below catches one way a binary
gives itself away.
""",
            ),
            (
                "code",
                """
from stargazer.tools import search_signals

candidate = search_signals(time, flux)["signals"][0]
print({key: candidate[key] for key in ("period", "t0", "duration", "depth_ppm", "sde")})
""",
            ),
            (
                "markdown",
                """
## `fold`

The fold stacks every orbit on top of each other, as notebook 3 explains, and
returns the stacked curve in 40 bins. The output is short on purpose: the model
re-reads it on every later turn.

Each bin has:
- `phase`: the time from the transit centre, in days. Phase 0 is the centre of the dip.
- `flux_ppm`: the binned flux minus one, in parts per million. A flat star reads 0
  and a 1000 ppm transit reads about -1000 at phase 0.
- `depth_ppm`: the deepest bin.

The argument `phase_half_width` sets how much of the orbit to return, as a fraction
of the period. The two plots use the two useful settings:
- **0.5 shows the whole orbit**, from half a period before the dip to half a period
  after. Use it to look for a second dip. A second dip half way round means the
  candidate is a binary, or the true period is twice the trial one and the search
  folded two different eclipses onto each other.
- **0.05 zooms on the transit**, one twentieth of the orbit on each side. Use it to
  see the shape of the dip: flat bottom or V, and how wide it is.
""",
            ),
            (
                "code",
                """
from stargazer.tools import fold

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
for axis, width in zip(axes, (0.5, 0.05), strict=True):
    curve = fold(time, flux, period=candidate["period"], t0=candidate["t0"], phase_half_width=width)
    phases = np.array(curve["phase"], dtype=float)
    values = np.array([np.nan if value is None else value for value in curve["flux_ppm"]], dtype=float)
    axis.step(phases * 24, values, where="mid", color="tab:red")
    axis.set_xlabel("hours from the epoch")
    axis.set_ylabel("flux change (ppm)")
    axis.set_title(f"phase_half_width {width}: depth {curve['depth_ppm']:.0f} ppm")
plt.show()
""",
            ),
            (
                "markdown",
                r"""
## `fit_transit`

The box in notebook 3 was a crude shape. This tool fits a real transit curve, the
batman model, to the folded data. The curve has a round bottom from limb darkening,
the dimming of a star's disc towards its edge. Four numbers are free in the fit:
- the epoch offset, a small shift of the transit centre,
- `rp_rs`, the radius of the planet divided by the radius of the star,
- `a_rs`, the radius of the orbit divided by the radius of the star,
- `impact_parameter`, how far from the centre of the star the planet crosses. 0 is
  straight across the middle. 1 is along the edge.

The fit reports the depth as the square of `rp_rs`, the duration in hours and a
refined epoch. The two verdict words come from the geometry.

**The density verdict.** Kepler's third law links the period and the size of the
orbit to the mass of the star. With the orbit size in units of the star's radius, the
mass turns into a density:

$$\rho_\star = \frac{3 \pi}{G P^2} \left( \frac{a}{R_\star} \right)^3$$

So the fit says how dense the host star must be for a planet to make this dip at this
period. A normal star has a density between 0.1 and 20 grams per cubic centimetre,
and the verdict is `plausible` inside that range. A value outside it means the dip is
not a planet transit at this period. A blended binary or a wrong period both give an
impossible density.

**The shape verdict.** The sum of the impact parameter and the radius ratio says
whether the planet disc fits inside the star's disc at mid-transit. The shape is
`flat_bottom` when the sum is below 0.9: the whole planet is in front of the star for
a while, so the dip has a floor. The shape is `v_shaped` when the sum reaches 1: the
object only grazes the edge, so the dip is a V with no floor. Two grazing stars make a
V far more often than a planet does. The shape is `grazing_uncertain` between the two,
where the fit of a short transit cannot tell the two apart.

`significance_sigma` says how much better the transit curve fits than a flat line.
`n_bins` and `n_in_transit` are the number of bins the fit used and the number inside
the transit. The plot shows the binned fold with the fitted curve.
""",
            ),
            (
                "code",
                """
from stargazer.baseline import bin_lc, detrend
from stargazer.inject import TransitTruth, limb_darkening_for_teff, transit_model
from stargazer.tools import FIT_LIMB_DARKENING_TEFF, fit_transit

fit = fit_transit(time, flux, period=candidate["period"], t0=candidate["t0"], duration=candidate["duration"])
for key, value in fit.items():
    print(f"{key:22s} {value}")

flat_time, flat_flux = detrend(time, flux, 0.75)
period = candidate["period"]
phase = ((flat_time - fit["t0"] + 0.5 * period) % period) - 0.5 * period
window = np.abs(phase) < 2.5 * candidate["duration"]
bin_phase, bin_flux, _ = bin_lc(phase[window] + 2.5 * candidate["duration"], flat_flux[window], 5.0)
bin_phase = bin_phase - 2.5 * candidate["duration"]
u1, u2 = limb_darkening_for_teff(FIT_LIMB_DARKENING_TEFF)
inclination = float(np.degrees(np.arccos(min(fit["impact_parameter"], 0.999 * fit["a_rs"]) / fit["a_rs"])))
geometry = (fit["rp_rs"], fit["a_rs"], inclination, fit["impact_parameter"])
model = TransitTruth(period, 0.0, *geometry, u1, u2, 0.0, fit["rp_rs"] ** 2)
grid = np.linspace(bin_phase.min(), bin_phase.max(), 500)
fig, axis = plt.subplots()
axis.plot(bin_phase * 24, bin_flux, ".", color="0.3", label="5-minute bins")
axis.plot(grid * 24, transit_model(grid, model), color="tab:red", label="fitted model")
axis.set_xlabel("hours from the fitted epoch")
axis.set_ylabel("detrended flux")
axis.set_title(f"{fit['shape']}, density {fit['density_verdict']}, {fit['significance_sigma']} sigma")
axis.legend()
plt.show()
""",
            ),
            (
                "markdown",
                """
## `odd_even`

Two stars in a binary are not the same size and brightness, so the two eclipses in
one orbit have different depths. If the search lands on half the true period, it
folds the deep and the shallow eclipse onto each other and reports one dip. The
alternate transits then differ in depth: the first, third and fifth are the deep
eclipse and the second, fourth and sixth are the shallow one. A planet gives the
same dip every time.

The tool numbers the events in order and measures the depth of each. It averages the
odd-numbered and the even-numbered events, then compares the two averages:
- `odd_depth_ppm` and `even_depth_ppm`: the two average depths.
- `odd_events` and `even_events`: how many events went into each average.
- `difference_sigma`: the difference in the two depths, divided by its uncertainty.
  The uncertainty comes from how much the single events differ from each other, not
  from the point noise, so that leftover stellar variability counts.
- `verdict`: `consistent` below 3 sigma, `inconsistent` at or above it, and
  `insufficient_data` when either parity has fewer than two events.
""",
            ),
            (
                "code",
                """
from stargazer.tools import odd_even

parity = odd_even(time, flux, period=candidate["period"], t0=candidate["t0"], duration=candidate["duration"])
for key, value in parity.items():
    print(f"{key:18s} {value}")
""",
            ),
            (
                "markdown",
                """
## `secondary_eclipse`

In a binary the smaller star shines too, so half an orbit after it passes in front
of the larger star it passes behind it. That makes a second, shallower dip at phase
0.5. A planet reflects too little light for its own secondary to show in one TESS
sector, so a second dip at phase 0.5 marks a binary.

The tool folds the detrended flux and measures the depth of a box, one transit
duration wide, at 200 phases around the orbit. The spread of those 200 depths is the
noise. Phases within three durations of the primary are left out of the noise and of
the search, because the detrending leaves a residual around a deep dip.
- `primary_depth_ppm`: the depth at phase 0, for comparison.
- `secondary_depth_ppm` and `secondary_sigma`: the depth at phase 0.5 and that depth
  divided by the noise.
- `strongest_other_phase`, `strongest_other_depth_ppm` and `strongest_other_sigma`:
  the deepest box at any phase away from the primary. A strong dip at a phase other
  than 0.5 means the period is wrong or a second signal is present.
- `noise_ppm`: the noise, from the spread of the trial depths.
- `verdict`: `secondary_at_half_phase` if the phase-0.5 depth reaches 3 sigma,
  `dip_at_other_phase` if the strongest other dip reaches 5 sigma, else
  `no_secondary`.

The plot shows the 200 trial depths across the orbit. The line at phase 0.5 marks
where a binary's second eclipse would be.
""",
            ),
            (
                "code",
                """
from stargazer.tools import secondary_eclipse

second = secondary_eclipse(
    time, flux, period=candidate["period"], t0=candidate["t0"], duration=candidate["duration"]
)
for key, value in second.items():
    print(f"{key:26s} {value}")

phase_orbit = ((flat_time - candidate["t0"]) / period) % 1.0
half_width = 0.5 * candidate["duration"] / period
trials = np.arange(200) / 200
depths = []
for trial in trials:
    offset = np.abs(((phase_orbit - trial + 0.5) % 1.0) - 0.5)
    points = flat_flux[offset < half_width]
    depths.append((1.0 - points.mean()) * 1e6 if len(points) >= 3 else np.nan)
fig, axis = plt.subplots()
axis.plot(trials, depths, color="0.3")
axis.axvline(0.5, color="tab:red", alpha=0.5, label="phase 0.5")
axis.set_xlabel("orbital phase")
axis.set_ylabel("box depth (ppm)")
axis.set_title(f"verdict {second['verdict']}")
axis.legend()
plt.show()
""",
            ),
        ],
    ),
    Notebook(
        "05_single_events",
        [
            (
                "markdown",
                """
# 5. Single events

A planet with a period longer than half the data makes one transit, or none. A
period search cannot find it. The tool `single_events` scores a box of each width
at each time and returns the strongest dips that do not repeat. This notebook
injects one transit into the loaded episode and shows the statistic.
""",
            ),
            (
                "markdown",
                """
## One injected transit

The injection has a 45-day period and an epoch at day 25, so one transit lies in
the data, away from the gaps. The plot marks it. The host already has the
periodic signal of the episode, so the single transit must stand out beside it.
""",
            ),
            (
                "code",
                """
from stargazer.inject import inject, make_truth

star = {"teff": 5500.0, "logg": 4.45, "radius_rsun": 0.95, "tmag": 10.0, "mass_msun": 0.95}
lone = make_truth(period=45.0, t0=25.0, depth=0.003, b=0.2, star=star)
with_event = inject(time, flux, lone)
fig, axis = plt.subplots()
axis.plot(time, with_event, ",", color="0.3")
axis.axvline(25.0, color="tab:red", alpha=0.5)
axis.set_xlabel("time (days)")
axis.set_ylabel("relative flux")
axis.set_title(f"One {lone.depth * 1e6:.0f} ppm transit of {lone.duration * 24:.1f} hours at day 25")
plt.show()
""",
            ),
            (
                "markdown",
                """
## The single-event statistic

For a box of width $d$ the statistic is

$$\\mathrm{snr} = \\frac{\\bar{f}_\\mathrm{median} - \\bar{f}_\\mathrm{box}}{\\sigma_d}$$

where $\\bar{f}_\\mathrm{box}$ is the mean flux in the box and $\\sigma_d$ is the noise
at the box timescale, 1.4826 times the median absolute deviation of the box means
over the whole light curve. On white noise it equals the point scatter over the
square root of the points in the box. On an active star it is larger, so slow
variability does not produce an event with a high statistic. Each box width is
detrended with a window of at least six widths, so a short window cannot erase a
long event. A box near the start or the end of a data segment is skipped, because
the detrending filter leaves a dip-shaped residual there.
""",
            ),
            (
                "code",
                """
from stargazer.tools import single_events

events = single_events(time, with_event)
for event in events["events"]:
    print(event)
""",
            ),
            (
                "markdown",
                """
## The strongest event

The plot zooms on the strongest event in the detrended flux, with the window the
tool used for that box width. A transit-like event has a duration of hours, a flat
curve on each side, and no partner of the same depth elsewhere. A lone event that
passes those checks is submitted as a single event: a time, a duration and a depth,
and no period.
""",
            ),
            (
                "code",
                """
from stargazer.baseline import detrend

top = events["events"][0]
flat_time, flat_flux = detrend(time, with_event, max(0.75, 6.0 * top["duration"]))
near = np.abs(flat_time - top["time"]) < 0.5
fig, axis = plt.subplots()
axis.plot((flat_time[near] - top["time"]) * 24, flat_flux[near], ".", ms=3, color="0.3")
axis.axvspan(-top["duration"] * 12, top["duration"] * 12, color="tab:red", alpha=0.15, label="box")
axis.set_xlabel(f"hours from day {top['time']:.2f}")
axis.set_ylabel("detrended flux")
axis.set_title(f"snr {top['snr']}, depth {top['depth_ppm']:.0f} ppm, noise {top['noise_ppm']:.0f} ppm")
axis.legend()
plt.show()
""",
            ),
        ],
    ),
    Notebook(
        "06_scoring",
        [
            (
                "markdown",
                """
# 6. Scoring

The scorer compares one prediction with the hidden truth of one episode and returns
a number between 0 and 1 with an outcome word. It reads no files and uses no
network, so a training loop can call it on each rollout. This notebook runs it on
the loaded episode with several predictions and shows what each rule does.
""",
            ),
            (
                "markdown",
                """
## The periodic form

A prediction with a period is credited 1.0 at the true period, 0.5 at half or
double, 0.25 at a third or triple, within 1 percent. The epoch is folded on the true
period and must fall within half a duration of a transit centre; it is worth 30
percent of the credit. A wrong period scores zero, the same as a false alarm.
""",
            ),
            (
                "code",
                """
from stargazer.score import score_episode

transit = truth["transit"]
predictions = {
    "right period and epoch": {"detected": True, "period": transit["period"], "t0": transit["t0"]},
    "right period, wrong epoch": {
        "detected": True,
        "period": transit["period"],
        "t0": transit["t0"] + 0.4 * transit["period"],
    },
    "double period": {"detected": True, "period": 2 * transit["period"], "t0": transit["t0"]},
    "wrong period": {"detected": True, "period": 1.37 * transit["period"], "t0": transit["t0"]},
    "no detection": {"detected": False},
}
for label, prediction in predictions.items():
    row = score_episode(prediction, truth)
    print(f"{label:28s} outcome {row['outcome']:13s} score {row['score']:.2f}")
""",
            ),
            (
                "markdown",
                """
## The single-event form

A prediction with no period claims one event at a time. The event passes the same
epoch check. It scores 1.0 on an episode with one transit in the data, where no
period was measurable, and 0.5 where the data held more transits. The loaded
episode has many transits, so an event on one of them scores 0.5.
""",
            ),
            (
                "code",
                """
one_transit = {**truth, "n_transits": 1}
on_transit = {
    "detected": True,
    "t0": transit["t0"],
    "duration": transit["duration"],
    "depth": transit["depth"],
}
off_transit = {**on_transit, "t0": transit["t0"] + 0.5 * transit["period"]}
for label, prediction, record in (
    ("event on a transit, many in the data", on_transit, truth),
    ("event on a transit, one in the data", on_transit, one_transit),
    ("event between transits", off_transit, truth),
):
    row = score_episode(prediction, record)
    print(f"{label:38s} outcome {row['outcome']:13s} score {row['score']:.2f}")
""",
            ),
            (
                "markdown",
                """
## A null episode

On an episode with no injection, "none" scores 1.0 and any detection scores 0.0.
""",
            ),
            (
                "code",
                """
null = {"kind": "null"}
for label, prediction in (("no detection", {"detected": False}), ("a detection", on_transit)):
    row = score_episode(prediction, null)
    print(f"{label:14s} outcome {row['outcome']:12s} score {row['score']:.2f}")
""",
            ),
            (
                "markdown",
                """
## The score against the period error

The plot sweeps the predicted period across the true one and shows the score. The
tolerance is a strict band of 1 percent; the aliases at half and double are narrower
bands at half credit.
""",
            ),
            (
                "code",
                """
ratios = np.linspace(0.45, 2.1, 2000)
scores = [
    score_episode({"detected": True, "period": ratio * transit["period"], "t0": transit["t0"]}, truth)[
        "score"
    ]
    for ratio in ratios
]
fig, axis = plt.subplots()
axis.plot(ratios, scores, color="tab:red")
axis.set_xlabel("predicted period / true period")
axis.set_ylabel("score")
plt.show()
""",
            ),
        ],
    ),
    Notebook(
        "07_reading_a_run",
        [
            (
                "markdown",
                """
# 7. Reading a run

A run writes one results file: the configuration that produced it, each
prediction with its full trajectory, each score row, and the set-level summary.
This notebook reads the most recent full run of Inkling-Small on s98_v2 if it is
present, and the three-episode check otherwise, and shows how to read it.
""",
            ),
            (
                "code",
                """
import json

candidates = [
    Path("../results/full_v5fit/agent_inkling_small_s98_v2.json"),
    Path("../results/full_v4search/agent_inkling_small_s98_v2.json"),
    Path("../results/smoke/agent_inkling_small_s98_v2_fit3.json"),
]
results_path = next(path for path in candidates if path.exists())
results = json.loads(results_path.read_text())
print(results_path.name, "·", results["system"], "·", len(results["episodes"]), "episodes")
settings = ("backend_base_model", "backend_renderer", "period_rtol", "single_event_credit")
print({key: results["config"][key] for key in settings})
""",
            ),
            (
                "markdown",
                """
## The summary

The two headline numbers are the false-alarm rate, the share of null episodes with
a detection, and the completeness, the share of injected transits found. The
completeness by transit count shows the regime the single-event form opened.
""",
            ),
            (
                "code",
                """
summary = results["summary"]
print("false-alarm rate", round(summary["false_alarm_rate"], 3))
exact, with_aliases = summary["completeness_exact"], summary["completeness_with_aliases"]
print("completeness exact", round(exact, 3), "with aliases", round(with_aliases, 3))
print("mean score", round(summary["mean_score"], 3))
fig, axis = plt.subplots(figsize=(6, 3.5))
labels = list(summary["by_n_transits"])
values = [bin_["completeness"] or 0.0 for bin_ in summary["by_n_transits"].values()]
counts = [bin_["n"] for bin_ in summary["by_n_transits"].values()]
axis.bar(labels, values, color="tab:red")
for index, (value, count) in enumerate(zip(values, counts, strict=True)):
    axis.text(index, value + 0.01, f"n={count}", ha="center")
axis.set_xlabel("transits in the data")
axis.set_ylabel("completeness")
plt.show()
""",
            ),
            (
                "markdown",
                """
## The outcomes

Each episode has one outcome word. A hit is the right period; an alias is half or
double; a wrong period is a detection at an unrelated period; an event hit is a
credited single event; a miss is an injection the model declined; a true null and a
false alarm are the two answers on an empty star.
""",
            ),
            (
                "code",
                """
from collections import Counter

outcomes = Counter(record["score"]["outcome"] for record in results["episodes"].values())
for outcome, count in outcomes.most_common():
    print(f"{outcome:14s} {count}")
""",
            ),
            (
                "markdown",
                """
## One trajectory

The prediction of each episode holds the transcript: each tool call with its
arguments and its result. The cell prints the calls of the first hit, so the reader
sees the procedure the model followed. The `stargazer show` command prints the same
with the tool results summarised.
""",
            ),
            (
                "code",
                """
hits = [episode for episode, record in results["episodes"].items() if record["score"]["outcome"] == "hit"]
first_hit = hits[0]
record = results["episodes"][first_hit]
print(first_hit, "·", record["score"]["outcome"], "· score", record["score"]["score"])
for step, call in enumerate(record["pred"]["transcript"], start=1):
    arguments = {key: value for key, value in call["arguments"].items() if key != "reasoning"}
    print(f"{step:2d} {call['tool']:18s} {arguments}")
print("reasoning:", record["pred"].get("reasoning"))
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
