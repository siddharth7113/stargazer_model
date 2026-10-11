import numpy as np

from stargazer import tools
from stargazer.inject import inject, make_truth


def test_summarize_reports_gap_and_scatter(lc):
    time, flux, _ = lc
    s = tools.summarize(time - time.min(), flux)
    assert s["n_points"] == len(time)
    assert abs(s["cadence_minutes"] - 2.0) < 0.01
    assert abs(s["point_scatter_ppm"] / 1000.0 - 1) < 0.05
    assert len(s["gaps_days"]) == 1 and abs(s["gaps_days"][0][1] - 1.0) < 0.01


def test_bls_peaks_top_peak_is_injection(lc, star):
    time, flux, _ = lc
    truth = make_truth(period=3.0, t0=time.min() + 1.0, depth=0.01, b=0.2, star=star)
    f = inject(time, flux, truth)
    r = tools.bls_peaks(time - time.min(), f)
    assert len(r["peaks"]) == 5
    assert abs(r["peaks"][0]["period"] / 3.0 - 1) < 0.01
    assert r["peaks"][0]["sde"] > 7
    ps = [p["period"] for p in r["peaks"]]
    assert all(abs(a / b - 1) >= 0.02 for i, a in enumerate(ps) for b in ps[:i])


def test_fold_shows_dip_at_zero(lc, star):
    time, flux, _ = lc
    truth = make_truth(period=3.0, t0=time.min() + 1.0, depth=0.01, b=0.0, star=star)
    f = inject(time, flux, truth)
    r = tools.fold(time - time.min(), f, period=3.0, t0=1.0, phase_half_width=0.05)
    i = int(np.nanargmin([np.nan if v is None else v for v in r["flux_ppm"]]))
    assert abs(r["phase"][i]) < 0.02
    assert 8000 < r["depth_ppm"] < 14000


def test_odd_even_equal_for_planet_and_unequal_for_binary(lc, star):
    time, flux, _ = lc
    t = time - time.min()
    truth = make_truth(period=3.0, t0=1.0, depth=0.01, b=0.0, star=star)
    f = inject(t, flux, truth)
    r = tools.odd_even(t, f, period=3.0, t0=1.0, duration=truth.duration)
    assert r["odd_n"] > 0 and r["even_n"] > 0 and r["difference_sigma"] < 3
    # a binary: every second event is shallower
    epoch = np.round((t - 1.0) / 3.0).astype(int)
    phase = ((t - 1.0 + 1.5) % 3.0) - 1.5
    shallow = (np.abs(phase) < 0.5 * truth.duration) & (epoch % 2 == 1)
    f2 = f.copy()
    f2[shallow] = flux[shallow] * (1 - 0.003)
    r2 = tools.odd_even(t, f2, period=3.0, t0=1.0, duration=truth.duration)
    assert r2["difference_sigma"] > 3
