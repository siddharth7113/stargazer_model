"""Thin command line layer over the SDK: ``stargazer <command> [options]``.

Each command maps to one or two public API calls. A logbook one-liner is then
reproducible from Python.
"""

from __future__ import annotations

import argparse
import json
import sys
import time as _time
from pathlib import Path


def cmd_targets(a: argparse.Namespace) -> None:
    """Select host stars for a sector and save the targets file."""
    from stargazer.targets import save_targets, select_targets

    d = select_targets(a.sector, a.n, a.seed, tmag_max=a.tmag_max)
    save_targets(d, a.out)
    print(f"selected {d['n_selected']} / {d['n_requested']} targets -> {a.out}")
    print(f"sector 2-min stars: {d['n_two_minute_in_sector']}, excluded by TOI: {d['n_excluded_by_toi']}")
    print(f"rejected by cuts: {d['rejected']}")


def cmd_fetch(a: argparse.Namespace) -> None:
    """Download light curves for each target. Record failures."""
    from stargazer.data import LightCurveCache
    from stargazer.provenance import Manifest
    from stargazer.targets import load_targets

    cache = LightCurveCache(a.cache)
    ok, failed = 0, []
    for t in load_targets(a.targets)["targets"]:
        try:
            cache.fetch(t["tic"], t["sector"])
            ok += 1
            print(f"cached TIC {t['tic']} s{t['sector']}", flush=True)
        except Exception as e:
            failed.append({"tic": t["tic"], "error": str(e)[:200]})
            print(f"FAILED TIC {t['tic']}: {e}", file=sys.stderr, flush=True)
    m = Manifest.of_directory(cache.root, pattern="*.npz").write(cache.root)
    print(f"cached {ok}, failed {len(failed)}, manifest {m}")
    if failed:
        (cache.root / "fetch_failures.json").write_text(json.dumps(failed, indent=2))


def cmd_build(a: argparse.Namespace) -> None:
    """Build an episode set."""
    from stargazer.data import LightCurveCache
    from stargazer.episodes import BuildConfig, EpisodeSet
    from stargazer.targets import load_targets

    cfg = BuildConfig(
        n=a.n,
        seed=a.seed,
        null_frac=a.null_frac,
        period_range=(a.min_period, a.max_period),
        depth_range=(a.min_depth, a.max_depth),
    )
    es = EpisodeSet.build(
        load_targets(a.targets), LightCurveCache(a.cache), Path(a.out), cfg, Path(a.targets)
    )
    print(f"built {len(es)} episodes -> {es.root}\nepisode set hash {es.hash}")


def _progress(k, p, tr):
    print(
        f"{k} {tr['kind']:8s} det={p.detected!s:5s} P={p.period:.4f} "
        f"sde={p.extra.get('sde', float('nan')):.1f} ({p.extra.get('seconds')}s)",
        flush=True,
    )


def cmd_baseline(a: argparse.Namespace) -> None:
    """Run the BLS baseline on an episode set and write results."""
    from stargazer.baseline import BLSDetector
    from stargazer.episodes import EpisodeSet
    from stargazer.evaluate import evaluate

    es = EpisodeSet(a.episodes)
    det = BLSDetector(**(json.loads(a.config) if a.config else {}))
    ids = es.ids[: a.limit] if a.limit else None
    res = evaluate(det, es, ids=ids, period_rtol=a.period_rtol, progress=_progress)
    out = a.out or Path("results") / f"{det.name}_{es.root.name}_{_time.strftime('%Y%m%dT%H%M%S')}.json"
    res.save(out)
    print(json.dumps(res.summary, indent=1))
    print(f"results -> {out}")


def cmd_score(a: argparse.Namespace) -> None:
    """Score an external predictions JSON against an episode set."""
    from stargazer.episodes import EpisodeSet
    from stargazer.evaluate import Results
    from stargazer.score import score_set

    es = EpisodeSet(a.episodes)
    preds = json.loads(Path(a.preds).read_text())
    rows, summary = score_set(preds, es.truth, period_rtol=a.period_rtol)
    res = Results(a.system, {"period_rtol": a.period_rtol}, str(es.root), es.hash, preds, rows, summary)
    out = a.out or Path("results") / f"{a.system}_{es.root.name}_{_time.strftime('%Y%m%dT%H%M%S')}.json"
    res.save(out)
    print(json.dumps(summary, indent=1))
    print(f"results -> {out}")


def cmd_calibrate(a: argparse.Namespace) -> None:
    """Sweep the detection threshold on a results file and print the table."""
    from stargazer.calibrate import choose_threshold, threshold_sweep
    from stargazer.episodes import EpisodeSet
    from stargazer.evaluate import Results

    res = Results.load(a.results)
    truth = EpisodeSet(res.episode_set).truth
    thresholds = [float(x) for x in a.thresholds.split(",")]
    rows = threshold_sweep(res, truth, thresholds, stat=a.stat)
    print(f"{'threshold':>9} {'false_alarm':>11} {'compl_exact':>11} {'compl_alias':>11} {'mean_score':>10}")
    for r in rows:
        print(
            f"{r['threshold']:9.2f} {r['false_alarm_rate']:11.3f} {r['completeness_exact']:11.3f} "
            f"{r['completeness_with_aliases']:11.3f} {r['mean_score']:10.3f}"
        )
    chosen = choose_threshold(res, truth, max_false_alarm=a.max_false_alarm, stat=a.stat)
    print(f"lowest threshold with false-alarm rate <= {a.max_false_alarm}: {chosen:.3f}")
    if a.out:
        Path(a.out).write_text(
            json.dumps(
                {
                    "results": a.results,
                    "episode_set_hash": res.episode_set_hash,
                    "stat": a.stat,
                    "sweep": rows,
                    "max_false_alarm": a.max_false_alarm,
                    "chosen_threshold": chosen,
                },
                indent=1,
            )
        )
        print(f"sweep -> {a.out}")


def cmd_verify(a: argparse.Namespace) -> None:
    """Recompute manifests. Exit with code 1 if a mismatch exists."""
    from stargazer.provenance import Manifest

    problems = []
    for d in a.dirs:
        p = Manifest.read(Path(d)).verify(Path(d))
        print(f"{d}: {'OK' if not p else f'{len(p)} problems'}")
        problems += p
    for p in problems:
        print("  " + p)
    sys.exit(1 if problems else 0)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    ap = argparse.ArgumentParser(prog="stargazer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("targets", help="select host stars with no TOI for a sector")
    p.add_argument("--sector", type=int, required=True)
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--tmag-max", type=float, default=12.0)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_targets)

    p = sub.add_parser("fetch", help="download and cache light curves for a targets file")
    p.add_argument("--targets", required=True)
    p.add_argument("--cache", default="data/cache")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("build", help="build an injection-recovery episode set")
    p.add_argument("--targets", required=True)
    p.add_argument("--cache", default="data/cache")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--null-frac", type=float, default=0.3)
    p.add_argument("--min-period", type=float, default=0.5)
    p.add_argument("--max-period", type=float, default=20.0)
    p.add_argument("--min-depth", type=float, default=1e-4)
    p.add_argument("--max-depth", type=float, default=1e-2)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("baseline", help="run the BLS baseline on an episode set and score it")
    p.add_argument("--episodes", required=True)
    p.add_argument("--config", default=None, help="JSON overrides for BLSDetector")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--period-rtol", type=float, default=0.01)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("score", help="score a predictions JSON against an episode set")
    p.add_argument("--episodes", required=True)
    p.add_argument("--preds", required=True)
    p.add_argument("--system", required=True)
    p.add_argument("--period-rtol", type=float, default=0.01)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("calibrate", help="sweep the detection threshold on a results file")
    p.add_argument("--results", required=True)
    p.add_argument("--stat", default="sde")
    p.add_argument("--thresholds", default="5,6,7,8,9,10,12,15")
    p.add_argument("--max-false-alarm", type=float, default=0.05)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("verify", help="recompute SHA-256 manifests")
    p.add_argument("dirs", nargs="+")
    p.set_defaults(func=cmd_verify)
    return ap


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    a = build_parser().parse_args(argv)
    if a.cmd == "targets" and a.out is None:
        a.out = f"data/targets_s{a.sector}.json"
    a.func(a)


if __name__ == "__main__":
    main()
