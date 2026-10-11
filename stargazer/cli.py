"""Command line entry point: stargazer <command> [options]."""

from __future__ import annotations

import argparse
import json
import sys
import time as _time
from pathlib import Path


def cmd_targets(a):
    from .targets import save_targets, select_targets

    d = select_targets(a.sector, a.n, a.seed, tmag_max=a.tmag_max)
    save_targets(d, a.out)
    print(f"selected {d['n_selected']} / {d['n_requested']} targets -> {a.out}")
    print(f"sector 2-min stars: {d['n_two_minute_in_sector']}, excluded by TOI: {d['n_excluded_by_toi']}")
    print(f"rejected by cuts: {d['rejected']}")


def cmd_fetch(a):
    from .data import fetch_lightcurve
    from .provenance import write_manifest
    from .targets import load_targets

    d = load_targets(a.targets)
    ok, failed = 0, []
    for t in d["targets"]:
        try:
            fetch_lightcurve(t["tic"], t["sector"], a.cache)
            ok += 1
            print(f"cached TIC {t['tic']} s{t['sector']}", flush=True)
        except Exception as e:  # keep going; record failures
            failed.append({"tic": t["tic"], "error": str(e)[:200]})
            print(f"FAILED TIC {t['tic']}: {e}", file=sys.stderr, flush=True)
    m = write_manifest(Path(a.cache), pattern="*.npz")
    print(f"cached {ok}, failed {len(failed)}, manifest {m}")
    if failed:
        Path(a.cache, "fetch_failures.json").write_text(json.dumps(failed, indent=2))


def cmd_build(a):
    from .episodes import build_episode_set, episode_set_hash
    from .targets import load_targets

    d = load_targets(a.targets)
    out = build_episode_set(d, Path(a.cache), Path(a.out), a.n, a.seed, null_frac=a.null_frac,
                            period_range=(a.min_period, a.max_period),
                            depth_range=(a.min_depth, a.max_depth), targets_path=Path(a.targets))
    print(f"built {a.n} episodes -> {out}\nepisode set hash {episode_set_hash(out)}")


def cmd_baseline(a):
    from .baseline import DEFAULT_CONFIG, detect
    from .episodes import episode_set_hash, list_episodes, load_episode, load_truth
    from .score import score_set, write_results

    cfg = {**DEFAULT_CONFIG, **(json.loads(a.config) if a.config else {})}
    truth = load_truth(a.episodes)
    ids = list_episodes(a.episodes)
    if a.limit:
        ids = ids[: a.limit]
    preds = {}
    t_start = _time.time()
    for k in ids:
        t, f, fe = load_episode(a.episodes, k)
        t1 = _time.time()
        preds[k] = detect(t, f, fe, cfg)
        preds[k]["seconds"] = round(_time.time() - t1, 2)
        print(f"{k} {truth[k]['kind']:8s} det={preds[k]['detected']!s:5s} "
              f"P={preds[k]['period']:.4f} sde={preds[k]['sde']:.1f} ({preds[k]['seconds']}s)", flush=True)
    truth_sub = {k: truth[k] for k in ids}
    rows, summary = score_set(preds, truth_sub, period_rtol=a.period_rtol)
    cfg["period_rtol"] = a.period_rtol
    cfg["wall_seconds"] = round(_time.time() - t_start, 1)
    out = a.out or Path("results") / f"baseline_bls_{Path(a.episodes).name}_{_time.strftime('%Y%m%dT%H%M%S')}.json"
    write_results(out, system="baseline_bls", config=cfg, episode_set=a.episodes,
                  episode_set_hash=episode_set_hash(a.episodes), preds=preds, rows=rows, summary=summary)
    print(json.dumps(summary, indent=1))
    print(f"results -> {out}")


def cmd_score(a):
    from .episodes import episode_set_hash, load_truth
    from .score import score_set, write_results

    preds = json.loads(Path(a.preds).read_text())
    truth = load_truth(a.episodes)
    rows, summary = score_set(preds, truth, period_rtol=a.period_rtol)
    out = a.out or Path("results") / f"{a.system}_{Path(a.episodes).name}_{_time.strftime('%Y%m%dT%H%M%S')}.json"
    write_results(out, system=a.system, config={"period_rtol": a.period_rtol}, episode_set=a.episodes,
                  episode_set_hash=episode_set_hash(a.episodes), preds=preds, rows=rows, summary=summary)
    print(json.dumps(summary, indent=1))
    print(f"results -> {out}")


def cmd_verify(a):
    from .provenance import verify_manifest

    problems = []
    for d in a.dirs:
        p = verify_manifest(Path(d))
        print(f"{d}: {'OK' if not p else len(p)} {'' if not p else 'problems'}")
        problems += p
    for p in problems:
        print("  " + p)
    sys.exit(1 if problems else 0)


def main(argv=None):
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
    p.add_argument("--min-depth", type=float, default=2e-4)
    p.add_argument("--max-depth", type=float, default=2e-2)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("baseline", help="run the BLS baseline on an episode set and score it")
    p.add_argument("--episodes", required=True)
    p.add_argument("--config", default=None, help="JSON overrides for baseline config")
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

    p = sub.add_parser("verify", help="recompute SHA-256 manifests")
    p.add_argument("dirs", nargs="+")
    p.set_defaults(func=cmd_verify)

    a = ap.parse_args(argv)
    if a.cmd == "targets" and a.out is None:
        a.out = f"data/targets_s{a.sector}.json"
    a.func(a)


if __name__ == "__main__":
    main()
