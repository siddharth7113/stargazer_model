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


def cmd_targets(args: argparse.Namespace) -> None:
    """Select host stars for a sector and save the targets file."""
    from stargazer.targets import save_targets, select_targets

    selection = select_targets(args.sector, args.n, args.seed, tmag_max=args.tmag_max)
    save_targets(selection, args.out)
    print(f"selected {selection['n_selected']} / {selection['n_requested']} targets -> {args.out}")
    print(f"sector 2-min stars: {selection['n_two_minute_in_sector']}")
    print(f"excluded by TOI: {selection['n_excluded_by_toi']}")
    print(f"rejected by cuts: {selection['rejected']}")


def cmd_remove_binaries(args: argparse.Namespace) -> None:
    """Remove the known eclipsing binaries from a targets file and save the result."""
    from stargazer.targets import load_targets, remove_binaries, save_targets

    selection = remove_binaries(load_targets(args.targets))
    save_targets(selection, args.out)
    removed = ", ".join(
        f"{entry['tic']} ({' '.join(entry['evidence'])})" for entry in selection["removed_binaries"]
    )
    print(f"kept {selection['n_selected']} targets -> {args.out}")
    print(f"removed {len(selection['removed_binaries'])}: {removed}")


def cmd_fetch(args: argparse.Namespace) -> None:
    """Download light curves for each target. Record failures."""
    from stargazer.data import LightCurveCache
    from stargazer.provenance import Manifest
    from stargazer.targets import load_targets

    cache = LightCurveCache(args.cache)
    ok, failed = 0, []
    for target in load_targets(args.targets)["targets"]:
        try:
            cache.fetch(target["tic"], target["sector"])
            ok += 1
            print(f"cached TIC {target['tic']} s{target['sector']}", flush=True)
        except Exception as e:
            failed.append({"tic": target["tic"], "error": str(e)[:200]})
            print(f"FAILED TIC {target['tic']}: {e}", file=sys.stderr, flush=True)
    manifest_path = Manifest.of_directory(cache.root, pattern="*.npz").write(cache.root)
    print(f"cached {ok}, failed {len(failed)}, manifest {manifest_path}")
    if failed:
        (cache.root / "fetch_failures.json").write_text(json.dumps(failed, indent=2))


def cmd_build(args: argparse.Namespace) -> None:
    """Build an episode set."""
    from stargazer.data import LightCurveCache
    from stargazer.episodes import BuildConfig, EpisodeSet
    from stargazer.targets import load_targets

    cfg = BuildConfig(
        n_episodes=args.n,
        seed=args.seed,
        null_frac=args.null_frac,
        period_range=(args.min_period, args.max_period),
        depth_range=(args.min_depth, args.max_depth),
    )
    es = EpisodeSet.build(
        load_targets(args.targets), LightCurveCache(args.cache), Path(args.out), cfg, Path(args.targets)
    )
    print(f"built {len(es)} episodes -> {es.root}\nepisode set hash {es.hash}")


def _progress(episode_id, prediction, truth_record):
    print(
        f"{episode_id} {truth_record['kind']:8s} det={prediction.detected!s:5s} P={prediction.period:.4f} "
        f"sde={prediction.extra.get('sde', float('nan')):.1f} ({prediction.extra.get('seconds')}s)",
        flush=True,
    )


def cmd_baseline(args: argparse.Namespace) -> None:
    """Run the BLS baseline on an episode set and write results."""
    from stargazer.baseline import BLSDetector
    from stargazer.episodes import EpisodeSet
    from stargazer.evaluate import evaluate

    es = EpisodeSet(args.episodes)
    det = BLSDetector(**(json.loads(args.config) if args.config else {}))
    ids = es.ids[: args.limit] if args.limit else None
    res = evaluate(det, es, ids=ids, period_rtol=args.period_rtol, progress=_progress)
    out = args.out or Path("results") / f"{det.name}_{es.root.name}_{_time.strftime('%Y%m%dT%H%M%S')}.json"
    res.save(out)
    print(json.dumps(res.summary, indent=1))
    print(f"results -> {out}")


def cmd_context(args: argparse.Namespace) -> None:
    """Fetch the catalogue context of each host and write context.json into an episode set."""
    from stargazer.context import CatalogueFetcher, ContextWriter
    from stargazer.provenance import Manifest
    from stargazer.targets import load_targets

    episode_dir = Path(args.episodes)
    path = ContextWriter(CatalogueFetcher()).write(episode_dir, load_targets(args.targets))
    Manifest.of_directory(episode_dir).write(episode_dir)
    n_contexts = len(json.loads(path.read_text()))
    print(
        f"wrote {path} for {n_contexts} episodes; new episode set hash {Manifest.digest_of(episode_dir).hex}"
    )


def cmd_score(args: argparse.Namespace) -> None:
    """Score an external predictions JSON against an episode set."""
    from stargazer.episodes import EpisodeSet
    from stargazer.evaluate import Results
    from stargazer.score import score_set

    es = EpisodeSet(args.episodes)
    preds = json.loads(Path(args.preds).read_text())
    rows, summary = score_set(preds, es.truth, period_rtol=args.period_rtol)
    res = Results(args.system, {"period_rtol": args.period_rtol}, str(es.root), es.hash, preds, rows, summary)
    out = args.out or Path("results") / f"{args.system}_{es.root.name}_{_time.strftime('%Y%m%dT%H%M%S')}.json"
    res.save(out)
    print(json.dumps(summary, indent=1))
    print(f"results -> {out}")


def cmd_agent(args: argparse.Namespace) -> None:
    """Run the tool-using agent with a model backend on a few episodes and score it."""
    from stargazer.agent import PROMPTS, ToolAgentDetector
    from stargazer.episodes import EpisodeSet
    from stargazer.evaluate import evaluate

    if args.backend == "openai":
        from stargazer.backends import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend(
            model=args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            temperature=0.7 if args.temperature is None else args.temperature,
            seed=args.seed,
        )
    else:
        from stargazer.backends import TinkerBackend

        backend = TinkerBackend(
            base_model=args.model,
            model_path=args.model_path,
            renderer=args.renderer,
            temperature=args.temperature,
            seed=args.seed,
        )
    es = EpisodeSet(args.episodes)
    det = ToolAgentDetector(backend, max_tool_calls=args.max_tool_calls, system_prompt=PROMPTS[args.prompt])
    ids = es.ids[: args.limit] if args.limit else None

    def progress(episode_id, prediction, truth_record):
        if "error" in prediction.extra:
            print(f"{episode_id} {truth_record['kind']:8s} ERROR {prediction.extra['error']}", flush=True)
            return
        usage = prediction.extra.get("usage", {})
        tokens = int(usage.get("input_tokens", 0) + usage.get("output_tokens", 0))
        extra = prediction.extra
        cost = f" usd={usage['usd']:.4f}" if "usd" in usage else ""
        print(
            f"{episode_id} {truth_record['kind']:8s} det={prediction.detected!s:5s} stop={extra.get('stop')} "
            f"calls={extra.get('n_tool_calls')} tokens={tokens}{cost}",
            flush=True,
        )

    res = evaluate(det, es, ids=ids, period_rtol=args.period_rtol, progress=progress, workers=args.workers)
    tag = args.model.replace("/", "_")
    out = (
        args.out
        or Path("results")
        / f"agent_{args.backend}_{tag}_{es.root.name}_{_time.strftime('%Y%m%dT%H%M%S')}.json"
    )
    res.save(out)
    costs = [pred["usage"]["usd"] for pred in res.preds.values() if "usd" in pred.get("usage", {})]
    print(json.dumps(res.summary, indent=1))
    if costs:
        print(f"estimated cost usd={sum(costs):.2f} over {len(costs)} episodes")
    print(f"results -> {out}")


def cmd_show(args: argparse.Namespace) -> None:
    """Print the trajectory of one episode from a results file: truth, thinking, tool calls, answer."""
    from rich.console import Console

    from stargazer.viewer import load, render_trajectory

    results, truth = load(Path(args.results))
    trajectory = render_trajectory(
        results.preds[args.episode],
        truth[args.episode],
        results.rows[args.episode],
        show_thinking=not args.no_thinking,
        full_results=args.full,
    )
    console = Console(width=args.width)
    console.rule(f"[bold]{args.episode}[/bold]  system={results.system}", align="left")
    console.print(trajectory)


def cmd_view(args: argparse.Namespace) -> None:
    """Open the terminal viewer on a results file."""
    from stargazer.viewer import run_viewer

    run_viewer(Path(args.results))


def cmd_calibrate(args: argparse.Namespace) -> None:
    """Sweep the detection threshold on a results file and print the table."""
    from stargazer.calibrate import choose_threshold, threshold_sweep
    from stargazer.episodes import EpisodeSet
    from stargazer.evaluate import Results

    res = Results.load(args.results)
    truth = EpisodeSet(res.episode_set).truth
    thresholds = [float(value) for value in args.thresholds.split(",")]
    rows = threshold_sweep(res, truth, thresholds, stat=args.stat)
    print(f"{'threshold':>9} {'false_alarm':>11} {'compl_exact':>11} {'compl_alias':>11} {'mean_score':>10}")
    for row in rows:
        print(
            f"{row['threshold']:9.2f} {row['false_alarm_rate']:11.3f} {row['completeness_exact']:11.3f} "
            f"{row['completeness_with_aliases']:11.3f} {row['mean_score']:10.3f}"
        )
    chosen = choose_threshold(res, truth, max_false_alarm=args.max_false_alarm, stat=args.stat)
    print(f"lowest threshold with false-alarm rate <= {args.max_false_alarm}: {chosen:.3f}")
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "results": args.results,
                    "episode_set_hash": res.episode_set_hash,
                    "stat": args.stat,
                    "sweep": rows,
                    "max_false_alarm": args.max_false_alarm,
                    "chosen_threshold": chosen,
                },
                indent=1,
            )
        )
        print(f"sweep -> {args.out}")


def cmd_compare(args: argparse.Namespace) -> None:
    """Print a comparison table for several results files on one episode set."""
    from stargazer.compare import compare, format_table

    rows, episode_set_hash = compare([Path(path) for path in args.results])
    print(f"episode set {episode_set_hash[:12]}\n")
    print(format_table(rows))


def cmd_verify(args: argparse.Namespace) -> None:
    """Recompute manifests. Exit with code 1 if a mismatch exists."""
    from stargazer.provenance import Manifest

    problems = []
    for directory in args.dirs:
        problems_here = Manifest.read(Path(directory)).verify(Path(directory))
        print(f"{directory}: {'OK' if not problems_here else f'{len(problems_here)} problems'}")
        problems += problems_here
    for problems_here in problems:
        print("  " + problems_here)
    sys.exit(1 if problems else 0)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    ap = argparse.ArgumentParser(prog="stargazer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    parser = sub.add_parser(
        "targets", help="select host stars with no TOI and no known eclipses for a sector"
    )
    parser.add_argument("--sector", type=int, required=True)
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tmag-max", type=float, default=12.0)
    parser.add_argument("--out", default=None)
    parser.set_defaults(func=cmd_targets)

    parser = sub.add_parser("remove-binaries", help="drop the known eclipsing binaries from a targets file")
    parser.add_argument("--targets", required=True)
    parser.add_argument("--out", required=True)
    parser.set_defaults(func=cmd_remove_binaries)

    parser = sub.add_parser("fetch", help="download and cache light curves for a targets file")
    parser.add_argument("--targets", required=True)
    parser.add_argument("--cache", default="data/cache")
    parser.set_defaults(func=cmd_fetch)

    parser = sub.add_parser("build", help="build an injection-recovery episode set")
    parser.add_argument("--targets", required=True)
    parser.add_argument("--cache", default="data/cache")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--null-frac", type=float, default=0.3)
    parser.add_argument("--min-period", type=float, default=0.5)
    parser.add_argument("--max-period", type=float, default=20.0)
    parser.add_argument("--min-depth", type=float, default=1e-4)
    parser.add_argument("--max-depth", type=float, default=1e-2)
    parser.set_defaults(func=cmd_build)

    parser = sub.add_parser("context", help="fetch catalogue rows for the hosts of an episode set")
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--targets", required=True)
    parser.set_defaults(func=cmd_context)

    parser = sub.add_parser("baseline", help="run the BLS baseline on an episode set and score it")
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--config", default=None, help="JSON overrides for BLSDetector")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--period-rtol", type=float, default=0.01)
    parser.add_argument("--out", default=None)
    parser.set_defaults(func=cmd_baseline)

    parser = sub.add_parser("score", help="score a predictions JSON against an episode set")
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--period-rtol", type=float, default=0.01)
    parser.add_argument("--out", default=None)
    parser.set_defaults(func=cmd_score)

    parser = sub.add_parser("agent", help="run the tool-using agent with a model backend")
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--backend", choices=["openai", "tinker"], default="openai")
    parser.add_argument("--model", required=True, help="provider model id, or Tinker base model")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--model-path", default=None, help="tinker:// checkpoint path")
    parser.add_argument(
        "--renderer",
        default="qwen3_disable_thinking",
        help="tinker_cookbook renderer name, for example qwen3_8_disable_thinking for Qwen3.8",
    )
    parser.add_argument(
        "--prompt",
        choices=["base", "rules"],
        default="base",
        help="system prompt: base states the task only; rules adds the procedure and decision rules",
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="default: the model card value for the renderer"
    )
    parser.add_argument("--seed", type=int, default=0, help="sampling seed; vary it to measure variance")
    parser.add_argument(
        "--max-tool-calls", type=int, default=None, help="safety stop; default is the detector's (40)"
    )
    parser.add_argument("--limit", type=int, default=3, help="episodes to run; 0 means all")
    parser.add_argument("--workers", type=int, default=8, help="episodes evaluated at the same time")
    parser.add_argument("--period-rtol", type=float, default=0.01)
    parser.add_argument("--out", default=None)
    parser.set_defaults(func=cmd_agent)

    parser = sub.add_parser("show", help="print one episode's trajectory from a results file")
    parser.add_argument("--results", required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--width", type=int, default=120, help="console width in characters")
    parser.add_argument("--full", action="store_true", help="print tool results in full")
    parser.add_argument("--no-thinking", action="store_true", help="hide the thinking trace")
    parser.set_defaults(func=cmd_show)

    parser = sub.add_parser("view", help="browse the trajectories in a results file (terminal UI)")
    parser.add_argument("--results", required=True)
    parser.set_defaults(func=cmd_view)

    parser = sub.add_parser("calibrate", help="sweep the detection threshold on a results file")
    parser.add_argument("--results", required=True)
    parser.add_argument("--stat", default="sde")
    parser.add_argument("--thresholds", default="5,6,7,8,9,10,12,15")
    parser.add_argument("--max-false-alarm", type=float, default=0.05)
    parser.add_argument("--out", default=None)
    parser.set_defaults(func=cmd_calibrate)

    parser = sub.add_parser("compare", help="table of several results files on one episode set")
    parser.add_argument("results", nargs="+")
    parser.set_defaults(func=cmd_compare)

    parser = sub.add_parser("verify", help="recompute SHA-256 manifests")
    parser.add_argument("dirs", nargs="+")
    parser.set_defaults(func=cmd_verify)
    return ap


def main(argv: list[str] | None = None) -> None:
    """Entry point. Loads ``.env`` from the current directory first, if one exists."""
    from dotenv import load_dotenv

    load_dotenv()
    args = build_parser().parse_args(argv)
    if args.cmd == "targets" and args.out is None:
        args.out = f"data/targets_s{args.sector}.json"
    args.func(args)


if __name__ == "__main__":
    main()
