#!/bin/sh
# Three-episode checks of the four large models on the frozen toolbox (segment-edge verdict in the search, commit
# f7f77f2), rules prompt, seed 1, side by side, before the final Phase 3 pass. Episodes: ep_0134 (a ramp signal first
# and a masked second signal behind it), ep_0063 (a findable 3.14-day planet on a host whose ramp was submitted as a
# false alarm), ep_0033 (a ramp only). Launched 2026-10-31 with the owner's yes to the pass.
cd /home/siddharth899/personal_code/star_finder
set -a; . ./.env; set +a
check() {
  .venv/bin/python - "$1" "$2" "$3" <<'PY' > results/smoke/agent_$1_s98_v3_rules3_frozen.log 2>&1
import sys
from stargazer.agent import PROMPTS, ToolAgentDetector
from stargazer.backends import TinkerBackend
from stargazer.episodes import EpisodeSet
from stargazer.evaluate import evaluate
tag, model, renderer = sys.argv[1:4]
backend = TinkerBackend(base_model=model, renderer=renderer, seed=1)
det = ToolAgentDetector(backend, system_prompt=PROMPTS["rules"])
def progress(episode_id, prediction, truth_record):
    extra = prediction.extra
    print(episode_id, truth_record["kind"], "det=", prediction.detected, "period=", prediction.period, "stop=", extra.get("stop"),
          "calls=", extra.get("n_tool_calls"), "usd=", extra.get("usage", {}).get("usd"), "error=", extra.get("error"), flush=True)
res = evaluate(det, EpisodeSet("data/episodes/s98_v3"), ids=["ep_0134", "ep_0063", "ep_0033"], progress=progress, workers=3)
res.save(f"results/smoke/agent_{tag}_s98_v3_rules3_frozen.json")
print(res.summary); print("cost", sum(p["usage"]["usd"] for p in res.preds.values() if "usd" in p.get("usage", {})))
PY
  echo "$(date -u +%FT%TZ) finished $1 exit $?" >> results/smoke/check_frozen.progress
}
check inkling thinkingmachines/Inkling tml_v0 &
check nemotron3_ultra nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 nemotron3_ultra &
check qwen35_397b Qwen/Qwen3.5-397B-A17B qwen3_5 &
check kimi_k26 moonshotai/Kimi-K2.6 kimi_k26 &
wait
echo "$(date -u +%FT%TZ) CHECKS_DONE" >> results/smoke/check_frozen.progress
