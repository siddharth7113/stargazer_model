#!/bin/sh
# Three-episode check of the rules prompt's new parity sentence ("with two or more events of one parity the signal
# is still a period"): Inkling-Small, seed 1, on s98_real_v2 ep_0015 (TOI-2184.01, four odd transits and no even,
# declined on 2026-11-01), ep_0012 (TOI-942.01, missed) and ep_0041 (a null host). Run before the measurement.
cd /home/siddharth899/personal_code/star_finder
set -a; . ./.env; set +a
.venv/bin/python - <<'PY' > results/smoke/agent_inkling_small_s98_real_v2_rules3_parity.log 2>&1
from stargazer.agent import PROMPTS, ToolAgentDetector
from stargazer.backends import TinkerBackend
from stargazer.episodes import EpisodeSet
from stargazer.evaluate import evaluate

backend = TinkerBackend(base_model="thinkingmachines/Inkling-Small", renderer="tml_v0", seed=1)
es = EpisodeSet("data/episodes/s98_real_v2")
det = ToolAgentDetector(backend, system_prompt=PROMPTS["rules"])
def progress(episode_id, prediction, truth_record):
    extra = prediction.extra
    print(episode_id, truth_record["kind"], "det=", prediction.detected, "period=", prediction.period,
          "stop=", extra.get("stop"), "calls=", extra.get("n_tool_calls"), "usd=", extra.get("usage", {}).get("usd"), flush=True)
res = evaluate(det, es, ids=["ep_0012", "ep_0015", "ep_0041"], progress=progress, workers=3)
res.save("results/smoke/agent_inkling_small_s98_real_v2_rules3_parity.json")
print(res.summary)
print("cost", sum(p["usage"]["usd"] for p in res.preds.values() if "usd" in p.get("usage", {})))
PY
echo "$(date -u +%FT%TZ) finished check_parity exit $?" >> results/smoke/check_parity.progress
