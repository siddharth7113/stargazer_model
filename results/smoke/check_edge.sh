#!/bin/sh
# Three-episode check of the segment-edge verdict in bls_peaks and search_signals: Inkling-Small, rules prompt,
# seed 1, on the three null episodes whose 14.4-day and 14.6-day ramp signals every model submitted
# (TIC 33831807 ep_0134 and ep_0155, TIC 146420104 ep_0033). Run before the three-seed measurement.
cd /home/siddharth899/personal_code/star_finder
set -a; . ./.env; set +a
.venv/bin/python - <<'PY' > results/smoke/agent_inkling_small_s98_v3_rules3_edge.log 2>&1
from stargazer.agent import PROMPTS, ToolAgentDetector
from stargazer.backends import TinkerBackend
from stargazer.episodes import EpisodeSet
from stargazer.evaluate import evaluate

backend = TinkerBackend(base_model="thinkingmachines/Inkling-Small", renderer="tml_v0", seed=1)
es = EpisodeSet("data/episodes/s98_v3")
det = ToolAgentDetector(backend, system_prompt=PROMPTS["rules"])
def progress(episode_id, prediction, truth_record):
    extra = prediction.extra
    print(episode_id, truth_record["kind"], "det=", prediction.detected, "period=", prediction.period,
          "stop=", extra.get("stop"), "calls=", extra.get("n_tool_calls"), "usd=", extra.get("usage", {}).get("usd"), flush=True)
res = evaluate(det, es, ids=["ep_0033", "ep_0134", "ep_0155"], progress=progress, workers=3)
res.save("results/smoke/agent_inkling_small_s98_v3_rules3_edge.json")
print(res.summary)
print("cost", sum(p["usage"]["usd"] for p in res.preds.values() if "usd" in p.get("usage", {})))
PY
echo "$(date -u +%FT%TZ) finished check_edge exit $?" >> results/smoke/check_edge.progress
