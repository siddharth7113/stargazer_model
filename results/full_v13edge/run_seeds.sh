#!/bin/sh
# Rerun of the measurement of the segment-edge verdict in bls_peaks and search_signals on the fixed code (edges from
# the unmasked light curve, commit f7f77f2; the first measurement in results/full_v12edge/ carried a wrong edge word
# after a mask, LOGBOOK 2026-10-31). Inkling-Small, rules prompt, seeds 1, 2, 3 on s98_v3 side by side, against the
# 2026-10-16 table (results/full_v8seeds/), and the confirmation on the held-out set s92_v1 at seed 1 against
# results/s92_v1/agent_inkling_small_s92_v1_rules_seed1.json.
cd /home/siddharth899/personal_code/star_finder
run() {
  out=results/full_v13edge/agent_inkling_small_$1_rules_seed$2
  .venv/bin/stargazer agent --episodes data/episodes/$1 --backend tinker --model thinkingmachines/Inkling-Small \
      --renderer tml_v0 --prompt rules --seed $2 --limit 0 --workers 8 --out $out.json > $out.log 2>&1
  echo "$(date -u +%FT%TZ) finished $out exit $?" >> results/full_v13edge/progress.log
}
run s98_v3 1 & run s98_v3 2 & run s98_v3 3 & run s92_v1 1 &
wait
echo "$(date -u +%FT%TZ) EDGE_RUNS_DONE" >> results/full_v13edge/progress.log
