#!/bin/sh
# First measurement of the orbit shape verdict in fold (commit 77cc5b3): Inkling-Small on s98_v3,
# rules prompt, seeds 1, 2, 3 side by side, against the 2026-10-16 table (results/full_v8seeds/).
cd /home/siddharth899/personal_code/star_finder
run() {
  out=results/full_v10shape/agent_inkling_small_s98_v3_rules_seed$1
  .venv/bin/stargazer agent --episodes data/episodes/s98_v3 --backend tinker --model thinkingmachines/Inkling-Small \
      --renderer tml_v0 --prompt rules --seed $1 --limit 0 --workers 8 --out $out.json > $out.log 2>&1
  echo "$(date -u +%FT%TZ) finished $out exit $?" >> results/full_v10shape/progress.log
}
run 1 & run 2 & run 3 &
wait
echo "$(date -u +%FT%TZ) SHAPE_SEEDS_DONE" >> results/full_v10shape/progress.log
