#!/bin/sh
# Measurement of the rules prompt's parity sentence (step 3: "with two or more events of one parity the signal is
# still a period"; the decision rule accepts odd_even insufficient_data in that case). Decided with the owner on
# 2026-11-01 after TOI-2184.01 (four odd transits, no even, declined). The prompt hash changes, so this breaks the
# comparison with every rules run on file. Inkling-Small, frozen toolbox, seeds 1, 2, 3 on s98_v3 against
# results/full_v13edge/, seed 1 on the held-out s92_v1 against results/full_v13edge/, and seed 1 on the real-planet
# set s98_real_v2 against results/s98_real_v2/. Five processes at eight workers, about $50.
cd /home/siddharth899/personal_code/star_finder
run() {
  out=results/full_v15parity/agent_inkling_small_$1_rules_seed$2
  .venv/bin/stargazer agent --episodes data/episodes/$1 --backend tinker --model thinkingmachines/Inkling-Small \
      --renderer tml_v0 --prompt rules --seed $2 --limit 0 --workers 8 --out $out.json > $out.log 2>&1
  echo "$(date -u +%FT%TZ) finished $out exit $?" >> results/full_v15parity/progress.log
}
run s98_v3 1 & run s98_v3 2 & run s98_v3 3 & run s92_v1 1 & run s98_real_v2 1 &
wait
echo "$(date -u +%FT%TZ) PARITY_RUNS_DONE" >> results/full_v15parity/progress.log
