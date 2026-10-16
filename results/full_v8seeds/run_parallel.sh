#!/bin/sh
# The remaining runs side by side. base seed 2 was already running from run_seeds.sh when the chain was stopped.
cd /home/siddharth899/personal_code/star_finder
run() {
  out=results/full_v8seeds/agent_inkling_small_s98_v3_$1_seed$2
  .venv/bin/stargazer agent --episodes data/episodes/s98_v3 --backend tinker --model thinkingmachines/Inkling-Small \
      --renderer tml_v0 --prompt $1 --seed $2 --limit 0 --workers 8 --out $out.json > $out.log 2>&1
  echo "$(date -u +%FT%TZ) finished $out exit $?" >> results/full_v8seeds/progress.log
}
run base 3 & run rules 1 & run rules 2 & run rules 3 &
wait
echo "$(date -u +%FT%TZ) PARALLEL_DONE" >> results/full_v8seeds/progress.log
