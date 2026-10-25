#!/bin/sh
# Reference runs on the held-out set s92_v1: Inkling-Small, both prompts, seed 1, side by side,
# on the twelve-tool toolbox (orbit_shape as its own tool, commit 2e077bb).
cd /home/siddharth899/personal_code/star_finder
run() {
  out=results/s92_v1/agent_inkling_small_s92_v1_$1_seed1
  .venv/bin/stargazer agent --episodes data/episodes/s92_v1 --backend tinker --model thinkingmachines/Inkling-Small \
      --renderer tml_v0 --prompt $1 --seed 1 --limit 0 --workers 8 --out $out.json > $out.log 2>&1
  echo "$(date -u +%FT%TZ) finished $out exit $?" >> results/s92_v1/progress.log
}
run rules & run base &
wait
echo "$(date -u +%FT%TZ) REFERENCE_DONE" >> results/s92_v1/progress.log
