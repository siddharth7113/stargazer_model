#!/bin/sh
# Six full runs of Inkling-Small on s98_v3: three seeds of each prompt, one after another.
cd /home/siddharth899/personal_code/star_finder
for prompt in base rules; do
  for seed in 1 2 3; do
    out=results/full_v8seeds/agent_inkling_small_s98_v3_${prompt}_seed${seed}
    .venv/bin/stargazer agent --episodes data/episodes/s98_v3 --backend tinker --model thinkingmachines/Inkling-Small \
        --renderer tml_v0 --prompt $prompt --seed $seed --limit 0 --workers 8 --out $out.json > $out.log 2>&1
    echo "$(date -u +%FT%TZ) finished $out exit $?" >> results/full_v8seeds/progress.log
  done
done
echo "$(date -u +%FT%TZ) ALL_RUNS_DONE" >> results/full_v8seeds/progress.log
