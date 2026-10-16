#!/bin/sh
# The remaining runs of the four large models on s98_v3: base seeds 2 and 3, rules seeds 1, 2, 3.
# Two chains for each model run at once (eight processes at eight workers), a step up from the four of run_large.sh.
cd /home/siddharth899/personal_code/star_finder
run() {
  out=results/full_v9large/agent_$1_s98_v3_$4_seed$5
  .venv/bin/stargazer agent --episodes data/episodes/s98_v3 --backend tinker --model $2 \
      --renderer $3 --prompt $4 --seed $5 --limit 0 --workers 8 --out $out.json > $out.log 2>&1
  echo "$(date -u +%FT%TZ) finished $out exit $?" >> results/full_v9large/progress.log
}
model() {
  (run $1 $2 $3 base 2; run $1 $2 $3 rules 1; run $1 $2 $3 rules 3) &
  (run $1 $2 $3 base 3; run $1 $2 $3 rules 2) &
}
model inkling thinkingmachines/Inkling tml_v0
model kimi_k26 moonshotai/Kimi-K2.6 kimi_k26
model qwen35_397b Qwen/Qwen3.5-397B-A17B qwen3_5
model nemotron3_ultra nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 nemotron3_ultra
wait
echo "$(date -u +%FT%TZ) SEEDS_LARGE_DONE" >> results/full_v9large/progress.log
