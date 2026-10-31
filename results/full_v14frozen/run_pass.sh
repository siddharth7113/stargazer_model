#!/bin/sh
# The final Phase 3 table: the four large models on the frozen toolbox (commit f7f77f2, segment-edge verdict in the
# search), rules prompt, seed 1, s98_v3, four processes at eight workers. Owner's yes on 2026-10-31, about $300.
# Compare with the rules seed-1 rows of results/full_v9large/ (twelve-tool box without the verdict).
cd /home/siddharth899/personal_code/star_finder
run() {
  out=results/full_v14frozen/agent_$1_s98_v3_rules_seed1
  .venv/bin/stargazer agent --episodes data/episodes/s98_v3 --backend tinker --model $2 \
      --renderer $3 --prompt rules --seed 1 --limit 0 --workers 8 --out $out.json > $out.log 2>&1
  echo "$(date -u +%FT%TZ) finished $out exit $?" >> results/full_v14frozen/progress.log
}
run inkling thinkingmachines/Inkling tml_v0 &
run nemotron3_ultra nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 nemotron3_ultra &
run qwen35_397b Qwen/Qwen3.5-397B-A17B qwen3_5 &
run kimi_k26 moonshotai/Kimi-K2.6 kimi_k26 &
wait
echo "$(date -u +%FT%TZ) PASS_DONE" >> results/full_v14frozen/progress.log
