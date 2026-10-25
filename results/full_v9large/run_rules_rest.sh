#!/bin/sh
# The eight rules runs not launched on 2026-10-16, and the Kimi K2.6 rerun with the raised
# caps (--max-turns 60 --max-tokens 32768) at one seed for each prompt. Launched 2026-10-25
# with the owner's yes. All ten at once, each at eight workers, to fit the owner's window.
cd /home/siddharth899/personal_code/star_finder
run() {
  out=results/full_v9large/agent_$1_s98_v3_$4_seed$5$6
  .venv/bin/stargazer agent --episodes data/episodes/s98_v3 --backend tinker --model $2 \
      --renderer $3 --prompt $4 --seed $5 --limit 0 --workers 8 $7 --out $out.json > $out.log 2>&1
  echo "$(date -u +%FT%TZ) finished $out exit $?" >> results/full_v9large/progress.log
}
CAPS="--max-turns 60 --max-tokens 32768"
run inkling thinkingmachines/Inkling tml_v0 rules 3 &
run nemotron3_ultra nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 nemotron3_ultra rules 3 &
run qwen35_397b Qwen/Qwen3.5-397B-A17B qwen3_5 rules 1 &
run qwen35_397b Qwen/Qwen3.5-397B-A17B qwen3_5 rules 2 &
run qwen35_397b Qwen/Qwen3.5-397B-A17B qwen3_5 rules 3 &
run kimi_k26 moonshotai/Kimi-K2.6 kimi_k26 rules 1 &
run kimi_k26 moonshotai/Kimi-K2.6 kimi_k26 rules 2 &
run kimi_k26 moonshotai/Kimi-K2.6 kimi_k26 rules 3 &
run kimi_k26 moonshotai/Kimi-K2.6 kimi_k26 base 1 _caps "$CAPS" &
run kimi_k26 moonshotai/Kimi-K2.6 kimi_k26 rules 1 _caps "$CAPS" &
wait
echo "$(date -u +%FT%TZ) RULES_REST_DONE" >> results/full_v9large/progress.log
