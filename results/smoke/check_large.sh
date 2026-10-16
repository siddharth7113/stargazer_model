#!/bin/sh
# Three-episode checks of the large models on s98_v3, base prompt, side by side.
cd /home/siddharth899/personal_code/star_finder
check() {
  out=results/smoke/agent_$1_s98_v3_base3
  .venv/bin/stargazer agent --episodes data/episodes/s98_v3 --backend tinker --model $2 \
      --renderer $3 --prompt base --limit 3 --out $out.json > $out.log 2>&1
  echo "$(date -u +%FT%TZ) finished $1 exit $?" >> results/smoke/check_large.progress
}
check inkling thinkingmachines/Inkling tml_v0 &
check kimi_k26 moonshotai/Kimi-K2.6 kimi_k26 &
check qwen35_397b Qwen/Qwen3.5-397B-A17B qwen3_5 &
check nemotron3_ultra nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 nemotron3_ultra &
check glm53 "zai-org/GLM-5.3:peft:262144" glm5_3_high_reasoning &
wait
echo "$(date -u +%FT%TZ) CHECKS_DONE" >> results/smoke/check_large.progress
