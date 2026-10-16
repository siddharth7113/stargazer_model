#!/bin/sh
# One full run of each large model on s98_v3, base prompt, thinking formats, seed 1, side by side.
cd /home/siddharth899/personal_code/star_finder
run() {
  out=results/full_v9large/agent_$1_s98_v3_base_seed1
  .venv/bin/stargazer agent --episodes data/episodes/s98_v3 --backend tinker --model $2 \
      --renderer $3 --prompt base --seed 1 --limit 0 --workers 8 --out $out.json > $out.log 2>&1
  echo "$(date -u +%FT%TZ) finished $1 exit $?" >> results/full_v9large/progress.log
}
run inkling thinkingmachines/Inkling tml_v0 &
run kimi_k26 moonshotai/Kimi-K2.6 kimi_k26 &
run qwen35_397b Qwen/Qwen3.5-397B-A17B qwen3_5 &
run nemotron3_ultra nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 nemotron3_ultra &
wait
echo "$(date -u +%FT%TZ) LARGE_DONE" >> results/full_v9large/progress.log
