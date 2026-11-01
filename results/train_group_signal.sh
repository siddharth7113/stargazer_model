#!/bin/sh
# Group-signal measurement (PLAN handoff 2026-11-01, step 1 of the next session): one training step each on the same
# eight stars of s98_v3 (seed 0, batch 0, as the cost test), Inkling-Small, rules prompt, learning rate 5e-5 (one
# step; the gradient is not the measurement). Read env/all/by_group/frac_mixed in each metrics.jsonl against the cost
# test's step 0 (group 4, temperature 1.0: 1 mixed group of 8). Three runs side by side.
cd /home/siddharth899/personal_code/star_finder
run() {
  name=$1; shift
  .venv/bin/stargazer train --episodes data/episodes/s98_v3 --model thinkingmachines/Inkling-Small \
      --renderer tml_v0 --prompt rules --batch-size 8 --steps 1 --learning-rate 5e-5 --max-tokens 4096 \
      --save-every 0 --seed 0 --log-path data/train/group_signal_$name "$@" > results/train_group_signal_$name.log 2>&1
  echo "$(date -u +%FT%TZ) finished $name exit $?" >> results/train_group_signal.progress
}
run g8_t10 --group-size 8 --temperature 1.0 &
run g16_t10 --group-size 16 --temperature 1.0 &
run g4_t13 --group-size 4 --temperature 1.3 &
wait
echo "$(date -u +%FT%TZ) GROUP_SIGNAL_DONE" >> results/train_group_signal.progress
