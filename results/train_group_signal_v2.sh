#!/bin/sh
# Group-signal measurement, second version: the detector's caps now reach the rollout runner (commit 06c7d9b;
# LOGBOOK 2026-11-01; the first version ran under the cookbook's 10-turn preset and up to a third of
# the rollouts were cut unscored). One training step each on the same eight stars of s98_v3 (seed 0, batch 0),
# Inkling-Small, rules prompt, learning rate 5e-5 (one step; the gradient is not the measurement). Read
# env/all/by_group/frac_mixed in each metrics.jsonl. Four runs side by side: group 4 at the cookbook's training
# temperature 1.0 is the baseline, then group 8, group 16, and group 4 at temperature 1.3.
cd /home/siddharth899/personal_code/star_finder
run() {
  name=$1; shift
  .venv/bin/stargazer train --episodes data/episodes/s98_v3 --model thinkingmachines/Inkling-Small \
      --renderer tml_v0 --prompt rules --batch-size 8 --steps 1 --learning-rate 5e-5 --max-tokens 4096 \
      --save-every 0 --seed 0 --log-path data/train/group_signal2_$name "$@" > results/train_group_signal2_$name.log 2>&1
  echo "$(date -u +%FT%TZ) finished $name exit $?" >> results/train_group_signal2.progress
}
run g4_t10 --group-size 4 --temperature 1.0 &
run g8_t10 --group-size 8 --temperature 1.0 &
run g16_t10 --group-size 16 --temperature 1.0 &
run g4_t13 --group-size 4 --temperature 1.3 &
wait
echo "$(date -u +%FT%TZ) GROUP_SIGNAL2_DONE" >> results/train_group_signal2.progress
