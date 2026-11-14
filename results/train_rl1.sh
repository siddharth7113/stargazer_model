#!/bin/sh
# Phase 5: the first real training run (PLAN.md Handoff of 2026-11-01). Inkling-Small, rules prompt, plain
# benchmark score as the reward, trained on s98_v3 and evaluated on the held-out set s92_v1 at steps 0, 10
# and 20. Group 8 at temperature 1.2 from the group-signal measurement (LOGBOOK 2026-11-01). The Handoff
# named 30 steps, but s98_v3 has 200 episodes, which gives 25 whole batches of 8; the cookbook would have
# trained 25 steps under a record of 30, so TrainingRun now refuses that and this run is one pass, 25 steps.
# A checkpoint every 10 steps and the final one; CheckpointPruner keeps every sampler checkpoint and only
# the newest training state on the free store (empty at the start, read through the REST client on
# 2026-11-14). Estimate: about $3 and 6 minutes a step, and three evaluations of 200 rollouts, so about
# $100 and three hours. Owner's standing yes for small models, 2026-11-01.
cd /home/siddharth899/personal_code/star_finder
name=rl1_inkling_small_s98_v3
git diff > results/train/${name}_uncommitted.diff
.venv/bin/stargazer train --episodes data/episodes/s98_v3 --test-episodes data/episodes/s92_v1 \
    --model thinkingmachines/Inkling-Small --renderer tml_v0 --prompt rules --reward score \
    --batch-size 8 --group-size 8 --temperature 1.2 --learning-rate 5e-5 \
    --steps 25 --eval-every 10 --save-every 10 --seed 0 \
    --log-path data/train/$name > results/train_$name.log 2>&1
echo "$(date -u +%FT%TZ) finished $name exit $?" >> results/train_rl1.progress
