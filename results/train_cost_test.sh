#!/bin/sh
# Phase 5 step 3: the cost of a training step. Inkling-Small, rules prompt, s98_v3, three steps of 8 groups of 4
# (96 episodes), learning rate 5e-5 (not calibrated for Inkling in the cookbook), no checkpoint. The bill in the
# Tinker console against the LOGBOOK estimate (2026-11-01) is the measurement. Owner's standing yes for small
# models, 2026-11-01.
cd /home/siddharth899/personal_code/star_finder
.venv/bin/stargazer train --episodes data/episodes/s98_v3 --model thinkingmachines/Inkling-Small \
    --renderer tml_v0 --prompt rules --batch-size 8 --group-size 4 --steps 3 \
    --learning-rate 5e-5 --max-tokens 4096 --save-every 0 \
    --log-path data/train/cost_test_inkling_small_s98_v3 > results/train_cost_test.log 2>&1
echo "$(date -u +%FT%TZ) finished train_cost_test exit $?" >> results/train_cost_test.progress
