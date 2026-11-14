#!/bin/sh
# Step 3 of 2026-11-14: a real-planet episode set over many sectors. For each sector, the hosts with exactly one
# confirmed TOI planet (CP or KP), Tmag 12 or brighter, a full ephemeris and a positive period
# (stargazer targets --real), then their 120 s light curves from MAST. Sectors 90 to 100 without 92 and 98: 92
# and 98 already hold sets, and sectors 101 and later did not list their stars on 2026-11-14. The light
# curves are from 2025, after the models' training. Public archives only; no API cost.
cd /home/siddharth899/personal_code/star_finder
for sector in 90 91 93 94 95 96 97 99 100; do
  .venv/bin/stargazer targets --real --sector $sector --out data/targets_s${sector}_real.json \
      > results/real_multi/targets_s${sector}.log 2>&1
  echo "$(date -u +%FT%TZ) targets s$sector exit $?" >> results/real_multi/progress.log
  .venv/bin/stargazer fetch --targets data/targets_s${sector}_real.json > results/real_multi/fetch_s${sector}.log 2>&1
  echo "$(date -u +%FT%TZ) fetch s$sector exit $? $(tail -1 results/real_multi/fetch_s${sector}.log)" >> results/real_multi/progress.log
done
echo "$(date -u +%FT%TZ) SELECT_AND_FETCH_DONE" >> results/real_multi/progress.log
