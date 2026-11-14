#!/bin/sh
# The scripted-rules baseline (stargazer/scripted.py): the rules prompt's procedure and decision rules as fixed
# code over the same tools, with no language model. Default limits, none tuned on a set: the prompt's numbers,
# the tool descriptions' windows and the Kepler 7.1-sigma threshold. The three sets of the full_v15parity
# reference rows (LOGBOOK 2026-11-01). No API cost. The code is not committed yet; the diff is saved beside
# the results.
cd /home/siddharth899/personal_code/star_finder
git diff > results/scripted/uncommitted.diff
git status --short >> results/scripted/uncommitted.diff
for set in s98_v3 s92_v1 s98_real_v2; do
  .venv/bin/stargazer baseline --episodes data/episodes/$set --detector scripted --workers 8 \
      --out results/scripted/scripted_rules_$set.json > results/scripted/scripted_rules_$set.log 2>&1
  echo "$(date -u +%FT%TZ) finished $set exit $?" >> results/scripted/run_reference.progress
done
