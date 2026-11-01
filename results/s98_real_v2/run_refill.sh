#!/bin/sh
# The real-planet set with its eclipsing-binary entries filled from the disk copy of the TESS catalogue
# (context hash aaddf8d5..., LOGBOOK 2026-11-01), on the rules prompt with the parity sentence (commit 568dd9b,
# code 0d96a14). Inkling-Small, seed 1. Compare with results/full_v15parity/agent_inkling_small_s98_real_v2_rules_seed1.json
# (same prompt, empty eclipsing-binary fields) and results/s98_real_v2/agent_inkling_small_s98_real_v2_rules_seed1.json
# (the earlier prompt).
cd /home/siddharth899/personal_code/star_finder
out=results/s98_real_v2/agent_inkling_small_s98_real_v2_rules_seed1_ebfilled
.venv/bin/stargazer agent --episodes data/episodes/s98_real_v2 --backend tinker --model thinkingmachines/Inkling-Small \
    --renderer tml_v0 --prompt rules --seed 1 --limit 0 --workers 8 --out $out.json > $out.log 2>&1
echo "$(date -u +%FT%TZ) finished $out exit $?" >> results/s98_real_v2/progress.log
echo "$(date -u +%FT%TZ) EB_REFILL_RUN_DONE" >> results/s98_real_v2/progress.log
