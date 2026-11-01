#!/bin/sh
# The real-planet set with its eclipsing-binary entries filled from the disk copy and the entries at the hidden
# planet's period masked (commit a466ba9, LOGBOOK 2026-11-01): the context is rewritten first, then Inkling-Small on
# the rules prompt with the parity sentence, seed 1. Compare with the unmasked refill run
# (agent_inkling_small_s98_real_v2_rules_seed1_ebfilled.json, the leak) and the parity run on the empty fields
# (results/full_v15parity/agent_inkling_small_s98_real_v2_rules_seed1.json).
cd /home/siddharth899/personal_code/star_finder
set -a; . ./.env; set +a
.venv/bin/stargazer context --episodes data/episodes/s98_real_v2 --targets data/targets_s98_real.json \
    data/targets_s98_v3.json --mask-tois --gaia-source mast \
    --eclipsing-binaries data/catalogues/hlsp_tess-ebs_tess_lcf-ffi_s0001-s0026_tess_v1.0_cat.csv \
    > results/s98_real_v2/context_refill_masked.log 2>&1
echo "$(date -u +%FT%TZ) context refill masked exit $? $(tail -1 results/s98_real_v2/context_refill_masked.log)" >> results/s98_real_v2/progress.log
out=results/s98_real_v2/agent_inkling_small_s98_real_v2_rules_seed1_ebmasked
.venv/bin/stargazer agent --episodes data/episodes/s98_real_v2 --backend tinker --model thinkingmachines/Inkling-Small \
    --renderer tml_v0 --prompt rules --seed 1 --limit 0 --workers 8 --out $out.json > $out.log 2>&1
echo "$(date -u +%FT%TZ) finished $out exit $?" >> results/s98_real_v2/progress.log
echo "$(date -u +%FT%TZ) EB_MASKED_RUN_DONE" >> results/s98_real_v2/progress.log
