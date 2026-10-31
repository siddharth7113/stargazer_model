#!/bin/sh
# First run on the real-planet set s98_real_v1 (43 confirmed single-planet hosts of sector 98 with the flux as
# observed and the truth from the TOI list, 40 null episodes from the s98_v3 hosts, TOI entries masked from the
# context): Inkling-Small on the frozen toolbox (commit f7f77f2), rules prompt, seed 1, and the BLS baseline.
cd /home/siddharth899/personal_code/star_finder
.venv/bin/stargazer baseline --episodes data/episodes/s98_real_v1 --out results/baseline_bls_s98_real_v1.json \
    > results/baseline_bls_s98_real_v1.log 2>&1
echo "$(date -u +%FT%TZ) finished baseline exit $?" >> results/s98_real_v1/progress.log
out=results/s98_real_v1/agent_inkling_small_s98_real_v1_rules_seed1
.venv/bin/stargazer agent --episodes data/episodes/s98_real_v1 --backend tinker --model thinkingmachines/Inkling-Small \
    --renderer tml_v0 --prompt rules --seed 1 --limit 0 --workers 8 --out $out.json > $out.log 2>&1
echo "$(date -u +%FT%TZ) finished $out exit $?" >> results/s98_real_v1/progress.log
echo "$(date -u +%FT%TZ) REAL_REFERENCE_DONE" >> results/s98_real_v1/progress.log
