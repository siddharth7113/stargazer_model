#!/bin/sh
# The planet-free stars of the multi-sector real set: for each sector of select_and_fetch.sh, 60 quiet dwarf
# hosts with no TOI and no known eclipses (stargazer targets), seed 1. The binary cut reads the TESS
# eclipsing-binary catalogue from the committed MAST copy and the Gaia classes from the Harvard Vizier mirror:
# on 2026-11-14 Strasbourg answered but returned 404 for every query. The downloads wait for the planet hosts'
# downloads, so that two fetch runs never rewrite the cache manifest at the same time.
cd /home/siddharth899/personal_code/star_finder
ebs=data/catalogues/hlsp_tess-ebs_tess_lcf-ffi_s0001-s0026_tess_v1.0_cat.csv
for sector in 90 91 93 94 95 96 97 99 100; do
  .venv/bin/stargazer targets --sector $sector --n 60 --seed 1 --eclipsing-binaries $ebs \
      --vizier-server vizier.cfa.harvard.edu --out data/targets_s${sector}_null.json \
      > results/real_multi/targets_null_s${sector}.log 2>&1
  echo "$(date -u +%FT%TZ) null targets s$sector exit $?" >> results/real_multi/progress_null.log
done
until grep -q SELECT_AND_FETCH_DONE results/real_multi/progress.log 2>/dev/null; do sleep 30; done
for sector in 90 91 93 94 95 96 97 99 100; do
  .venv/bin/stargazer fetch --targets data/targets_s${sector}_null.json > results/real_multi/fetch_null_s${sector}.log 2>&1
  echo "$(date -u +%FT%TZ) null fetch s$sector exit $? $(tail -1 results/real_multi/fetch_null_s${sector}.log)" >> results/real_multi/progress_null.log
done
echo "$(date -u +%FT%TZ) NULL_DONE" >> results/real_multi/progress_null.log
