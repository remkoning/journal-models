#!/bin/zsh
# Overnight credit runs (v2 design) for the given journals, e.g.:  nohup scripts/credit_overnight.sh SMJ MgmtSci &
# If the plan's usage limit interrupts a run, wait and rerun: completed calls come back from the cache.
cd "${0:A:h}/.."
M=600
for J in "$@"; do
  for attempt in 1 2 3 4 5 6 7 8; do
    .venv/bin/python -m scripts.run_credit --journal $J --design own --anonymous --k 10 --n-probes 48 \
      --m $M --workers 6 > results/credit_${J}_v2.log 2>&1
    grep -q "$M/$M subsets succeeded" results/credit_${J}_v2.log && break
    echo "$(date) $J attempt $attempt incomplete; waiting 20 min" >> results/credit_overnight.log
    sleep 1200
  done
  echo "$(date) $J done: $(grep 'subsets succeeded' results/credit_${J}_v2.log)" >> results/credit_overnight.log
done
