#!/bin/bash
# Progress of a running sweep: stages done, runs finished, ETA.
cd "$(dirname "$0")/.." || exit 1
echo "time: $(date +%H:%M)   sweep alive: $(pgrep -cf '[s]weep.py')"
grep -E "=== stage" results/sweep.log 2>/dev/null | tail -4
cur=$(ls -t results/logs/*.log 2>/dev/null | head -1)
[ -n "$cur" ] && echo "--- $(basename "$cur") ---" && grep -E "^\[[0-9]+/" "$cur" | tail -3
echo "runs finished: $(find results/runs -name summary.json 2>/dev/null | wc -l)"
tail -1 "$cur" 2>/dev/null
