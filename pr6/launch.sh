#!/bin/bash
# Host side: checkout PR 6 head with host git, free RAM, then run all rows in the distrobox.
cd ~/work/bench-speed
BR=devin/1795309676-lingbot-fp8-restart
OUT=outputs/bench_pr6; S=$OUT/summary.log
mkdir -p $OUT; : > $S
if ! { git fetch -q origin $BR && git checkout -q -B $BR origin/$BR && git rev-parse HEAD > pr6/sha.txt; } >> $S 2>&1; then
  echo "checkout failed exit=9" >> $S; git status --short | head -20 >> $S; echo ALL_DONE >> $S; exit 1
fi
echo "HEAD $(git log --oneline -1)" >> $S
for i in $(seq 60); do
  avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  [ "$avail" -ge 25600 ] && break
  [ $i -eq 1 ] && pgrep -x firefox >/dev/null && { pkill -x firefox; echo "killed firefox (MemAvailable ${avail} MiB)" >> $S; }
  sleep 10
done
echo "MemAvailable $(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo) MiB, GPU: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" >> $S
if command -v distrobox >/dev/null && distrobox list 2>/dev/null | grep -q fastframes; then distrobox enter fastframes -- bash pr6/run_pr6.sh; else bash pr6/run_pr6.sh; fi
echo "run_pr6 exit=$?" >> $S
grep -q ALL_DONE $S || echo ALL_DONE >> $S
