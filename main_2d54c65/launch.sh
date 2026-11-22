#!/bin/bash
# Host side: pin main 2d54c65 (its own tracked harness), wait for GPU/RAM, run in the distrobox, restore checkout.
cd ~/work/bench-speed
OUT=outputs/bench_main_2d54c65; S=$OUT/summary.log; mkdir -p $OUT .bench_backup; : > $S
PREV=$(git rev-parse --abbrev-ref HEAD); [ "$PREV" = HEAD ] && PREV=$(git rev-parse HEAD); BK=.bench_backup/evaluation_bench_$(date +%s); MOVED=0
finish() {
  if [ $MOVED = 1 ]; then
    git checkout -q -f "$PREV" >> $S 2>&1 && rm -rf evaluation/bench && cp -a $BK evaluation/bench \
      && echo "restored checkout $PREV + untracked evaluation/bench" >> $S || echo "RESTORE FAILED (backup $BK)" >> $S
  fi
  grep -q ALL_DONE $S || echo ALL_DONE >> $S
}
trap finish EXIT
echo "prev branch $PREV; dirty: $(git status --porcelain | grep -v '^??' | wc -l) tracked changes" >> $S
git fetch -q origin main >> $S 2>&1 || echo "fetch failed (continuing with local objects)" >> $S
echo "origin/main = $(git rev-parse origin/main)" >> $S
SHA=$(git rev-parse --verify 2d54c65^{commit}) || { echo "no 2d54c65 exit=9" >> $S; exit 1; }
if [ -n "$(git ls-files evaluation/bench)" ]; then echo "evaluation/bench tracked on $PREV; not moving" >> $S
else cp -a evaluation/bench $BK && rm -rf evaluation/bench && MOVED=1; fi
if ! git checkout -q -B main-2d54c65-bench $SHA >> $S 2>&1; then
  echo "checkout failed exit=9" >> $S; git status --short | head >> $S; exit 1; fi
git rev-parse HEAD > main_2d54c65/sha.txt; echo "HEAD $(git log --oneline -1); tracked changes: $(git status --porcelain | grep -v '^??' | wc -l)" >> $S
for i in $(seq 120); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; [ $i = 1 ] && echo "waiting for GPU: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ' ')" >> $S; sleep 10; done
pgrep -x firefox >/dev/null && { pkill -x firefox; echo "killed firefox" >> $S; sleep 5; }
echo "MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MiB, GPU $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader), apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')" >> $S
if command -v distrobox >/dev/null && distrobox list 2>/dev/null | grep -q fastframes; then distrobox enter fastframes -- bash main_2d54c65/run.sh; else bash main_2d54c65/run.sh; fi
echo "run.sh rc=$?" >> $S
