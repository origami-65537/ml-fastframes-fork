#!/bin/bash
# Inside the distrobox: main 2d54c65. LingBot: own cuDNN reference at main defaults, then default x3. FF-S: expandable_segments.
cd "$(dirname "$0")/.."
OUT=outputs/bench_main_2d54c65_b; S=$OUT/summary.log; PY=.venv/bin/python
LBREF=outputs/bench/reference_main_defaults_cudnn; REF=outputs/bench/reference
export LINGBOT_WORLD_ROOT=$PWD/upstream/lingbot-world-v2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for v in $(compgen -e | grep '^LB_'); do unset $v; done
unset PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF LD_PRELOAD
$PY -c "import lpips" 2>/dev/null && echo "lpips ok" >> $S || echo "LPIPS MISSING (gate incomplete)" >> $S
$PY -c "from apps.lingbot.sage_attention import load_sage; load_sage(); print('sage importable (auto -> sage)')" >> $S 2>&1
waitram() { for i in $(seq 24); do [ "$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)" -ge 25600 ] && break; sleep 5; done
  echo "MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MiB" >> $S; }
row() { local name=$1 refdir=$2 envs=$3; shift 3; waitram
  rm -rf $OUT/$name; mkdir -p $OUT/$name
  echo "=== $name $(date -Is) env: $envs ref: $refdir args: $*" >> $S
  env $envs $PY evaluation/bench/fixed_bench.py "$@" --reference-dir $refdir --history $OUT/$name/history.jsonl --run-dir $OUT/$name/runs > $OUT/$name.log 2>&1 &
  local pid=$! hwm=0 v rc
  while kill -0 $pid 2>/dev/null; do v=$(awk '/VmHWM/{print $2}' /proc/$pid/status 2>/dev/null); [ -n "$v" ] && hwm=$v; sleep 2; done
  wait $pid; rc=$?
  echo "exit=$rc peak_host_rss_gb=$(awk -v k=$hwm 'BEGIN{printf "%.1f", k/1048576}') $(date -Is)" >> $S
  grep -E "WARNING: no reference|FAIL:|LOW-CONFIDENCE|Max \|delta\||[Ss]aved" $OUT/$name.log | head -8 >> $S
  [ $rc -ne 0 ] && tail -n 25 $OUT/$name.log >> $S
  grep -inE "recompil|Traceback|OutOfMemory" $OUT/$name.log | head -10 > $OUT/$name.warnings.txt
  return $rc; }
mkdir -p $LBREF
row lb_ref_cudnn $LBREF "LB_ATTN=cudnn" --model lingbot-1.3b --frames 81 --warmup 1 --repeats 1 --save-reference
[ $? = 137 ] && exit 137
ls -la $LBREF >> $S
row lingbot_default $LBREF "" --model lingbot-1.3b --frames 81 --warmup 2 --repeats 3 --stage-timing
[ $? = 137 ] && exit 137
FFKEY=$($PY -c "import sys; sys.path.insert(0,'.'); from evaluation.bench.fixed_bench import workload_key as k; print(k({'model':'ff-s','prompt':'A corgi surfing a wave at sunset','seed':0,'frames':81,'example':None}))")
echo "ff-s reference key $FFKEY" >> $S
EXP="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
if [ -f $REF/$FFKEY.pt ]; then echo "ff-s reference exists; not overwriting" >> $S
else row ffs_save_reference $REF "$EXP" --model ff-s --warmup 1 --repeats 1 --save-reference; [ $? = 137 ] && exit 137
  ls -la $REF/$FFKEY.* >> $S 2>&1; fi
[ -f $REF/$FFKEY.pt ] && row ffs_expandable $REF "$EXP" --model ff-s --warmup 2 --repeats 3
$PY main_2d54c65/summary.py $OUT/lb_ref_cudnn $OUT/lingbot_default $OUT/ffs_save_reference $OUT/ffs_expandable > $OUT/summary.md 2>&1
exit 0
