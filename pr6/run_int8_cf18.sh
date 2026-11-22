#!/bin/bash
# INT8 W8A8 rows at CF18 (default cache), host-direct, same shape as PR6 matrix.
cd "$(dirname "$0")/.."
OUT=outputs/bench_int8_cf18
PY=.venv/bin/python
mkdir -p "$OUT" && cp -an outputs/bench_int8/reference_cf18 "$OUT/reference" 2>/dev/null || true
export LINGBOT_WORLD_ROOT=$PWD/upstream/lingbot-world-v2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_COMPILE_THREADS=2
unset PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF LD_PRELOAD
export LB_CACHE_FRAMES=18 LB_STEPS=2 LB_ATTN=sage LB_ATTN_CROSS=cudnn \
  LB_COMPILE_DIT=1 LB_COMPILE_REFINER=0 LB_COMPILE_DECODER=0 \
  LB_GRAPHS_DIT=0 LB_GRAPHS_REFINER=0 LB_GRAPHS_DECODER=0 LB_EXPANDABLE_SEGMENTS=1 \
  LB_FP8=0 LB_SAGE_PRECENTER=0 LB_REFINER_DIRECT_KV=0 LB_REFINER_HEADS_PER_STREAM=5
S=$OUT/summary.log
run() {
  local name=$1 envs=$2; shift 2
  mkdir -p "$OUT/$name"
  echo "$(date -Is) start $name [$envs] $*" >> $S
  env $envs PR6_SAVE=$OUT/$name.video.pt $PY -u pr6/bench_row.py --model lingbot-1.3b --frames 81 --eager --warmup 2 --repeats 3 \
    --reference-dir $OUT/reference --history "$OUT/$name/history.jsonl" --run-dir "$OUT/$name/runs" \
    --stage-timing "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "$name|$rc|$envs|$*" >> $OUT/rows.txt
  echo "$(date -Is) $name exit=$rc" >> $S
}
: > $OUT/rows.txt
run ctrl "LB_INT8=0"
run int8_all "LB_INT8=1"
run int8_ffn "LB_INT8=ffn"
run int8_attention "LB_INT8=attention"
echo ALL_DONE >> $S
