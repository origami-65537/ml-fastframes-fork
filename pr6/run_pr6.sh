#!/bin/bash
# Runs inside the fastframes distrobox; launch.sh did the checkout on the host.
cd "$(dirname "$0")/.."
OUT=outputs/bench_pr6
eval "$(grep -m1 "^PY=" pr5/run_pr5.sh)"; PY=${PY:-python}
mkdir -p "$OUT/reference"
for f in outputs/bench/reference/*; do ln -sfn "$(realpath "$f")" "$OUT/reference/$(basename "$f")"; done
export PR6_SHA=$(cat pr6/sha.txt)
export LINGBOT_WORLD_ROOT=$PWD/upstream/lingbot-world-v2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF LD_PRELOAD
export LB_CACHE_FRAMES=10 LB_STEPS=2 LB_ATTN=sage LB_ATTN_CROSS=cudnn \
  LB_COMPILE_DIT=1 LB_COMPILE_REFINER=0 LB_COMPILE_DECODER=0 \
  LB_GRAPHS_DIT=0 LB_GRAPHS_REFINER=0 LB_GRAPHS_DECODER=0 LB_EXPANDABLE_SEGMENTS=1 \
  LB_FP8=0 LB_FP8_DIAG=scaled LB_FP8_DIRECT_INPUT=0 LB_DIT_PACKED_ONLY=0 \
  LB_SAGE_PRECENTER=0 LB_REFINER_DIRECT_KV=0 LB_REFINER_HEADS_PER_STREAM=5
S=$OUT/summary.log
G="LB_GRAPHS_DIT=1 LB_GRAPHS_REFINER=1 LB_GRAPHS_DECODER=1"
EX01="--example $LINGBOT_WORLD_ROOT/examples/01"

# run <name> <kind bench|fp8> "<env overrides>" [fixed_bench args]
run() {
  local name=$1 kind=$2 envs=$3; shift 3
  mkdir -p "$OUT/$name"
  echo "$(date -Is) start $name [$envs] $*" >> $S
  env $envs PR6_SAVE=$OUT/$name.video.pt $PY pr6/bench_row.py --model lingbot-1.3b --frames 81 --eager --warmup 2 --repeats 3 \
    --reference-dir $OUT/reference --history "$OUT/$name/history.jsonl" --run-dir "$OUT/$name/runs" \
    --stage-timing "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "$name|$kind|$rc|$envs|$*" >> $OUT/rows.txt
  echo "$(date -Is) $name exit=$rc" >> $S
  return $rc
}
smoke() {
  local name=$1 envs=$2
  echo "$(date -Is) start $name [$envs]" >> $S
  env $envs $PY pr6/smoke.py > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "$name|smoke|$rc|$envs|" >> $OUT/rows.txt
  echo "$(date -Is) $name exit=$rc" >> $S
}

: > $OUT/rows.txt
run ctrl_a bench "LB_FP8=0"
rc=$?
if [ $rc -ne 0 ] && [ -z "$(ls $OUT/ctrl_a/runs/*.json 2>/dev/null)" ]; then
  echo "ctrl_a crashed exit=$rc; aborting" >> $S; tail -40 $OUT/ctrl_a.log >> $S; echo ALL_DONE >> $S; exit 1
fi
run ctrl_b bench "LB_FP8=0"
run packed_off bench "LB_DIT_PACKED_ONLY=1"
smoke smoke_graphs "$G"
smoke smoke_graphs_packed "$G LB_DIT_PACKED_ONLY=1"
run graphs bench "$G"
run graphs_packed bench "$G LB_DIT_PACKED_ONLY=1"
for p in self_qkv self_out cross_q cross_out ffn_up ffn_down; do run fp8_$p fp8 "LB_FP8=$p"; done
run fp8_all fp8 "LB_FP8=1"
read worst passing < <($PY pr6/analyze.py pick)
echo "diag scope=$worst passing=$passing" >> $S
for d in weight activation both; do run diag_${d}_$worst fp8 "LB_FP8=$worst LB_FP8_DIAG=$d"; done
for p in self_qkv ffn_up cross_q; do run direct_$p fp8 "LB_FP8=$p LB_FP8_DIRECT_INPUT=1"; done
if [[ "$passing" == *,* ]]; then run combo_passing fp8 "LB_FP8=$passing"; fi
border=$($PY pr6/analyze.py border | head -4)
echo "boundary rows: $border" >> $S
if [ -n "$border" ]; then
  run ref_s1 bench "LB_ATTN=cudnn" --seed 1 --save-reference
  run ref_ex01 bench "LB_ATTN=cudnn" $EX01 --save-reference
  run ctrl_s1 bench "LB_FP8=0" --seed 1
  run ctrl_ex01 bench "LB_FP8=0" $EX01
  for b in $border; do
    envs=$(awk -F'|' -v n=$b '$1==n {print $4}' $OUT/rows.txt)
    run ${b}_s1 fp8 "$envs" --seed 1
    run ${b}_ex01 fp8 "$envs" $EX01
  done
fi
$PY pr6/analyze.py table > $OUT/analyze.log 2>&1; echo "analyze exit=$?" >> $S
grep -l "rowwise _scaled_mm unavailable" $OUT/*.log > $OUT/rowwise_warn.txt 2>/dev/null
echo ALL_DONE >> $S
