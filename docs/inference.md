# Inference

Run commands from the repository root after [installation and checkpoint setup](installation.md).
Measurements below use NVIDIA B200 GPUs with PyTorch 2.9.1 / CUDA 12.8.
For camera-controlled generation and Fig. 5, see the [LingBot guide](lingbot.md).

## Profiling

### Full-step latency

These commands profile each component and sum their latencies. Baselines require
[their upstream clones](installation.md#baselines); eight-GPU runs require one
node with NVSwitch (NVLink multicast). Comments give the paper's B200 measurements:

```bash
uv run python evaluation/latency/benchmark.py ff-s                           # 36 ms
uv run python evaluation/latency/benchmark.py ff-s --gpus 8                  # 14 ms
uv run python evaluation/latency/benchmark.py ff-l                           # 158 ms
uv run python evaluation/latency/benchmark.py ff-l --gpus 8                   # 45 ms
uv run --extra baselines python evaluation/latency/benchmark.py cfpp         # 272 ms
uv run --extra baselines python evaluation/latency/benchmark.py self-forcing # 717 ms
```

Weights default to `checkpoints/` (`--checkpoint-dir`); records and `latency.json`
go to `outputs/latency/<method>_<gpus>gpu/` (`--output-dir`). The first run compiles
and autotunes; `TORCHINDUCTOR_CACHE_DIR` selects the compiler cache location.

Measurements use batch 1, BF16/TF32, and a full 21-latent-frame cache. After warmup,
each component is timed over a sustained window (default 60 s), taking the median
time per call and the slowest rank. The Wan VAE uses CUDA-event timing over streamed
decodes. Step latency sums component medians and excludes text encoding, host
transfer and display. Each latent frame produces four RGB frames.

| Method | Generation unit | Step latency |
|---|---|---|
| FastFrames | 1 latent frame | 2 DiT evaluations + KV refinement + pixel decoder |
| CF++ | 1 latent frame | 3 DiT evaluations + VAE decode |
| Self-Forcing | 3 latent frames | 5 DiT evaluations + 3 single-latent VAE decodes |
| CausVid | 3 latent frames | 4 DiT evaluations + 3 single-latent VAE decodes |

Baseline DiT counts include the clean KV-cache write. Timings vary with hardware
conditions and compiler autotuning. Fig. 1 plots these latencies against VBench Total.

### Components and summaries

For the Fig. 2–3 components and Table 2 profiles:

```bash
CKPT=checkpoints
OUT=outputs/latency/components
P="uv run python evaluation/latency/profile_components.py"
$P dit --model 1.3b --dit $CKPT/causal-forcing++/framewise-2step.pt \
    --timestep 1000 --output $OUT/dit_first.json
$P dit --model 1.3b --dit $CKPT/causal-forcing++/framewise-2step.pt --output $OUT/dit_final.json
$P kv-refine --model 1.3b --dit $CKPT/causal-forcing++/framewise-2step.pt \
    --refiner $CKPT/fastframes-s/kv_refiner.safetensors --streams 6 --output $OUT/kv.json
$P pixel-decoder --model 1.3b --decoder $CKPT/fastframes-s/pixel_decoder.safetensors \
    --output $OUT/decoder.json
$P vae --vae $CKPT/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth --output $OUT/vae.json
uv run python evaluation/latency/summarize.py outputs/latency --output outputs/latency/summary.json
```

Use `--model 14b` and FastFrames-L checkpoints for the larger model. For eight-GPU
DiT, KV-refinement and pixel-decoder profiles, replace `uv run python` with
`uv run torchrun --standalone --nproc_per_node 8`. KV refinement uses
`--shard tokens` for 1.3B and `--shard layers` for 14B. Set `--streams 1` for
14B or multi-GPU KV profiles. Add
`--resolution 720p` for Table 2, using separate output filenames. Omit checkpoint
flags to profile random weights; full-step benchmarks require checkpoints.

DiT options include `--mode baseline`, `--split-kv 1`, and `--chunk 3`; VAE also
accepts `--chunk 3`. `--dry-run` prints a component's configuration without running it.
`check_dit.py` compares optimized and reference outputs with released weights:

```bash
uv run python evaluation/latency/check_dit.py --model 1.3b \
    --dit $CKPT/causal-forcing++/framewise-2step.pt --fp32-reference --output $OUT/check.json
```

The summarizer selects the fastest matching component record and lists its
candidates. Use separate result directories when comparing runs or environments.

## Generation

### Single prompts and batches

```bash
uv run python scripts/generate.py --model ff-s --prompt "A corgi surfing a wave at sunset" \
    --seed 0 --output outputs/corgi.mp4
```

Use `--model ff-l` for FastFrames-L or `--checkpoint-dir` to change the weights
directory. Both produce 832 × 480 H.264 MP4s, with 81 frames at 16 fps.
For a file with one prompt per line, start one process per GPU:

```bash
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i uv run --no-sync python scripts/generate.py --model ff-l \
      --prompt-file prompts.txt --output-dir outputs/ff-l --shard $i --num-shards 8 &
done; wait
```

Prompt index `i` uses seed `--seed + i` and writes `0000.mp4`, `0001.mp4`, etc.

### Project-page videos and figures

```bash
uv run python scripts/generate_project_page.py --model ff-s --output-dir outputs/project_page/ff-s
```

Use `--model ff-l` for FastFrames-L. The script selects the project-page prompts
and model-specific noise convention. For Figs. 3–4, generate all 109
[demo prompts](installation.md#training-inputs), with stills at 0 and 5 seconds:

```bash
Q="uv run python scripts/generate_project_page.py --indices all --stills 0,5"
$Q --model ff-s --context dit --decoder vae --output-dir outputs/figures/cfpp
$Q --model ff-s --context dit --decoder pixel --output-dir outputs/figures/pixel
$Q --model ff-s --output-dir outputs/figures/ff-s
$Q --model ff-l --output-dir outputs/figures/ff-l
```

The first two commands compare decoders on the same latents. Zero-based prompt
index `i` uses seed `1000003 * i`; outputs include videos, stills and records of
the prompt, seed and latent hash.

## Evaluation

Complete the [VBench environment and weight setup](installation.md#vbench), which
sets `VBENCH_PYTHON` and `VBENCH_CACHE_DIR`. Generation uses the FastFrames
environment; scoring uses its separate pinned environment.

### Protocol

- The 946 prompt rows contain 944 unique prompts. Extended prompts pair by row
  with originals; the last occurrence supplies duplicate prompts' extensions.
  FastFrames, Self-Forcing and CF++ use Self-Forcing extended prompts; CausVid uses
  VBench's GPT-enhanced `all_dimension_longer.txt`.
- Generate five samples per prompt, except the 75 temporal-flickering prompts,
  which get 25 candidates. RAFT visits these in numeric order and selects the
  first five static clips. This yields 6,220 raw videos and up to 375 filtered clips.
- Sample `k` of prompt `i`, in `VBench_full_info.json` order, uses seed `25 * i + k`.
  Save as `<original_prompt>-<sample_index>.mp4`: 832 × 480, 81 frames, 16 fps,
  H.264 with the MP4 index first.
- Score temporal flickering on the filtered clips and the other 15 dimensions on
  raw videos in `vbench_standard` mode, using eight B200 GPUs and metric RNG seed 0.

Prompt files, pairings and seeds are pinned by hash. Inspect them with
`uv run python evaluation/vbench/protocol.py summary`.

### Generate and score

```bash
RUN=outputs/vbench/ff-s
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i uv run python evaluation/vbench/generate_corpus.py --model ff-s \
      --checkpoint-dir checkpoints --output-dir $RUN/raw_videos --shard $i --num-shards 8 &
done; wait
GPUS="0 1 2 3 4 5 6 7" uv run bash evaluation/vbench/score.sh $RUN ff-s
```

Use `ff-l` for FastFrames-L. After [baseline setup](installation.md#baselines),
generate and score CF++ with:

```bash
RUN=outputs/vbench/cfpp
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i uv run --extra baselines python evaluation/vbench/generate_baselines.py --model cfpp \
      --upstream upstream/Causal-Forcing --checkpoint checkpoints/causal-forcing++/framewise-2step.pt \
      --output-dir $RUN/raw_videos --shard $i --num-shards 8 &
done; wait
GPUS="0 1 2 3 4 5 6 7" uv run bash evaluation/vbench/score.sh $RUN cfpp
```

| Baseline | `--upstream` | `--checkpoint` | Denoising steps / latent frames per chunk |
|---|---|---|---|
| `self-forcing` | `upstream/Self-Forcing` | `checkpoints/self-forcing/checkpoints/self_forcing_dmd.pt` | 4 / 3 |
| `cfpp` | `upstream/Causal-Forcing` | `checkpoints/causal-forcing++/framewise-2step.pt` | 2 (4 on the first frame) / 1 |
| `causvid` | `upstream/CausVid` | `checkpoints/causvid/autoregressive_checkpoint/model.pt` | 3 / 3 |

Baselines generate one video per call, without classifier-free guidance. To use
another generator, run `uv run python evaluation/vbench/protocol.py jobs --output jobs.jsonl`;
condition on each row's `conditioning_prompt`, use its `seed`, and save as `filename`.
`protocol.write_faststart_video` writes the required format with OpenH264.

Generation reruns skip valid existing videos. Remove leftover `.vbench-*.mp4`
temporary files from interrupted writes before scoring. `score.sh` verifies,
filters and scores the corpus, writing `score_summary.json`, per-dimension results
in `vbench_results/`, media checks, filter provenance and `logs/` under `$RUN`.
Restart the same scoring command to reuse completed work. After changing raw videos,
delete `temporal_filter_contract.json`, `filtered_temporal/` and `vbench_results/`
before scoring again. Recompute aggregates with
`uv run python evaluation/vbench/summarize.py --from-summary "$RUN/score_summary.json"`.

### Fixed benchmark guard

On a single CUDA GPU, save a lossless reference at the baseline commit, then run
the same command without `--save-reference` after each change:

```bash
python evaluation/bench/fixed_bench.py --model ff-s --save-reference
python evaluation/bench/fixed_bench.py --model ff-s
python evaluation/bench/fixed_bench.py --model ff-s --profile
python evaluation/bench/fixed_bench.py --model ff-s --warmup 2 --repeats 10 --profile --profile-repeats 3
python evaluation/bench/compare.py
# Alternative workload (upstream example 00 by default):
python evaluation/bench/fixed_bench.py --model lingbot-1.3b --lingbot-root "$LINGBOT_WORLD_ROOT" --save-reference
python evaluation/bench/fixed_bench.py --model lingbot-1.3b --lingbot-root "$LINGBOT_WORLD_ROOT" --stage-timing
```

FastFrames-S uses 832x480, 81 frames, seed 0 and a fixed corgi prompt. LingBot uses
its example prompt (override with `--prompt`) and rounds `--frames 81` down to
whole chunks: normally 77 frames at 832x464. Actual shapes are recorded and checked.
Use identical model weights, software, GPU and settings when comparing results.
This is an explicit per-change command, not an automatically installed CI job.
No GPU generation was validated on the CPU-only development VM; 16GB feasibility
depends on model/runtime memory and is not guaranteed (especially LingBot).

Two untimed warm-ups (`--warmup`, nonnegative) precede ten synchronized end-to-end
generations (`--repeats`, positive), including CPU output transfer. Reports include
all samples, median/mean/sample-stdev/min/max/CV for fps and ms per frame, and peak
allocated/reserved **decimal GB**. History is appended to `outputs/bench/history.jsonl`;
full JSON records and optional trace/profile reports are under `outputs/bench/runs/`.
Use `--history`, `--run-dir`, and `--reference-dir` to relocate artifacts.
References are uint8 `.pt` tensors with a viewing-only MP4; keys include model,
resolved prompt, seed, requested frames and example path. Do not overwrite a
baseline reference on candidate commits. Reference creation refuses invalid video.

Every frame must meet `--min-psnr 30`, `--min-ssim 0.90`, and, when the optional
`lpips` package is installed, `--max-lpips 0.10`. LPIPS may download AlexNet weights
on first use. Metrics report mean/min/max and per-frame values; infinite PSNR is
serialized as the string `"inf"` for standards-compliant JSON. Constant/non-finite
frames, shape changes and mean luma outside `[0.01,0.99]` fail independently of a
reference (tune `--constant-eps`, `--min-luma`, `--max-luma` if necessary). Failures
exit 1; a missing reference warns and exits 0 unless sanity/performance gates fail.
Optionally use `--max-fps-drop 0.10` to fail a >10% median fps drop against the
previous same-key history entry. The baseline must be run on the same GPU. With the
default 3 repeats the median is fragile, so when this run's or the previous entry's fps
CV exceeds `--max-fps-cv 0.02` the gate is skipped and the run is printed and recorded
as LOW-CONFIDENCE (`fps_gate.status`); rerun the condition before trusting the fps.
LingBot keys also include `timing_scope` ("steps only (start untimed)"): condition
encoding in `start()` is outside the timed and peak-VRAM windows.

Every timed repeat is checked after its timer stops, then discarded before the
next generation (only the last video is retained for optional reference saving;
the previous one is kept just long enough to record `max_abs_diff_vs_previous`, and
the run's largest value as `repeat_max_abs_diff`, to expose run-to-run nondeterminism).
Any failed repeat fails the run, with 1-based repeat numbers in failure reasons
and per-repeat metrics/verdicts under `repeat_quality` in JSON. The legacy top-level
`quality` table columns describe the last repeat, not the whole run. Existing
references are compared on every repeat even with `--save-reference`; replacement
is refused if any repeat fails. Quality checks are outside generation wall timing
but add CPU work between trials. Empty output fails quality and reports zero
fps/ms-per-frame for that invalid sample rather than dividing by zero.

`--profile` runs three extra generations in separate profiler sessions
(`--profile-repeats N`), exporting `trace_00.json`, etc. `profile.md` aggregates
kernel calls and mean±sample-stdev ms/run (absent kernels count as zero), category
mean±stdev percentages, union-of-GPU-intervals/wall busy ratio and idle variance.
The JSON record links every trace and records profiler wall-time samples.
CV means sample stdev / absolute mean (zero for all-zero samples); singleton
stdev is reported as zero but does not establish repeatability. Old history
entries without new statistics display `-` for missing fields.

Hints cite their measured mean, stdev, CV and run count. Thresholds remain
attention >40% of GPU time, idle >15% of wall time, copies >5% of GPU time, or over
half of kernels <10 microseconds. A hint is **low-confidence** if its measurement
has CV >5% or only one run. Missing GPU events suppress hints rather than imply
CPU bottlenecks. Categories are name-based classifications of measured kernel
durations, not estimated performance; recommendations are hypotheses, not measured
speedups. Profiler overhead can exaggerate idle time.

`--stage-timing` (LingBot only) runs one additional diagnostic generation after
timing/profiling, with CUDA-event proxies around `pipe.dit`, `pipe.refiner` and
`pipe.decoder` call objects. It prints and records median **summed ms/chunk** and
all chunk samples. DiT includes its first-chunk clean pass; the refiner has no
chunk-0 sample. Events bracket the outer, already-warmed call/graph replay, not
captured graph bodies. Current single-GPU implementations enqueue replay on the
current stream. Compilation/capture should already be warmed by the timed runs.
No application code is patched and proxies are restored even on failure.
These measurements exclude camera rays, `dit.begin_chunk`, encoding in `start`,
scheduler work and output conversion; they are not a complete stage decomposition
or a repeatability claim. Event instrumentation runs separately to avoid changing
benchmark fps/VRAM. CUDA-event timing is implemented but **not GPU-validated on this
CPU-only VM**; graph replay/device behavior must be validated on the target GPU.

### Results (Table 1)

| Model | Total | Quality | Semantic |
|---|---:|---:|---:|
| CausVid | 83.18 | 84.15 | 79.31 |
| Self-Forcing | 84.10 | 84.84 | 81.13 |
| Causal Forcing++ | 84.11 | 85.09 | 80.20 |
| FastFrames-S | 84.23 | 85.30 | 79.92 |
| FastFrames-L | 84.44 | 85.48 | 80.29 |

[Raw dimension scores](../evaluation/vbench/tests/data/table1_raw_scores.json) back these rows.
Keep the scorer, metric weights, dependencies and GPU count fixed when comparing runs.

### Scorer provenance

Scoring uses `vbench==0.1.5` with
[upstream updates](../evaluation/vbench/patches/upstream_post_0.1.5.diff) followed by
[local changes](../evaluation/vbench/patches/vbench_package.diff).
Score-affecting changes disable MUSIQ/ViCLIP dropout and average appearance-style
frame scores across GPUs. [Script patches](../evaluation/vbench/patches/scorer_scripts.diff)
document the evaluator and static filter changes. Other patches handle validation,
distributed output and numeric static-filter ordering.

[scorer_manifest.json](../evaluation/vbench/scorer_manifest.json) records file and
tree hashes; filtering and scoring verify them. VBench is
[Apache-2.0](../evaluation/vbench/VBench/LICENSE). Verify the built tree or run CPU checks:

```bash
uv run python evaluation/vbench/setup_scorer.py --verify
uv run --extra dev python -m pytest evaluation/vbench/tests
```
