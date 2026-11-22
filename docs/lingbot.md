# FastFrames-LingBot

FastFrames accelerates LingBot-World v2 1.3B with KV refinement, a streaming pixel decoder,
and optimized inference. LingBot-World v2 code and weights follow their upstream
[CC BY-NC-SA 4.0 license](../LICENSE_LINGBOT); FastFrames code uses the
[Apple Sample Code License](../LICENSE).
Run commands from the repository root.

The default checkpoint directory is `checkpoints/`. Override locations with
`--checkpoint-dir`, `--lingbot-root`, `--lingbot-checkpoint` or `--assets-dir`.
Eight-GPU runs require one node with NVSwitch (NVLink multicast) and the default CUDA
allocator: unset `PYTORCH_ALLOC_CONF` and `PYTORCH_CUDA_ALLOC_CONF` if either enables
`expandable_segments`.

On one supported NVIDIA GPU, self-attention defaults to SageAttention when
`sageattention==1.0.6` is installed (`uv pip install sageattention==1.0.6`); otherwise
it uses cuDNN. Set `LB_ATTN=cudnn` to force cuDNN or `LB_ATTN=sage` to require SageAttention.
The one-GPU loader enables expandable CUDA allocator segments by default when no PyTorch
allocator configuration is set. Set `LB_EXPANDABLE_SEGMENTS=0` to disable that default.

On one GPU, the refiner uses compilation and writes directly to the rolling KV cache
by default. Set `LB_COMPILE_REFINER=0` to disable refiner compilation (also off with
`--eager` unless explicitly enabled) or `LB_REFINER_DIRECT_KV=0` to use staged writes.
Multi-GPU runs keep staged writes.

Experimental settings (off unless specified):

| Setting | Effect |
|---|---|
| `LB_FP8=1`, `ffn`, or `attention` | Opt in to dynamically scaled FP8 for the selected DiT linears. Defaults to `0` (BF16). Cross-attention K/V stay BF16. |
| `LB_ATTN_CROSS=sage` | Use SageAttention for cross-attention on one GPU. |
| `LB_SAGE_PRECENTER=1` | Center SageAttention keys out of place instead of cloning and then centering. |
| `LB_DIT_CAMERA_SLICED=1` | With DiT compilation enabled, compile camera writes per layer so the compiler sees only one destination slice. |
| `LB_REFINER_FUSED_EPILOGUE=1` | Compile the KV repair, RoPE and cache-write epilogues but leave refiner GEMMs eager. Requires direct KV writes and `LB_COMPILE_REFINER=0`. |
| `LB_REFINER_HEADS_PER_STREAM=N` | Group N DiT layers per refiner CUDA stream; N must divide the layer count. |

On the RTX 5060 Ti 16 GB fixed benchmark (832 × 464, 77 output frames), all three FP8
settings failed the quality gate against the BF16/cuDNN reference. Side-by-side frames
showed trajectory drift around frames 15–25, rather than visible noise or color shifts.
Keep FP8 off when the fixed-benchmark quality gate must pass.

On the same GPU, DiT denoising graph warmup ran out of memory during the first step of the two-start smoke
test. Set `LB_GRAPHS_DIT=0` to run DiT without CUDA graphs while keeping DiT compilation
enabled. With that setting, both starts produced bit-identical first chunks even with
refiner and decoder graphs left on.

## Interactive demo (1.3B)

After [LingBot setup](installation.md#lingbot), launch one mode:

```bash
# One GPU
CUDA_VISIBLE_DEVICES=0 uv run --extra lingbot --extra demo python apps/lingbot/demo.py --gpus 1

# Eight GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --extra lingbot --extra demo python apps/lingbot/demo.py --gpus 8
```

The launcher manages its GPU workers; run it directly, without `torchrun`.
Open `http://localhost:8000`. For a remote GPU server, forward the port from your laptop:

```bash
ssh -N -L 8000:127.0.0.1:8000 user@gpu-server
```

`--port` changes the port. The server binds to localhost by default; `--host 0.0.0.0`
allows LAN access and should only be used on a trusted network, since there is no authentication.

Choose an example or upload a landscape image, edit the prompt and seed, and click
**Start world**. Use WASD to move and the left/right arrows to pan. Tap for one block,
or hold for 0.4 seconds to repeat. Images are centered and cropped to 832 × 464.

The first start compiles and warms the model, which takes several minutes. Subsequent
starts reuse the loaded model. Each world lasts 60 blocks (957 frames). One visitor
controls the world at a time; others can join a waiting room. Stop or disconnect releases
the turn. When someone is waiting, two minutes without input also releases it.
Stop the server with **Ctrl-C** to release GPU memory.

Observed peak device memory on B200 was 44.8 GiB for one GPU and 41.3 GiB per GPU in
eight-GPU mode. Allow headroom above these measurements. To measure your setup:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --extra lingbot --extra demo python apps/lingbot/demo.py --gpus 1 \
    --profile-vram outputs/vram-1gpu.json
```

The browser reports model generation latency separately from server, network and browser
overhead. Its 50-fps playback rate is independent of the model's 16-fps video timeline.

## Generation

Generate from an upstream example or a condition JSON file:

```bash
uv run --extra lingbot python apps/lingbot/generate.py --model lingbot-1.3b \
    --example "$LINGBOT_WORLD_ROOT/examples/00" --frames 961 --output outputs/lingbot_1.3b.mp4

uv run --extra lingbot torchrun --nproc_per_node 8 apps/lingbot/generate.py --model lingbot-1.3b \
    --example "$LINGBOT_WORLD_ROOT/examples/00" --output outputs/lingbot_1.3b_8gpu.mp4
```

Output is 832 × 464 at 16 fps. The first chunk contains 13 frames and subsequent chunks
contain 16; `--frames 961` selects 60 whole chunks, producing 957 frames.

For your own scene, replace `--example ...` with `--condition scene.json`:

```json
{"image": "street.jpg", "prompt": "A rainy street at night", "seed": 0,
 "poses": "poses.npy", "intrinsics": "intrinsics.npy"}
```

Paths are relative to the JSON file. `poses.npy` contains `[N, 4, 4]` OpenCV
camera-to-world matrices at 16 fps. `intrinsics.npy` contains `fx, fy, cx, cy` for
832 × 480 frames, as a `[4]` vector or `[N, 4]` array (the first row is used).
Use images close to 16:9. Camera-path translations are normalized, preserving direction.

Alternatively, replace `poses` and `intrinsics` with an `actions` list, one action per chunk:

```json
{"image": "street.jpg", "prompt": "A rainy street at night", "seed": 0,
 "actions": ["idle", "forward", "forward", "yaw_left", "forward"]}
```

Available actions: `forward`, `backward`, `left`, `right`, `up`, `down`, `yaw_left`,
`yaw_right`, `pitch_up`, `pitch_down`, `idle`. `--prompt` and `--seed` override the JSON.

## Latency (Fig. 5)

Run the one- and eight-GPU configurations on a B200 node, with three fresh processes per row:

```bash
PYTHON=python CHECKPOINTS=checkpoints \
    uv run --extra lingbot bash apps/lingbot/fig5.sh outputs/fig5
```

The script writes per-run JSON records and `outputs/fig5/fig5.json`. To select rows, append
`ff13_1` or `ff13_8` after the output directory.

Each measurement covers camera rays, cache updates, four denoising steps, KV refinement
and pixel decoding for one chunk. Chunks 0–7 are warmup; at least 30 chunks and 30 seconds
are timed. Results use the median chunk latency of the slowest rank, then the median
across three processes. Timing excludes host transfer and display.

| Model | 1 GPU (ms) | 8 GPUs (ms) |
|---|---:|---:|
| 1.3B | 220.7 | 50.5 |

These B200 measurements use PyTorch 2.9.1 / CUDA 12.8. Timings vary with hardware and
compiler autotuning. For individual runs and options, use
`uv run --extra lingbot python apps/lingbot/latency.py --help`.
