#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Effective-TFLOPS check for LingBot DiT attention shapes vs. the GEMM peak.

Standalone microbenchmark: times cuDNN/efficient SDPA at the LingBot 1.3B
shape (batch=1, heads=12, head_dim=128, Q=6032, K=15080 for LB_CACHE_FRAMES=10)
and plain cuBLAS GEMMs, reporting achieved TFLOPS. Use it to verify whether
attention is already at the bf16 dense roofline of the card.

    python scripts/sdpa_roofline.py
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda"
H, HD = 12, 128
Q_LEN, K_LEN = 6032, 15080  # CF=10


def bench(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def report(name, ms, flops):
    print(f"{name:40s} {ms:8.3f} ms   {flops / ms / 1e9:8.2f} TFLOPS")


# ---- attention shapes (same as LingBot DiT: batch=1, H=12, hd=128)
q = torch.randn(1, H, Q_LEN, HD, device=dev, dtype=torch.bfloat16)
k = torch.randn(1, H, K_LEN, HD, device=dev, dtype=torch.bfloat16)
v = torch.randn(1, H, K_LEN, HD, device=dev, dtype=torch.bfloat16)
attn_flops = 4 * H * Q_LEN * K_LEN * HD  # QK^T + AV

for name, backend in [("cuDNN", SDPBackend.CUDNN_ATTENTION),
                      ("efficient", SDPBackend.EFFICIENT_ATTENTION)]:
    try:
        def run(b=backend):
            with sdpa_kernel(b):
                return F.scaled_dot_product_attention(q, k, v)
        ms = bench(run)
        report(f"SDPA {name} (model shape)", ms, attn_flops)
    except Exception as e:
        print(f"SDPA {name}: FAILED {e}")

# causal variant
try:
    ms = bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
    report("SDPA causal=True", ms, attn_flops / 2)
except Exception as e:
    print(f"SDPA causal: FAILED {e}")

# ---- pure GEMM dense peak
for m, n, kd in [(8192, 8192, 8192), (16384, 16384, 8192), (6032, 1536, 1536),
                 (6032, 8960, 1536), (6032, 1536, 8960)]:
    a = torch.randn(m, kd, device=dev, dtype=torch.bfloat16)
    b = torch.randn(kd, n, device=dev, dtype=torch.bfloat16)
    ms = bench(lambda a=a, b=b: a @ b)
    report(f"GEMM {m}x{n}x{kd}", ms, 2 * m * n * kd)
    del a, b
    torch.cuda.empty_cache()

# GEMM batched like attention QK^T
qb = torch.randn(1, H, Q_LEN, HD, device=dev, dtype=torch.bfloat16)
kb = torch.randn(1, H, HD, K_LEN, device=dev, dtype=torch.bfloat16)
ms = bench(lambda: qb @ kb)
report("batched QK^T only", ms, 2 * H * Q_LEN * K_LEN * HD)

print("done")
