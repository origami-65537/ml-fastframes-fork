#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Stage-by-stage VRAM and latency profile for FastFrames-LingBot on one GPU.

    # run from the repo root; paths are relative to it
    python scripts/profile_lingbot.py

Environment (same levers as the LingBot pipeline):

    LB_COMPILE=1        compile DiT/refiner/decoder (default off here)
    LB_GRAPHS=1         enable CUDA graphs
    LB_TIME_ATTN=1      also time the attention() calls inside DiT
    LB_CACHE_FRAMES, LB_STEPS, LB_COMPILE_DIT, ... all pipeline envs apply

Prints VRAM after load and start(), then per-chunk wall time split into
dit / attn / refiner / decoder via monkeypatched forwards.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

import apps.lingbot.dit as dit_mod
from apps.lingbot.conditioning import CHUNK, Trajectory
from apps.lingbot.dit import LingBotDiT
from apps.lingbot.generate import OFFICIAL, SHARED_ASSETS
from apps.lingbot.pipeline import FastFramesLingBot, configure_backends
from apps.lingbot.refiner import KVRefiner
from apps.lingbot.upstream import import_upstream

torch.cuda.set_device(0)
import_upstream("upstream/lingbot-world-v2")


def vram(tag):
    torch.cuda.synchronize()
    a = torch.cuda.memory_allocated() / 2**30
    r = torch.cuda.memory_reserved() / 2**30
    p = torch.cuda.max_memory_allocated() / 2**30
    print(f"[{tag}] alloc={a:.2f}G reserved={r:.2f}G peak={p:.2f}G", flush=True)


base = Path("upstream/lingbot-world-v2/examples/00")
image = Image.open(base / "image.jpg").convert("RGB")
prompt = (base / "prompt.txt").read_text().strip()
poses = np.load(base / "poses.npy")
intr = np.load(base / "intrinsics.npy")
CHUNKS = 8  # 32 latent frames
traj = Trajectory(poses, intr, CHUNKS * 16, device="cuda")

configure_backends()
t0 = time.perf_counter()
pipe = FastFramesLingBot.from_pretrained(
    "lingbot-1.3b", lingbot_root="upstream/lingbot-world-v2",
    lingbot_checkpoint=Path("checkpoints") / OFFICIAL["lingbot-1.3b"],
    assets_dir=Path("checkpoints") / SHARED_ASSETS,
    checkpoint_dir=Path("checkpoints") / "lingbot-1.3b",
    world=1, rank=0, compile=(os.environ.get("LB_COMPILE", "0") == "1"),
    graphs=(os.environ.get("LB_GRAPHS", "0") == "1"))
print(f"load: {time.perf_counter()-t0:.1f}s")
vram("loaded")

t0 = time.perf_counter()
pipe.start(image, prompt, seed=0, latents=CHUNK * CHUNKS)
print(f"start(cond): {time.perf_counter()-t0:.1f}s")
vram("after start")

stats = {"dit": 0.0, "refiner": 0.0, "decoder": 0.0, "attn": 0.0}

if os.environ.get("LB_TIME_ATTN") == "1":
    _orig_attn = dit_mod.attention

    def attn_wrap(*a, **k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = _orig_attn(*a, **k)
        torch.cuda.synchronize()
        stats["attn"] += time.perf_counter() - t
        return r

    dit_mod.attention = attn_wrap


def patch(cls, key):
    orig = cls.forward if hasattr(cls, "forward") else cls.__call__

    def wrap(self, *a, **k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = orig(self, *a, **k)
        torch.cuda.synchronize()
        stats[key] += time.perf_counter() - t
        return r

    cls.forward = wrap
    if cls.__call__ is not object.__call__:
        cls.__call__ = wrap


patch(LingBotDiT, "dit")
patch(KVRefiner, "refiner")
patch(type(pipe.decoder), "decoder")

lat = []
for c in range(CHUNKS):
    t0 = time.perf_counter()
    pipe.step(traj.rays(c))
    torch.cuda.synchronize()
    lat.append(time.perf_counter() - t0)
    print(f"chunk {c}: {lat[-1]*1000:.0f}ms  dit={stats['dit']*1000:.0f} "
          f"attn={stats['attn']*1000:.0f} ref={stats['refiner']*1000:.0f} "
          f"dec={stats['decoder']*1000:.0f}", flush=True)
    stats.update({k: 0.0 for k in stats})
vram("after chunks")
print(f"median chunk: {sorted(lat)[len(lat)//2]*1000:.0f}ms  "
      f"-> {4/sorted(lat)[len(lat)//2]*4:.1f} px frames/s")

pipe.close()
