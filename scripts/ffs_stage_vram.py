#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""FF-S on a small card: reports peak VRAM per stage.

    # run from the repo root; paths are relative to it
    python scripts/ffs_stage_vram.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from fastframes.pipeline import MODELS, FastFrames, configure_backends, to_uint8
from fastframes.video import save_video

spec = MODELS["ff-s"]
root = Path("checkpoints")
configure_backends(spec.config)


def vram(tag):
    print(f"{tag:24s} alloc={torch.cuda.memory_allocated()/1e9:6.2f}GB "
          f"peak={torch.cuda.max_memory_allocated()/1e9:6.2f}GB "
          f"reserved={torch.cuda.memory_reserved()/1e9:6.2f}GB", flush=True)


t0 = time.perf_counter()
model = FastFrames.from_pretrained(
    spec.config, root / spec.wan_dir, root / spec.dit, root / spec.refiner, root / spec.decoder)
print(f"load done {time.perf_counter()-t0:.1f}s")
vram("after load")

prompt = "A corgi surfing a wave at sunset"
t0 = time.perf_counter()
embeds = model.encode_prompt(prompt)
print(f"encode done {time.perf_counter()-t0:.1f}s")
vram("after encode+offload")

t0 = time.perf_counter()
latents, features, modulation = model.rollout(embeds, seed=0)
print(f"rollout done {time.perf_counter()-t0:.1f}s")
vram("after rollout")

t0 = time.perf_counter()
pixels = model.decode(features, modulation)
frames = to_uint8(pixels[0]).cpu()
print(f"decode done {time.perf_counter()-t0:.1f}s")
vram("after decode")

save_video("/tmp/ffs_16gb.mp4", frames)
print("saved /tmp/ffs_16gb.mp4")
