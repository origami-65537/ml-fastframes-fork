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
dit / attn / refiner / decoder via monkeypatched forwards, which are restored
once the chunks are done.
"""
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])  # repository root, not the caller's directory
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import apps.lingbot.dit as dit_mod  # noqa: E402
from apps.lingbot.conditioning import CHUNK, Trajectory  # noqa: E402
from apps.lingbot.dit import LingBotDiT  # noqa: E402
from apps.lingbot.pipeline import FastFramesLingBot, configure_backends  # noqa: E402
from apps.lingbot.refiner import KVRefiner  # noqa: E402
from apps.lingbot.upstream import import_upstream  # noqa: E402

CHUNKS = 8  # 32 latent frames
TEMPORAL_UPSAMPLE = 4  # pixel frames decoded per latent frame
PIXEL_FRAMES_PER_CHUNK = CHUNK * TEMPORAL_UPSAMPLE

_MISSING = object()


def vram(tag):
    torch.cuda.synchronize()
    a = torch.cuda.memory_allocated() / 2**30
    r = torch.cuda.memory_reserved() / 2**30
    p = torch.cuda.max_memory_allocated() / 2**30
    print(f"[{tag}] alloc={a:.2f}G reserved={r:.2f}G peak={p:.2f}G", flush=True)


def _timed(fn, stats, key):
    def wrap(*a, **k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = fn(*a, **k)
        torch.cuda.synchronize()
        stats[key] += time.perf_counter() - t
        return r

    return wrap


@contextmanager
def stage_timers(stats, classes, time_attn=False):
    """Accumulate synchronized wall time per stage into ``stats`` while active.

    ``classes`` maps a stage key to the class whose ``forward``/``__call__`` is timed;
    ``time_attn`` also times ``apps.lingbot.dit.attention``. Everything patched is
    put back on exit.
    """
    saved = []

    def replace(owner, name, value):
        saved.append((owner, name, owner.__dict__.get(name, _MISSING)))
        setattr(owner, name, value)

    try:
        if time_attn:
            replace(dit_mod, "attention", _timed(dit_mod.attention, stats, "attn"))
        for key, cls in classes.items():
            orig = cls.forward if hasattr(cls, "forward") else cls.__call__
            wrap = _timed(orig, stats, key)
            replace(cls, "forward", wrap)
            if cls.__call__ is not object.__call__:
                replace(cls, "__call__", wrap)
        yield stats
    finally:
        for owner, name, value in reversed(saved):
            if value is _MISSING:
                delattr(owner, name)
            else:
                setattr(owner, name, value)


def main():
    from apps.lingbot.generate import OFFICIAL, SHARED_ASSETS

    torch.cuda.set_device(0)
    import_upstream("upstream/lingbot-world-v2")

    base = Path("upstream/lingbot-world-v2/examples/00")
    image = Image.open(base / "image.jpg").convert("RGB")
    prompt = (base / "prompt.txt").read_text().strip()
    poses = np.load(base / "poses.npy")
    intr = np.load(base / "intrinsics.npy")
    traj = Trajectory(poses, intr, CHUNKS * PIXEL_FRAMES_PER_CHUNK, device="cuda")

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
    stages = {"dit": LingBotDiT, "refiner": KVRefiner, "decoder": type(pipe.decoder)}
    lat = []
    with stage_timers(stats, stages, time_attn=os.environ.get("LB_TIME_ATTN") == "1"):
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
    median = sorted(lat)[len(lat) // 2]
    print(f"median chunk: {median*1000:.0f}ms  "
          f"-> {PIXEL_FRAMES_PER_CHUNK/median:.1f} px frames/s")

    pipe.close()


if __name__ == "__main__":
    main()
