#!/usr/bin/env python3
"""Stage VRAM and latency on one GPU. Run with python -m apps.lingbot.profile_lb."""

import os
import time
from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path
from statistics import median

import numpy as np
import torch
from PIL import Image

from apps.lingbot import dit
from apps.lingbot.conditioning import CHUNK, Trajectory
from apps.lingbot.generate import OFFICIAL, SHARED_ASSETS
from apps.lingbot.pipeline import FastFramesLingBot, configure_backends
from apps.lingbot.upstream import import_upstream


def vram(tag):
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 2**30
    reserved = torch.cuda.memory_reserved() / 2**30
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"[{tag}] alloc={allocated:.2f}G reserved={reserved:.2f}G peak={peak:.2f}G", flush=True)


@contextmanager
def timed(target, name, stats, key):
    original = getattr(target, name)

    @wraps(original)
    def wrapped(*args, **kwargs):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = original(*args, **kwargs)
        torch.cuda.synchronize()
        stats[key] += time.perf_counter() - started
        return result

    setattr(target, name, wrapped)
    try:
        yield
    finally:
        setattr(target, name, original)


def main():
    torch.cuda.set_device(0)
    root = Path("upstream/lingbot-world-v2")
    import_upstream(root)
    example = root / "examples/00"
    with Image.open(example / "image.jpg") as source:
        image = source.convert("RGB")
    prompt = (example / "prompt.txt").read_text().strip()
    frames_per_chunk = 4 * CHUNK
    trajectory = Trajectory(np.load(example / "poses.npy"), np.load(example / "intrinsics.npy"),
                            8 * frames_per_chunk, device="cuda")

    configure_backends()
    started = time.perf_counter()
    pipe = FastFramesLingBot.from_pretrained(
        "lingbot-1.3b", lingbot_root=root,
        lingbot_checkpoint=Path("checkpoints") / OFFICIAL["lingbot-1.3b"],
        assets_dir=Path("checkpoints") / SHARED_ASSETS,
        checkpoint_dir=Path("checkpoints") / "lingbot-1.3b",
        compile=os.environ.get("LB_COMPILE") == "1", graphs=os.environ.get("LB_GRAPHS") == "1",
    )

    with ExitStack() as stack:
        stack.callback(pipe.close)
        print(f"load: {time.perf_counter() - started:.1f}s")
        vram("loaded")
        started = time.perf_counter()
        pipe.start(image, prompt, seed=0, latents=trajectory.latents)
        print(f"start(cond): {time.perf_counter() - started:.1f}s")
        vram("after start")

        stats = dict(dit=0.0, attn=0.0, refiner=0.0, decoder=0.0)
        if os.environ.get("LB_TIME_ATTN") == "1":
            stack.enter_context(timed(dit, "attention", stats, "attn"))
        for component in ("dit", "refiner", "decoder"):
            stack.enter_context(timed(type(getattr(pipe, component)), "__call__", stats, component))

        latencies = []
        for chunk in range(trajectory.num_chunks):
            started = time.perf_counter()
            pipe.step(trajectory.rays(chunk))
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - started)
            stages = " ".join(f"{key}={value * 1000:.0f}" for key, value in stats.items())
            print(f"chunk {chunk}: {latencies[-1] * 1000:.0f}ms  {stages}", flush=True)
            for key in stats:
                stats[key] = 0.0

        vram("after chunks")
        if len(latencies) > 1:
            typical = median(latencies[1:])
            print(f"median steady-state chunk: {typical * 1000:.0f}ms -> {frames_per_chunk / typical:.1f} fps")


if __name__ == "__main__":
    main()
