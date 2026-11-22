#!/usr/bin/env python3
"""start -> step -> start -> step (no restart()), latency + memory per boundary + DiT capture memory (PR 6)."""
import os
import sys
import time
import traceback
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd()))
from pr6 import hooks  # noqa: E402

hooks.install()
from apps.lingbot.generate import CHUNK, add_model_arguments, import_upstream, load_pipeline, read_condition  # noqa: E402

p = argparse.ArgumentParser()
add_model_arguments(p)
args = p.parse_args(["--model", "lingbot-1.3b"])
root = os.environ["LINGBOT_WORLD_ROOT"]
args.lingbot_root, args.example, args.frames, args.seed, args.prompt, args.eager = root, Path(root) / "examples/00", 29, 0, None, True
import_upstream(root)
image, prompt, rays, chunks, seed = read_condition(args)
pipe = load_pipeline(args, 1, 0)
hooks.mem("after load")
outs, attempt, op = [], 0, "init"
try:
    with torch.inference_mode():
        for attempt in range(1, 3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            op = f"start#{attempt}"
            pipe.start(image, prompt, seed=seed, latents=CHUNK * chunks)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            hooks.mem(f"after start#{attempt}")
            op = f"step#{attempt}"
            frame = pipe.step(rays(0))
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            hooks.mem(f"after step#{attempt}")
            outs.append(frame.float().cpu())
            print(f"LATENCY start#{attempt}: start={1000*(t1-t0):.1f} ms first_step={1000*(t2-t1):.1f} ms "
                  f"start->first_frame={1000*(t2-t0):.1f} ms", flush=True)
        d = (outs[0] - outs[1]).abs().max().item()
        print(f"SAME_INPUT max|diff| first chunk start#1 vs start#2: {d:.6f} bit_identical={d == 0}")
        print(f"PEAK alloc/res GiB: {torch.cuda.max_memory_allocated()/2**30:.3f}/{torch.cuda.max_memory_reserved()/2**30:.3f}")
        print("SMOKE_OK")
except BaseException:
    print(f"SMOKE_FAILED during {op} (completed steps: {len(outs)})", flush=True)
    traceback.print_exc(file=sys.stdout)
    hooks.mem("at failure")
    print(torch.cuda.memory_summary(abbreviated=True), flush=True)
    sys.exit(1)
finally:
    pipe.close()
