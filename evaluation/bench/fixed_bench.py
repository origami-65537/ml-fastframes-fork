#!/usr/bin/env python3
"""Fixed single-GPU benchmark and lossless reference quality guard."""
import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.bench.compare import append_history, read_history, table  # noqa: E402
from evaluation.bench.profiling import profile_report  # noqa: E402
from evaluation.bench.quality import compare, sanity, verdict  # noqa: E402
from evaluation.bench.stats import sample_stats  # noqa: E402


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["ff-s", "lingbot-1.3b"], default="ff-s")
    p.add_argument("--prompt", help="default: corgi prompt for ff-s; example prompt for LingBot")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frames", type=int, default=81, help="ff-s requires 81; LingBot rounds to whole chunks")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--lingbot-root")
    p.add_argument("--lingbot-checkpoint", type=Path)
    p.add_argument("--assets-dir", type=Path)
    p.add_argument("--eager", action="store_true")
    p.add_argument("--example", type=Path, help="default: $LINGBOT_WORLD_ROOT/examples/00")
    p.add_argument("--save-reference", action="store_true")
    p.add_argument("--reference-dir", type=Path, default=Path("outputs/bench/reference"))
    p.add_argument("--history", type=Path, default=Path("outputs/bench/history.jsonl"))
    p.add_argument("--run-dir", type=Path, default=Path("outputs/bench/runs"))
    p.add_argument("--min-psnr", type=float, default=30)
    p.add_argument("--min-ssim", type=float, default=0.90)
    p.add_argument("--max-lpips", type=float, default=0.10)
    p.add_argument("--min-luma", type=float, default=0.01)
    p.add_argument("--max-luma", type=float, default=0.99)
    p.add_argument("--constant-eps", type=float, default=1e-6)
    p.add_argument("--max-fps-drop", type=float, help="allowed fractional drop vs last same-key run, e.g. 0.1")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--profile-repeats", type=int, default=3)
    p.add_argument("--stage-timing", action="store_true", help="extra LingBot CUDA-event diagnostic generation")
    return p


def workload_key(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]


def adapter(args):
    import torch

    if args.model == "ff-s":
        from scripts.generate import load_model
        args.prompt = args.prompt if args.prompt is not None else "A corgi surfing a wave at sunset"
        model = load_model("ff-s", args.checkpoint_dir)
        return lambda: model.generate(args.prompt, args.seed), lambda: None, (81, 480, 832, 3), None
    from apps.lingbot.generate import CHUNK, add_model_arguments, import_upstream, load_pipeline, read_condition

    model_parser = argparse.ArgumentParser()
    add_model_arguments(model_parser)
    model_args = model_parser.parse_args(["--model", args.model])
    for name in vars(model_args):
        setattr(model_args, name, getattr(args, name))

    root = args.lingbot_root or os.environ.get("LINGBOT_WORLD_ROOT")
    if args.example is None:
        if not root:
            raise ValueError("LingBot needs --lingbot-root or $LINGBOT_WORLD_ROOT for example 00")
        args.example = Path(root) / "examples/00"
    import_upstream(root)
    image, args.prompt, rays, chunks, seed = read_condition(args)
    pipe = load_pipeline(model_args, 1, 0)
    chunk_index = [0]

    def generate():
        pipe.start(image, args.prompt, seed=seed, latents=CHUNK * chunks)
        frames = []
        for chunk in range(chunks):
            chunk_index[0] = chunk
            frames.append(pipe.step(rays(chunk)))
        return torch.cat(frames).cpu()

    def stage_timing():
        from evaluation.bench.stages import measure_stages
        return measure_stages(pipe, generate, lambda: chunk_index[0])

    return generate, pipe.close, (CHUNK * chunks * 4 - 3, 464, 832, 3), stage_timing


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.repeats < 1 or args.frames < 13 or (args.model == "ff-s" and args.frames != 81):
        p.error("repeats must be positive; ff-s requires 81 frames; LingBot needs at least 13")
    if args.max_fps_drop is not None and not 0 <= args.max_fps_drop <= 1:
        p.error("--max-fps-drop must be in [0,1]")
    if args.warmup < 0 or args.profile_repeats < 1:
        p.error("--warmup must be nonnegative and --profile-repeats must be positive")
    if args.stage_timing and args.model != "lingbot-1.3b":
        p.error("--stage-timing is only supported for lingbot-1.3b")
    import torch

    if not torch.cuda.is_available():
        p.error("a CUDA GPU is required for generation (metrics/tests work on CPU)")
    torch.cuda.set_device(0)
    generate, close, expected, stage_timing = adapter(args)
    config = {k: getattr(args, k) for k in ("model", "prompt", "seed", "frames")}
    config["example"] = str(args.example.resolve()) if args.example else None
    key = workload_key(config)
    root = Path(__file__).resolve().parents[2]

    def git(*cmd):
        return subprocess.check_output(["git", "-C", str(root), *cmd], text=True).strip()

    sha = git("rev-parse", "HEAD")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run = args.run_dir / f"{timestamp}_{sha[:10]}"
    run.mkdir(parents=True, exist_ok=True)
    try:
        with torch.inference_mode():
            for _ in range(args.warmup):
                warmup = generate()
                torch.cuda.synchronize()
                del warmup
            torch.cuda.reset_peak_memory_stats()
            seconds, counts = [], []
            video = None
            for _ in range(args.repeats):
                del video
                torch.cuda.synchronize()
                start = time.perf_counter()
                video = generate()
                torch.cuda.synchronize()
                seconds.append(time.perf_counter() - start)
                counts.append(len(video))
            memory = {"allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                      "reserved_gb": torch.cuda.max_memory_reserved() / 1e9}
            video = torch.as_tensor(video).cpu()
            fps = [n / s for n, s in zip(counts, seconds)]
            reasons = sanity(video, expected, args.constant_eps, args.min_luma, args.max_luma)
            refpath = args.reference_dir / f"{key}.pt"
            metrics = {}
            has_reference = refpath.exists()
            if args.save_reference:
                if not reasons:
                    from fastframes.video import save_video
                    refpath.parent.mkdir(parents=True, exist_ok=True)
                    save_video(refpath.with_suffix(".mp4"), video)
                    torch.save(video, refpath)
                    has_reference = True
            elif has_reference:
                reference = torch.load(refpath, map_location="cpu", weights_only=True)
                if reference.shape != video.shape:
                    reasons.append(f"reference shape mismatch: {tuple(reference.shape)} != {tuple(video.shape)}")
                else:
                    metrics = compare(video, reference)
            else:
                print(f"WARNING: no reference at {refpath}; use --save-reference at the baseline commit")
            previous = next((r for r in reversed(read_history(args.history)) if r["key"] == key), None)
            status, reasons = verdict(metrics, reasons, has_reference, args.min_psnr, args.min_ssim,
                                      args.max_lpips, statistics.median(fps),
                                      previous["fps"]["median"] if previous else None, args.max_fps_drop)
            record = {"key": key, "config": config, "timestamp": timestamp,
                      "git": {"sha": sha, "dirty": bool(git("status", "--porcelain"))},
                      "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "args": vars(args),
                      "actual_shape": list(video.shape), "expected_shape": expected,
                      "note": "LingBot uses whole chunks at 832x464, not 832x480" if args.model != "ff-s" else "",
                      "fps": sample_stats(fps), "seconds": seconds,
                      "ms_per_frame": sample_stats([1000/f for f in fps]),
                      "memory": memory, "quality": metrics, "verdict": status, "reasons": reasons,
                      "reference": str(refpath), "reference_saved": args.save_reference and not reasons}
            if args.profile:
                traces, walls = [], []
                for index in range(args.profile_repeats):
                    torch.cuda.synchronize()
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
                        start = time.perf_counter()
                        profiled = generate()
                        torch.cuda.synchronize()
                        walls.append(time.perf_counter() - start)
                        del profiled
                    trace = run / f"trace_{index:02d}.json"
                    prof.export_chrome_trace(str(trace))
                    traces.append(trace)
                (run / "profile.md").write_text(profile_report(traces, walls))
                record["profile"] = str(run / "profile.md")
                record["profile_samples"] = {"traces": traces, "wall_seconds": walls}
            if args.stage_timing:
                record["stage_timing"] = stage_timing()
            record = json_safe(record)
            Path(str(run) + ".json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
            append_history(args.history, record)
            print(table([record]))
            print(f"Actual video: {list(video.shape)}. {record['note']}")
            if args.stage_timing:
                for name, stats in record["stage_timing"]["stages"].items():
                    print(f"{name}: {stats['median']:.3f} median ms/chunk ({len(stats['samples'])} chunks)")
            for reason in reasons:
                print(f"FAIL: {reason}")
            return int(status == "fail")
    finally:
        close()


if __name__ == "__main__":
    sys.exit(main())
