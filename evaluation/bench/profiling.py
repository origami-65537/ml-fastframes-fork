"""Chrome trace kernel analysis. Heuristics describe symptoms, not promised speedups."""
import json
import re
from collections import defaultdict

from evaluation.bench.stats import sample_stats

CATEGORIES = [
    ("attention", r"attention|flash|sdpa|fmha"),
    ("GEMM", r"gemm|matmul|mma|cutlass"),
    ("conv", r"conv|cudnn"),
    ("norm", r"norm"),
    ("memcpy", r"memcpy|memset"),
    ("elementwise/triton fused", r"elementwise|triton|vectorized|pointwise|fused"),
]


def classify(name):
    return next((category for category, pattern in CATEGORIES if re.search(pattern, name, re.I)), "other")


def trace_stats(trace_path, wall_seconds):
    events = json.loads(trace_path.read_text())["traceEvents"]
    kernels = [e for e in events if e.get("cat", "").lower() in
               ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
    totals, categories = defaultdict(lambda: [0, 0.0]), defaultdict(float)
    for e in kernels:
        totals[e["name"]][0] += 1
        totals[e["name"]][1] += e["dur"]
        categories[classify(e["name"])] += e["dur"]
    total = sum(v[1] for v in totals.values())
    busy, end = 0.0, float("-inf")
    for e in sorted(kernels, key=lambda e: e["ts"]):
        stop = e["ts"] + e["dur"]
        busy += max(0, stop - max(end, e["ts"]))
        end = max(end, stop)
    return {"kernels": dict(totals), "categories": dict(categories), "total_us": total,
            "busy": min(1, busy / (wall_seconds * 1e6)),
            "tiny": sum(e["dur"] < 10 for e in kernels) / len(kernels) if kernels else 0}


def profile_report(trace_paths, wall_seconds):
    if not trace_paths or len(trace_paths) != len(wall_seconds):
        raise ValueError("one wall time per nonempty trace list is required")
    runs = [trace_stats(path, wall) for path, wall in zip(trace_paths, wall_seconds)]
    n = len(runs)

    def measurement(values, unit="%"):
        stats = sample_stats(values)
        scale = 100 if unit == "%" else 1
        text = f"{stats['mean']*scale:.1f}{unit}±{stats['stdev']*scale:.1f}{unit} over {n} runs"
        text += f" (CV {stats['cv']:.1%})"
        return text, stats

    lines = ["# Profile", "", f"{n} separate profiler sessions; stdev is sample stdev.",
             "Busy time is the union of GPU kernel/copy intervals; profiler overhead affects idle estimates.",
             "CV >5% marks hints low-confidence; a single run cannot establish repeatability.", ""]
    busy, _ = measurement([r["busy"] for r in runs])
    idle, idle_stats = measurement([1-r["busy"] for r in runs])
    tiny, tiny_stats = measurement([r["tiny"] for r in runs])
    lines += [f"GPU busy/wall: {busy}; idle: {idle}.", f"Kernels <10 us: {tiny}.",
              "", "## Top 25 CUDA kernels", "", "| Name | Calls (all runs) | Mean ms/run ± stdev | CV |",
              "|---|---:|---:|---:|"]
    names = {name for r in runs for name in r["kernels"]}
    kernel_stats = {name: sample_stats([r["kernels"].get(name, [0, 0])[1]/1000 for r in runs])
                    for name in names}
    for name in sorted(names, key=lambda name: -kernel_stats[name]["mean"])[:25]:
        stats = kernel_stats[name]
        calls = sum(r["kernels"].get(name, [0, 0])[0] for r in runs)
        lines.append(f"| {name.replace('|', '/')} | {calls} | {stats['mean']:.3f} ± {stats['stdev']:.3f} "
                     f"| {stats['cv']:.1%} |")
    lines += ["", "## Categories (summed kernel time, overlaps included)", ""]
    measured = {}
    for category in [c[0] for c in CATEGORIES] + ["other"]:
        measured[category] = measurement([r["categories"].get(category, 0)/r["total_us"]
                                          if r["total_us"] else 0 for r in runs])
        lines.append(f"- {category}: {measured[category][0]} of GPU time")
    lines += ["", "## Hints (measured threshold rules)", ""]
    if any(not r["total_us"] for r in runs):
        lines.append("No hints: at least one trace has no GPU events; measurement is unavailable.")
        return "\n".join(lines) + "\n"
    hints = []
    for label, (text, stats), threshold, advice in [
        ("attention", measured["attention"], 0.4, "attention kernels dominate; inspect flash/SDPA backend selection"),
        ("GPU idle", (idle, idle_stats), 0.15, "possible CPU/launch bottleneck; inspect CPU trace and CUDA graphs"),
        ("memcpy", measured["memcpy"], 0.05, "inspect host/device transfers and pinned memory"),
        ("kernels <10 us", (tiny, tiny_stats), 0.5, "many tiny kernels; inspect fusion/CUDA graphs"),
    ]:
        if stats["mean"] > threshold:
            confidence = " LOW-CONFIDENCE" if stats["cv"] > 0.05 or n < 2 else ""
            hints.append(f"- {label} {text} exceeds {threshold:.0%}:{confidence} {advice}.")
    lines += hints or ["No measured threshold triggered."]
    return "\n".join(lines) + "\n"
