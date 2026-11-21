"""Chrome trace kernel analysis. Heuristics describe symptoms, not promised speedups."""
import json
import re
from collections import defaultdict

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


def profile_report(trace_path, wall_seconds):
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
    ratio = min(1, busy / (wall_seconds * 1e6))
    lines = ["# Profile", "", f"GPU busy/wall: {ratio:.1%}; idle: {1-ratio:.1%}.",
             "Busy time is the union of GPU kernel/copy intervals; profiler overhead affects idle estimates.",
             "", "## Top 25 CUDA kernels", "", "| Name | Calls | Total ms | GPU time |",
             "|---|---:|---:|---:|"]
    for name, (calls, duration) in sorted(totals.items(), key=lambda item: -item[1][1])[:25]:
        lines.append(f"| {name.replace('|', '/')} | {calls} | {duration/1000:.3f} | "
                     f"{duration/max(total, 1):.1%} |")
    lines += ["", "## Categories (summed kernel time, overlaps included)", ""]
    for category in [c[0] for c in CATEGORIES] + ["other"]:
        lines.append(f"- {category}: {categories[category]/1000:.3f} ms "
                     f"({categories[category]/max(total, 1):.1%})")
    hints = []
    if categories["attention"] > total * 0.4:
        hints.append("Attention >40%: attention kernels dominate; inspect flash/SDPA backend selection.")
    if ratio < 0.85:
        hints.append("GPU idle >15%: possible CPU/launch bound workload; consider CUDA graphs.")
    if categories["memcpy"] > total * 0.05:
        hints.append("Memcpy >5%: inspect host/device transfers and pinned memory.")
    if kernels and sum(e["dur"] < 10 for e in kernels) > len(kernels) * 0.5:
        hints.append("Over half of kernels <10 us: many tiny kernels; consider fusion/CUDA graphs.")
    lines += ["", "## Hints (simple threshold rules)", ""] + [f"- {h}" for h in hints or
                                                               ["No heuristic threshold triggered."]]
    return "\n".join(lines) + "\n"
