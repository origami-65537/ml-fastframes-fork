#!/usr/bin/env python3
"""PR 6 bench: pick diag scope / boundary rows, and build the summary table + per-frame CSVs."""
import csv
import glob
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
OUT = Path("outputs/bench_pr6")
WARN = "rowwise _scaled_mm unavailable"
PROJ = ["self_qkv", "self_out", "cross_q", "cross_out", "ffn_up", "ffn_down"]


def rows():
    out = []
    for line in (OUT / "rows.txt").read_text().splitlines():
        name, kind, rc, envs, args = line.split("|")
        out.append({"name": name, "kind": kind, "rc": int(rc), "env": envs, "args": args})
    return out


def record(name):
    files = sorted(glob.glob(str(OUT / name / "runs" / "*.json")))
    return json.loads(Path(files[-1]).read_text()) if files else None


def gate_mean(name):
    rec = record(name)
    if not rec or not rec.get("repeat_quality"):
        return None
    return min(r["quality"]["psnr"]["mean"] for r in rec["repeat_quality"] if r["quality"])


def cmd_pick():
    """worst failing single projection (or 1), passing projections."""
    worst, passing = None, []
    for p in PROJ:
        rec = record(f"fp8_{p}")
        if not rec:
            continue
        if rec["verdict"] == "pass":
            passing.append(p)
        elif worst is None or gate_mean(f"fp8_{p}") < gate_mean(f"fp8_{worst}"):
            worst = p
    print(worst or "1", ",".join(passing))


def cmd_border():
    for r in rows():
        if r["kind"] == "fp8" and r["name"].startswith(("fp8_", "diag_", "direct_", "combo")):
            m = gate_mean(r["name"])
            if m is not None and 29 <= m <= 31:
                print(r["name"])


def load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True) if Path(path).exists() else None


def cmd_table():
    import torch
    from evaluation.bench.quality import compare
    controls = {"": "ctrl_a", "_s1": "ctrl_s1", "_ex01": "ctrl_ex01"}
    lines = ["| row | overrides | exit | ms/chunk median (min-max) | fps median | DiT ms/chunk | peak alloc/res GB | gate | "
             "PSNR mean/min vs cuDNN | SSIM min vs cuDNN | LPIPS max vs cuDNN | PSNR mean/min vs Sage ctrl | SSIM min vs Sage ctrl | rowwise warn |",
             "|" + "---|" * 14]
    extra = []
    for r in rows():
        if r["kind"] != "bench" and r["kind"] != "fp8":
            continue
        name, rec = r["name"], record(r["name"])
        log = (OUT / f"{name}.log").read_text(errors="replace")
        warn = "yes" if WARN in log else "no"
        if not rec:
            lines.append(f"| {name} | {r['env']} {r['args']} | {r['rc']} | crashed (see {name}.log) |" + " |" * 10 + f" {warn} |")
            continue
        chunks = 5
        ms = [1000 * s / chunks for s in rec["seconds"]]
        dit = rec.get("stage_timing", {}).get("stages", {})
        dit_ms = next((v["median"] for k, v in dit.items() if k.startswith("dit") and "forward" in k), None)
        if dit_ms is None:
            dit_ms = next((v["median"] for k, v in dit.items() if k.startswith("dit")), None)
        qs = [q["quality"] for q in rec["repeat_quality"] if q["quality"]]
        gp = f"{min(q['psnr']['mean'] for q in qs):.2f}/{min(q['psnr']['min'] for q in qs):.2f}" if qs else "-"
        gs = f"{min(q['ssim']['min'] for q in qs):.4f}" if qs else "-"
        gl = f"{max(q['lpips']['max'] for q in qs):.4f}" if qs and all("lpips" in q for q in qs) else "-"
        if gl == "-" and Path(OUT / f"{name}.video.pt").exists() and Path(rec["reference"]).exists():
            try:
                lp = compare(load(OUT / f"{name}.video.pt"), load(rec["reference"]), with_lpips=True).get("lpips")
                gl = f"{lp['max']:.4f}*" if lp else "-"
            except Exception as error:
                gl = f"err {type(error).__name__}"
        suffix = "_s1" if name.endswith("_s1") else "_ex01" if name.endswith("_ex01") else ""
        ctrl = controls[suffix]
        video, cvideo = load(OUT / f"{name}.video.pt"), load(OUT / f"{ctrl}.video.pt")
        cp = cs = "-"
        if video is not None and cvideo is not None and name != ctrl:
            m = compare(video, cvideo, with_lpips=False)
            cp, cs = f"{m['psnr']['mean']:.2f}/{m['psnr']['min']:.2f}", f"{m['ssim']['min']:.4f}"
            g = rec["quality"]
            with open(OUT / f"{name}.frames.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frame", "psnr_vs_cudnn", "ssim_vs_cudnn", "psnr_vs_sage_ctrl", "ssim_vs_sage_ctrl"])
                for i in range(len(m["psnr"]["frames"])):
                    w.writerow([i, g.get("psnr", {}).get("frames", [None] * (i + 1))[i],
                                g.get("ssim", {}).get("frames", [None] * (i + 1))[i],
                                m["psnr"]["frames"][i], m["ssim"]["frames"][i]])
            ref = load(rec["reference"])
            if ref is not None and ref.shape == video.shape and name.startswith(("fp8", "diag", "direct", "combo")):
                from PIL import Image
                worst = min(range(len(g["psnr"]["frames"])), key=lambda i: g["psnr"]["frames"][i])
                diff = ((video[worst].int() - ref[worst].int()).abs() * 4).clamp(0, 255).to(torch.uint8)
                (OUT / "vis").mkdir(exist_ok=True)
                Image.fromarray(torch.cat([ref[worst], video[worst], diff], 1).numpy()).save(OUT / "vis" / f"{name}_f{worst}.png")
            if name in ("ctrl_b", "packed_off"):
                extra.append(f"- {name} vs ctrl_a: bit_identical={torch.equal(video, cvideo)}, "
                             f"max|diff|={(video.int() - cvideo.int()).abs().max().item()}, PSNR mean/min {cp}")
        lines.append(f"| {name} | {r['env']} {r['args']} | {r['rc']} | {statistics.median(ms):.1f} ({min(ms):.1f}-{max(ms):.1f}) | "
                     f"{rec['fps']['median']:.2f} | {dit_ms if dit_ms is None else round(dit_ms, 1)} | "
                     f"{rec['memory']['allocated_gb']:.2f}/{rec['memory']['reserved_gb']:.2f} | {rec['verdict']} | "
                     f"{gp} | {gs} | {gl} | {cp} | {cs} | {warn} |")
    smoke = ["", "## Two-start smoke (start->step->start->step, 29 frames)", ""]
    for r in rows():
        if r["kind"] == "smoke":
            log = (OUT / f"{r['name']}.log").read_text(errors="replace").splitlines()
            keep = [x for x in log if x.startswith(("MEM before DiT capture #1:", "MEM at", "LATENCY", "SAME_INPUT",
                                                    "PEAK", "SMOKE_", "torch.OutOfMemoryError", "RuntimeError"))]
            smoke.append(f"### {r['name']} ({r['env']}) exit={r['rc']}")
            smoke += [f"    {x}" for x in keep] + [""]
    for r in rows():
        if r["kind"] == "bench" and r["name"].startswith("graphs"):
            log = (OUT / f"{r['name']}.log").read_text(errors="replace").splitlines()
            keep = [x for x in log if x.startswith(("MEM before DiT capture #1:", "MEM at"))]
            smoke.append(f"### {r['name']} bench capture memory")
            smoke += [f"    {x}" for x in keep[:4]] + [""]
    (OUT / "table.md").write_text("\n".join(lines + ["", "## Controls"] + extra + smoke) + "\n")
    print((OUT / "table.md").read_text())


if __name__ == "__main__":
    {"pick": cmd_pick, "border": cmd_border, "table": cmd_table}[sys.argv[1]]()
