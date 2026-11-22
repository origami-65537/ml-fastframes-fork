#!/usr/bin/env python3
"""Per-row summary of fixed_bench run JSONs."""
import json, sys
from pathlib import Path
for d in map(Path, sys.argv[1:]):
    js = sorted(d.glob("runs/*.json"))
    if not js:
        print(f"## {d.name}: no run record"); continue
    r = json.loads(js[-1].read_text()); f = r["fps"]
    print(f"## {d.name} sha={r['git']['sha'][:7]} dirty={r['git']['dirty']} shape={r['actual_shape']} verdict={r['verdict']}")
    print(f"fps samples={[round(x, 3) for x in f['samples']]} median={f['median']:.3f} cv={f['cv']:.4f} "
          f"ms/frame median={r['ms_per_frame']['median']:.2f} seconds={[round(s, 3) for s in r['seconds']]}")
    print(f"peak alloc/reserved GB={r['memory']['allocated_gb']:.2f}/{r['memory']['reserved_gb']:.2f} "
          f"ref={r['reference']} saved={r['reference_saved']} repeat_max_abs_diff={r['repeat_max_abs_diff']}")
    for q in r["repeat_quality"]:
        m = q["quality"]
        g = lambda k, s: m[k][s] if k in m else None  # noqa: E731
        print(f"  repeat {q['repeat']}: {q['verdict']} psnr mean/min={g('psnr','mean')}/{g('psnr','min')} "
              f"ssim min={g('ssim','min')} lpips max={g('lpips','max')} reasons={q['reasons']}")
    for name, st in r.get("stage_timing", {}).get("stages", {}).items():
        print(f"  stage {name}: {st['median']:.3f} ms/chunk median")
    print(f"reasons={r['reasons']}")
