"""Print benchmark history grouped by fixed workload key."""
import argparse
import json
from pathlib import Path


def read_history(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def append_history(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


def table(records):
    lines = []
    for key in sorted({r["key"] for r in records}):
        lines += [f"key: {key}", "sha          fps       VRAM alloc/res GB    PSNR       SSIM       verdict"]
        for r in records:
            if r["key"] != key:
                continue
            q = r.get("quality", {})
            lines.append(f"{r['git']['sha'][:10]:12} {r['fps']['median']:9.3f} "
                         f"{r['memory']['allocated_gb']:.3f}/{r['memory']['reserved_gb']:.3f}          "
                         f"{str(q.get('psnr', {}).get('mean', '-')):10.10} "
                         f"{str(q.get('ssim', {}).get('mean', '-')):10.10} {r['verdict']}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=Path("outputs/bench/history.jsonl"))
    print(table(read_history(parser.parse_args().history)))
