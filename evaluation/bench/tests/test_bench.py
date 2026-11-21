import json
import math

import pytest
import torch

from evaluation.bench.compare import append_history, read_history, table
from evaluation.bench.fixed_bench import json_safe, workload_key
from evaluation.bench.profiling import classify, profile_report
from evaluation.bench.quality import compare, sanity, verdict


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_metrics():
    generator = torch.Generator().manual_seed(0)
    ref = torch.randint(60, 190, (2, 24, 24, 3), generator=generator, dtype=torch.uint8)
    identical = compare(ref, ref, with_lpips=False)
    assert math.isinf(identical["psnr"]["mean"])
    assert identical["ssim"]["mean"] == 1
    noise = torch.randn(ref.shape, generator=generator)
    mild = compare((ref + noise * 5).clamp(0, 255).byte(), ref, with_lpips=False)
    strong = compare((ref + noise * 30).clamp(0, 255).byte(), ref, with_lpips=False)
    assert mild["psnr"]["mean"] > strong["psnr"]["mean"]
    assert mild["ssim"]["mean"] > strong["ssim"]["mean"]
    assert len(mild["ssim"]["frames"]) == 2
    assert not sanity(ref, ref.shape)
    assert "shape mismatch" in sanity(ref, (3, 24, 24, 3))[0]
    with pytest.raises(ValueError, match="shape mismatch"):
        compare(ref, ref[:1])


def test_sanity_errors():
    assert sanity(torch.empty(0, 2, 2, 3))
    assert "constant frame" in sanity(torch.zeros(1, 2, 2, 3, dtype=torch.uint8))
    assert "mean luminance out of range" in sanity(torch.full((1, 2, 2, 3), 255, dtype=torch.uint8))
    assert "non-finite pixels" in sanity(torch.full((1, 2, 2, 3), float("nan")))


def test_verdicts():
    assert verdict({}, [], False) == ("no-reference", [])
    assert verdict({}, [], True) == ("pass", [])
    assert verdict({}, ["bad frame"], False)[0] == "fail"
    assert verdict({"psnr": {"min": 29}}, [], True)[0] == "fail"
    assert verdict({"ssim": {"min": 0.89}}, [], True)[0] == "fail"
    assert verdict({"lpips": {"max": 0.11}}, [], True)[0] == "fail"
    assert verdict({}, [], True, fps=80, previous_fps=100, max_fps_drop=0.1)[0] == "fail"
    assert verdict({}, [], True, fps=95, previous_fps=100, max_fps_drop=0.1)[0] == "pass"


def test_history(tmp_path):
    path = tmp_path / "history.jsonl"
    assert read_history(path) == []
    row = {"key": "abc", "git": {"sha": "1234567890abcd"}, "fps": {"median": 42},
           "memory": {"allocated_gb": 2, "reserved_gb": 3}, "quality": {}, "verdict": "no-reference"}
    append_history(path, row)
    append_history(path, row)
    assert read_history(path) == [row, row]
    assert "42.000" in table(read_history(path))
    assert table([]) == ""
    assert json_safe(float("inf")) == "inf"
    assert workload_key({"seed": 0, "prompt": "a"}) == workload_key({"prompt": "a", "seed": 0})
    assert workload_key({"seed": 0}) != workload_key({"seed": 1})


@pytest.mark.parametrize("name,category", [("flash_fwd", "attention"), ("ampere_gemm", "GEMM"),
                                           ("cudnn_conv", "conv"), ("layer_norm", "norm"),
                                           ("triton_fused", "elementwise/triton fused"),
                                           ("Memcpy HtoD", "memcpy"), ("unknown", "other")])
def test_categories(name, category):
    assert classify(name) == category


def test_profile(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"traceEvents": [
        {"cat": "kernel", "name": "flash_fwd", "ts": 0, "dur": 600},
        {"cat": "kernel", "name": "gemm", "ts": 300, "dur": 100}]}))
    report = profile_report(trace, 0.001)
    assert "60.0%" in report
    assert "Attention >40%" in report
    assert "GPU idle >15%" in report
    trace.write_text('{"traceEvents": []}')
    assert "Top 25" in profile_report(trace, 1)
