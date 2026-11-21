import json
import math

import pytest
import torch

from evaluation.bench import fixed_bench
from evaluation.bench.compare import append_history, read_history, table
from evaluation.bench.fixed_bench import json_safe, parser, workload_key
from evaluation.bench.profiling import classify, profile_report
from evaluation.bench.quality import compare, sanity, verdict
from evaluation.bench.stages import TimingProxy, measure_stages, summarize_events
from evaluation.bench.stats import sample_stats


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


@pytest.mark.parametrize("case", ["normal", "constant", "empty", "reference", "shape"])
def test_every_repeat(monkeypatch, tmp_path, capsys, case):
    generator = torch.Generator().manual_seed(0)
    good = torch.randint(40, 210, (2, 12, 12, 3), generator=generator, dtype=torch.uint8)
    first = {"normal": good, "constant": torch.zeros_like(good), "empty": good[:0],
             "reference": 255-good, "shape": good[:1]}[case]
    videos = iter([first, good])
    monkeypatch.setattr(fixed_bench, "adapter", lambda args:
                        (lambda: next(videos), lambda: None, good.shape, None))
    for name in ("set_device", "synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda *args: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "fake CPU test")
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(fixed_bench, "workload_key", lambda config: "test")
    calls = []

    def compare_repeat(video, reference):
        calls.append(video.shape)
        return compare(video, reference, with_lpips=False)

    monkeypatch.setattr(fixed_bench, "compare", compare_repeat)
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    if case in ("reference", "shape", "normal"):
        torch.save(good, reference_dir / "test.pt")
    history = tmp_path / "history.jsonl"
    result = fixed_bench.main(["--warmup", "0", "--repeats", "2", "--reference-dir", str(reference_dir),
                               "--history", str(history), "--run-dir", str(tmp_path / "runs")])
    record = read_history(history)[0]
    assert result == int(case != "normal")
    assert len(record["repeat_quality"]) == 2
    assert record["repeat_quality"][1]["verdict"] != "fail"
    if case != "normal":
        assert record["repeat_quality"][0]["verdict"] == "fail"
        assert all(reason.startswith("repeat 1:") for reason in record["reasons"])
        assert "FAIL: repeat 1:" in capsys.readouterr().out
    if case in ("reference", "normal"):
        assert len(calls) == 2
    elif case == "shape":
        assert len(calls) == 1


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
    report = profile_report([trace] * 3, [0.001] * 3)
    assert "60.0%" in report
    assert "attention 85.7%±0.0% over 3 runs" in report
    assert "GPU idle 40.0%±0.0% over 3 runs" in report
    assert "LOW-CONFIDENCE" not in report
    assert "0.600 ± 0.000" in report
    trace.write_text('{"traceEvents": []}')
    empty = profile_report([trace], [1])
    assert "No hints" in empty
    assert "possible CPU" not in empty
    with pytest.raises(ValueError):
        profile_report([], [])


def test_profile_variation(tmp_path):
    traces = []
    for i, duration in enumerate([450, 650, 950]):
        trace = tmp_path / f"trace_{i}.json"
        trace.write_text(json.dumps({"traceEvents": [
            {"cat": "kernel", "name": "flash_fwd", "ts": 0, "dur": duration},
            {"cat": "kernel", "name": "gemm", "ts": duration, "dur": 1000-duration}]}))
        traces.append(trace)
    report = profile_report(traces, [0.001] * 3)
    assert "attention 68.3%±25.2% over 3 runs (CV 36.8%) exceeds 40%: LOW-CONFIDENCE" in report
    assert "0.683 ± 0.252" in report


def test_stats_and_defaults():
    stats = sample_stats([2, 4, 6])
    assert stats == {"samples": [2, 4, 6], "median": 4, "mean": 4, "stdev": 2,
                     "min": 2, "max": 6, "cv": 0.5}
    assert sample_stats([0])["cv"] == 0
    with pytest.raises(ValueError):
        sample_stats([])
    args = parser().parse_args([])
    assert (args.repeats, args.warmup, args.profile_repeats) == (10, 2, 3)
    row = {"key": "a", "git": {"sha": "abc"}, "fps": stats, "ms_per_frame": stats,
           "memory": {"allocated_gb": 1, "reserved_gb": 2}, "verdict": "pass"}
    report = table([row])
    assert "ms_per_frame" in report and "50.00%" in report and "stdev" in report


def test_stage_proxy():
    class Event:
        def record(self):
            pass

        def elapsed_time(self, end):
            return 2.5

    class Target:
        attribute = "forwarded"

        def __call__(self, x):
            if x is None:
                raise ValueError("failed")
            return x + 1

    events = []
    chunk = [0]
    proxy = TimingProxy(Target(), "dit", events, lambda: chunk[0], Event)
    assert proxy.attribute == "forwarded"
    assert proxy(2) == 3
    proxy(2)
    chunk[0] = 1
    proxy(2)
    assert summarize_events(events)["dit"]["samples"] == [5, 2.5]
    assert summarize_events(events)["dit"]["median"] == 3.75
    assert summarize_events([]) == {}
    with pytest.raises(ValueError, match="failed"):
        proxy(None)
    assert len(events) == 3


@pytest.mark.parametrize("fail", [False, True])
def test_stage_restoration(monkeypatch, fail):
    class Pipe:
        dit = staticmethod(lambda x: x)
        refiner = staticmethod(lambda x: x)
        decoder = staticmethod(lambda x: x)

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def elapsed_time(self, end):
            return 1

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    pipe = Pipe()
    originals = [pipe.dit, pipe.refiner, pipe.decoder]

    def generate():
        pipe.dit(1)
        if fail:
            raise ValueError("generation failed")
        pipe.decoder(1)

    if fail:
        with pytest.raises(ValueError, match="generation failed"):
            measure_stages(pipe, generate, lambda: 0)
    else:
        result = measure_stages(pipe, generate, lambda: 0)
        assert result["stages"]["dit"]["median"] == 1
        assert "refiner" not in result["stages"]
    assert [pipe.dit, pipe.refiner, pipe.decoder] == originals
