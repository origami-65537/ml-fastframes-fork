import json
import math
from pathlib import Path

import pytest
import torch

from evaluation.bench import fixed_bench
from evaluation.bench.compare import append_history, read_history, table
from evaluation.bench.fixed_bench import fps_baseline, fps_gate, json_safe, max_abs_diff, parser, workload_key
from evaluation.bench.profiling import classify, correlation_stages, profile_report
from evaluation.bench.quality import compare, sanity, verdict
from evaluation.bench.stages import TimingProxy, annotate_stages, measure_stages, summarize_events
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


def test_solid_color_frame():
    red = torch.zeros(1, 12, 12, 3, dtype=torch.uint8)
    red[..., 0] = 255
    assert sanity(red) == ["constant frame"]
    varying = red.clone()
    varying[:, :6, :, 0] = 100
    assert "constant frame" not in sanity(varying)
    assert "constant frame" in sanity(torch.cat([varying, red]))


@pytest.mark.parametrize("case", ["normal", "constant", "empty", "reference", "shape"])
def test_every_repeat(monkeypatch, tmp_path, capsys, case):
    generator = torch.Generator().manual_seed(0)
    good = torch.randint(40, 210, (2, 12, 12, 3), generator=generator, dtype=torch.uint8)
    first = {"normal": good, "constant": torch.zeros_like(good), "empty": good[:0],
             "reference": 255-good, "shape": good[:1]}[case]
    videos = iter([first, good])
    monkeypatch.setattr(fixed_bench, "adapter", lambda args:
                        fixed_bench.Workload(lambda: next(videos), lambda: None, good.shape))
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
    assert record["repeat_quality"][0]["max_abs_diff_vs_previous"] is None
    if case == "normal":
        assert record["repeat_max_abs_diff"] == 0
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
                                           ("cudnn_conv", "conv"), ("qk_int8_sv_f8_attn_kernel", "attention"),
                                           ("quant_per_block_int8_kernel", "attention quant"),
                                           ("mean_scale_fuse_quant_kernel", "attention quant"),
                                           ("cutlass_fp8_scaled_mm_dequant", "GEMM"), ("dequantize_gemm_fp8", "GEMM"),
                                           ("triton_red_fused_native_layer_norm_quant", "norm"),
                                           ("fp8_quant_kernel", "quant"), ("layer_norm", "norm"),
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
    assert (args.repeats, args.warmup, args.profile_repeats) == (3, 2, 3)
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


def test_profile_stages(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"traceEvents": [
        {"cat": "user_annotation", "name": "bench_stage/dit", "pid": 1, "tid": 1, "ts": 0, "dur": 100},
        {"cat": "user_annotation", "name": "bench_stage/refiner", "pid": 1, "tid": 1, "ts": 200, "dur": 100},
        {"cat": "user_annotation", "name": "other", "pid": 1, "tid": 1, "ts": 400, "dur": 100},
        {"cat": "cuda_runtime", "name": "cudaGraphLaunch", "pid": 1, "tid": 1, "ts": 10, "dur": 5,
         "args": {"correlation": 1}},
        {"cat": "cuda_runtime", "name": "cudaLaunchKernel", "pid": 1, "tid": 1, "ts": 210, "dur": 5,
         "args": {"correlation": 2}},
        {"cat": "cuda_runtime", "name": "cudaLaunchKernel", "pid": 1, "tid": 2, "ts": 20, "dur": 5,
         "args": {"correlation": 3}},
        {"cat": "kernel", "name": "flash_fwd", "ts": 20, "dur": 300, "args": {"correlation": 1}},
        {"cat": "kernel", "name": "gemm", "ts": 320, "dur": 100, "args": {"correlation": 1}},
        {"cat": "kernel", "name": "layer_norm", "ts": 420, "dur": 50, "args": {"correlation": 2}},
        {"cat": "kernel", "name": "gemm", "ts": 470, "dur": 50, "args": {"correlation": 3}}]}))
    report = profile_report([trace] * 2, [0.001] * 2)
    assert "| dit | 0.4 ± 0.0 | 80.0% |" in report
    assert "| refiner | 0.1 ± 0.0 | 10.0% |" in report
    assert "| unattributed | 0.1 ± 0.0 | 10.0% |" in report
    assert "### dit: categories" in report and "- attention: 75.0%±0.0%" in report
    assert "| flash_fwd | 1 | 0.300 ± 0.000 | 75.0% |" in report


def test_nested_stages_innermost(tmp_path):
    def annotation(name, ts, dur, tid=1):
        return {"cat": "user_annotation", "name": f"bench_stage/{name}", "pid": 1, "tid": tid, "ts": ts, "dur": dur}

    def launch(ts, correlation, tid=1):
        return {"cat": "cuda_runtime", "name": "cudaLaunchKernel", "pid": 1, "tid": tid, "ts": ts, "dur": 1,
                "args": {"correlation": correlation}}

    events = [annotation("step", 0, 200), annotation("dit", 0, 100), annotation("dit.begin_chunk", 10, 20),
              launch(15, 1), launch(50, 2), launch(150, 3), launch(250, 4), launch(15, 5, tid=2)]
    assert correlation_stages(events) == {1: "dit.begin_chunk", 2: "dit", 3: "step"}
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"traceEvents": events + [
        {"cat": "kernel", "name": f"k{c}", "ts": 300 + 10 * c, "dur": 10, "args": {"correlation": c}}
        for c in range(1, 6)]}))
    report = profile_report([trace], [0.001])
    assert "| dit.begin_chunk | 0.0 ± 0.0 | 20.0% |" in report
    assert "| step | 0.0 ± 0.0 | 20.0% |" in report
    assert "| unattributed | 0.0 ± 0.0 | 40.0% |" in report


def test_annotate_stages(monkeypatch):
    labels = []

    class Record:
        def __init__(self, label):
            labels.append(label)

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass

    class Stage:
        def __call__(self, x):
            return x + 1

        def begin_chunk(self, x):
            return x * 10

    class Pipe:
        dit, refiner, decoder = Stage(), Stage(), Stage()

        def step(self, x):
            return self.decoder(self.dit(self.dit.begin_chunk(x)))

    monkeypatch.setattr(torch.profiler, "record_function", Record)
    pipe = Pipe()
    original = pipe.dit
    with annotate_stages(pipe):
        assert pipe.step(1) == 12
    assert labels == ["bench_stage/step", "bench_stage/dit.begin_chunk", "bench_stage/dit", "bench_stage/decoder"]
    assert pipe.dit is original and "begin_chunk" not in vars(original)
    assert "step" not in vars(pipe) and pipe.step(1) == 12


def test_measure_stages_begin_chunk(monkeypatch):
    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def elapsed_time(self, end):
            return 1

    class Stage:
        def __call__(self, x):
            return x

        def begin_chunk(self, x):
            return x

    class Pipe:
        dit, refiner, decoder = Stage(), Stage(), Stage()

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    pipe = Pipe()
    result = measure_stages(pipe, lambda: pipe.dit(pipe.dit.begin_chunk(1)), lambda: 0)
    assert set(result["stages"]) == {"dit", "dit.begin_chunk"}
    assert "begin_chunk" not in vars(pipe.dit)


def test_fps_gate_and_repeat_diff():
    steady, noisy = sample_stats([10, 10.1, 10.05]), sample_stats([8, 10, 12])
    previous = {"fps": steady, "git": {"sha": "abc"}}
    assert fps_gate(steady, previous, None, 0.02) == {"status": "off"}
    assert fps_gate(steady, None, 0.1, 0.02)["status"] == "no-previous"
    assert fps_gate(noisy, None, 0.1, 0.02)["status"] == "low-confidence"
    assert fps_gate(steady, previous, 0.1, 0.02)["status"] == "gated"
    assert fps_gate(noisy, previous, 0.1, 0.02)["status"] == "low-confidence"
    assert fps_gate(steady, {"fps": {"median": 10}, "git": {"sha": "abc"}}, 0.1, 0.02)["status"] == "gated"
    history = [{"key": "k", "fps": steady, "id": 1}, {"key": "other", "fps": steady, "id": 2},
               {"key": "k", "fps": noisy, "id": 3}]
    assert fps_baseline(history, "k", 0.02)["id"] == 1
    assert fps_baseline(history, "k", 1.0)["id"] == 3
    assert fps_baseline(history[2:], "k", 0.02) is None
    a = torch.tensor([[0, 255], [10, 20]], dtype=torch.uint8)
    b = torch.tensor([[3, 250], [10, 20]], dtype=torch.uint8)
    assert max_abs_diff(a, b) == 5 and max_abs_diff(b, a) == 5
    assert max_abs_diff(a, None) is None and max_abs_diff(a, a[:1]) is None


def fake_bench(monkeypatch, generate=None, timing_scope=None):
    video = torch.randint(40, 210, (2, 12, 12, 3), generator=torch.Generator().manual_seed(0), dtype=torch.uint8)
    monkeypatch.setattr(fixed_bench, "adapter", lambda args: fixed_bench.Workload(
        generate or (lambda: video), lambda: None, video.shape, timing_scope=timing_scope))
    for name in ("set_device", "synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda *args: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "fake CPU test")
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)


def test_timing_scope_changes_key(monkeypatch, tmp_path):
    keys, reference_keys = {}, set()
    for scope in (None, "steps only (start untimed)"):
        fake_bench(monkeypatch, timing_scope=scope)
        history = tmp_path / f"{scope is None}.jsonl"
        fixed_bench.main(["--warmup", "0", "--repeats", "1", "--history", str(history),
                          "--reference-dir", str(tmp_path / "ref"), "--run-dir", str(tmp_path / "runs")])
        record = read_history(history)[0]
        keys[scope] = record["key"]
        reference_keys.add(record["reference_key"])
        assert record["timing_scope"] == (scope or "full generate()")
        assert record["config"].get("timing_scope") == scope
    assert keys[None] != keys["steps only (start untimed)"]
    assert reference_keys == {keys[None]}


def test_reference_survives_timing_scope(monkeypatch, tmp_path):
    import fastframes.video

    fake_bench(monkeypatch)
    monkeypatch.setattr(fastframes.video, "save_video", lambda path, video: None)
    argv = ["--warmup", "0", "--repeats", "1", "--reference-dir", str(tmp_path / "ref"),
            "--run-dir", str(tmp_path / "runs")]
    fixed_bench.main(argv + ["--history", str(tmp_path / "base.jsonl"), "--save-reference"])
    worse = lambda: torch.full((2, 12, 12, 3), 128, dtype=torch.uint8) | 1  # noqa: E731
    fake_bench(monkeypatch, generate=worse, timing_scope="steps only (start untimed)")
    history = tmp_path / "candidate.jsonl"
    assert fixed_bench.main(argv + ["--history", str(history)]) == 1
    record = read_history(history)[0]
    assert record["verdict"] == "fail" and record["quality"]


def test_timing_scoped_reference_still_loads(monkeypatch, tmp_path):
    fake_bench(monkeypatch, generate=lambda: torch.full((2, 12, 12, 3), 129, dtype=torch.uint8),
               timing_scope="steps only (start untimed)")
    history = tmp_path / "history.jsonl"
    argv = ["--warmup", "0", "--repeats", "1", "--history", str(history),
            "--reference-dir", str(tmp_path / "ref"), "--run-dir", str(tmp_path / "runs")]
    fixed_bench.main(argv)
    scoped = Path(read_history(history)[0]["reference"]).with_name(read_history(history)[0]["key"] + ".pt")
    scoped.parent.mkdir(parents=True)
    torch.save(torch.randint(40, 210, (2, 12, 12, 3), dtype=torch.uint8), scoped)
    assert fixed_bench.main(argv) == 1
    record = read_history(history)[-1]
    assert record["reference"] == str(scoped) and record["quality"] and record["verdict"] == "fail"


def test_noisy_run_never_becomes_baseline(monkeypatch, tmp_path):
    fake_bench(monkeypatch)
    monkeypatch.setattr(fixed_bench, "workload_key", lambda config: "test")
    history = tmp_path / "history.jsonl"
    append_history(history, {"key": "test", "git": {"sha": "base"}, "fps": sample_stats([100.0] * 3)})
    argv = ["--warmup", "0", "--history", str(history), "--max-fps-drop", "0.1",
            "--reference-dir", str(tmp_path / "ref"), "--run-dir", str(tmp_path / "runs")]
    statuses = []
    for seconds in ([2 / 70, 2 / 80, 2 / 90], [2 / 80] * 3):  # 2-frame fake video
        times = iter([t for s in seconds for t in (0.0, s)])
        monkeypatch.setattr(fixed_bench.time, "perf_counter", lambda: next(times))
        result = fixed_bench.main(argv)
        record = read_history(history)[-1]
        statuses.append((record["fps_gate"]["status"], result))
        if record["fps_gate"]["status"] == "gated":
            assert record["fps_gate"]["previous_sha"] == "base"
    # The noisy 80 fps run is skipped; the steady rerun still compares against the 100 fps baseline.
    assert statuses == [("low-confidence", 0), ("gated", 1)]


@pytest.mark.parametrize("seconds,expected", [([1.0, 1.0, 1.0], "gated"), ([1.0, 1.2, 0.8], "low-confidence")])
def test_low_confidence_skips_fps_gate(monkeypatch, tmp_path, capsys, seconds, expected):
    fake_bench(monkeypatch)
    times = iter([t for s in seconds for t in (0.0, s)])
    monkeypatch.setattr(fixed_bench.time, "perf_counter", lambda: next(times))
    history = tmp_path / "history.jsonl"
    previous = {"key": "test", "git": {"sha": "abc"}, "fps": sample_stats([100.0] * 3)}
    argv = ["--warmup", "0", "--history", str(history), "--max-fps-drop", "0.1",
            "--reference-dir", str(tmp_path / "ref"), "--run-dir", str(tmp_path / "runs")]
    monkeypatch.setattr(fixed_bench, "workload_key", lambda config: "test")
    append_history(history, previous)
    result = fixed_bench.main(argv)
    record = read_history(history)[-1]
    assert record["fps_gate"]["status"] == expected
    if expected == "gated":
        assert result == 1 and any("fps drop" in r for r in record["reasons"])
    else:
        assert not any("fps drop" in r for r in record["reasons"])
        assert "LOW-CONFIDENCE" in capsys.readouterr().out


@pytest.mark.parametrize("fail_start", [False, True])
def test_lingbot_adapter_start_once(monkeypatch, tmp_path, fail_start):
    import apps.lingbot.generate as lingbot

    calls = []

    class FakePipe:
        def start(self, image, prompt, *, seed, latents):
            calls.append(("start", seed, latents))
            if fail_start:
                raise RuntimeError("encode failed")

        def restart(self, *, seed):
            calls.append(("restart", seed))

        def step(self, rays):
            calls.append(("step", rays))
            return torch.zeros(1, 2, 2, 3, dtype=torch.uint8)

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(lingbot, "import_upstream", lambda root: None)
    monkeypatch.setattr(lingbot, "load_pipeline", lambda args, world, rank: FakePipe())
    monkeypatch.setattr(lingbot, "read_condition", lambda args: ("image", "prompt", lambda chunk: chunk, 2, 7))
    args = parser().parse_args(["--model", "lingbot-1.3b", "--example", str(tmp_path)])
    if fail_start:
        with pytest.raises(RuntimeError, match="encode failed"):
            fixed_bench.adapter(args)
        assert calls == [("start", 7, 2 * lingbot.CHUNK), ("close",)]
        return
    workload = fixed_bench.adapter(args)
    generate, close = workload.generate, workload.close
    assert workload.timing_scope == "steps only (start untimed)"
    assert workload.expected_shape == (2 * lingbot.CHUNK * 4 - 3, 464, 832, 3)
    assert calls == [("start", 7, 2 * lingbot.CHUNK)]
    for _ in range(3):
        assert generate().shape == (2, 2, 2, 3)
    assert calls[1:] == [("restart", 7), ("step", 0), ("step", 1)] * 3
    close()
    assert calls[-1] == ("close",)
