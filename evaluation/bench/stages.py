"""Opt-in outer-call CUDA events; never patch the captured graph bodies."""
from collections import defaultdict
from contextlib import contextmanager

from evaluation.bench.stats import sample_stats


class TimingProxy:
    def __init__(self, target, name, events, chunk, event_factory):
        self.target, self.name, self.events = target, name, events
        self.chunk, self.event_factory = chunk, event_factory

    def __getattr__(self, name):
        return getattr(self.target, name)

    def __call__(self, *args, **kwargs):
        start, end = self.event_factory(), self.event_factory()
        start.record()
        result = self.target(*args, **kwargs)
        end.record()
        self.events.append((self.name, self.chunk(), start, end))
        return result


STAGES = ("dit", "refiner", "decoder")


class AnnotationProxy:
    def __init__(self, target, label, record_function):
        self.target, self.label, self.record_function = target, label, record_function

    def __getattr__(self, name):
        return getattr(self.target, name)

    def __call__(self, *args, **kwargs):
        with self.record_function(self.label):
            return self.target(*args, **kwargs)


@contextmanager
def annotate_stages(pipe, prefix="bench_stage/"):
    """Label each stage call so profiler kernels can be attributed through launch correlation."""
    import torch

    originals = {name: getattr(pipe, name) for name in STAGES}
    try:
        for name, target in originals.items():
            setattr(pipe, name, AnnotationProxy(target, prefix + name, torch.profiler.record_function))
        if hasattr(originals["dit"], "begin_chunk"):
            # The camera branch runs in dit.begin_chunk, outside dit's own call.
            pipe.dit.begin_chunk = AnnotationProxy(originals["dit"].begin_chunk, prefix + "dit.begin_chunk",
                                                   torch.profiler.record_function)
        yield
    finally:
        for name, target in originals.items():
            setattr(pipe, name, target)


def summarize_events(events):
    chunks = defaultdict(lambda: defaultdict(float))
    for name, chunk, start, end in events:
        chunks[name][chunk] += start.elapsed_time(end)
    return {name: {**sample_stats(values.values()), "chunk_indices": list(values)}
            for name, values in chunks.items()}


def measure_stages(pipe, generate, chunk):
    import torch

    originals = {name: getattr(pipe, name) for name in STAGES}
    events = []
    try:
        for name, target in originals.items():
            setattr(pipe, name, TimingProxy(target, name, events, chunk,
                                           lambda: torch.cuda.Event(enable_timing=True)))
        torch.cuda.synchronize()
        output = generate()
        torch.cuda.synchronize()
        del output
        return {"method": "CUDA events outside stage call objects, one extra generation",
                "stages": summarize_events(events),
                "scope": "Sum calls per chunk; DiT includes first-chunk clean pass. Refiner omits chunk 0. "
                         "Excludes start/encoding, dit.begin_chunk, camera rays, scheduler and output conversion. "
                         "Events bracket warmed graph replay on the current stream, not graph capture bodies. "
                         "Diagnostic overhead is excluded from benchmark fps/VRAM; no stage repeatability claim."}
    finally:
        for name, target in originals.items():
            setattr(pipe, name, target)
