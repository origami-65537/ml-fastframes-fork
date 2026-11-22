"""Opt-in outer-call CUDA events; never patch the captured graph bodies."""
from collections import defaultdict

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


def summarize_events(events):
    chunks = defaultdict(lambda: defaultdict(float))
    for name, chunk, start, end in events:
        chunks[name][chunk] += start.elapsed_time(end)
    return {name: {**sample_stats(values.values()), "chunk_indices": list(values)}
            for name, values in chunks.items()}


def measure_stages(pipe, generate, chunk):
    import torch

    originals = {name: getattr(pipe, name) for name in ("dit", "refiner", "decoder")}
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
