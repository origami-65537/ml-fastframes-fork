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
def wrapped_stages(pipe, wrap, step=False):
    """Replace the stage calls (and dit.begin_chunk, optionally pipe.step) with wrap(target, name)."""
    originals = {name: getattr(pipe, name) for name in STAGES}
    own_step = vars(pipe).get("step")
    try:
        for name, target in originals.items():
            setattr(pipe, name, wrap(target, name))
        if hasattr(originals["dit"], "begin_chunk"):
            # The camera branch runs in dit.begin_chunk, outside dit's own call.
            pipe.dit.begin_chunk = wrap(originals["dit"].begin_chunk, "dit.begin_chunk")
        if step:
            pipe.step = wrap(pipe.step, "step")
        yield
    finally:
        for name, target in originals.items():
            setattr(pipe, name, target)
        if step:
            if own_step is None:
                vars(pipe).pop("step", None)
            else:
                pipe.step = own_step


def annotate_stages(pipe, prefix="bench_stage/"):
    """Label each stage call so profiler kernels can be attributed through launch correlation.

    pipe.step is labelled too; the innermost label wins, so `step` keeps only its own glue kernels.
    """
    import torch

    return wrapped_stages(pipe, lambda target, name: AnnotationProxy(target, prefix + name,
                                                                     torch.profiler.record_function), step=True)


def summarize_events(events):
    chunks = defaultdict(lambda: defaultdict(float))
    for name, chunk, start, end in events:
        chunks[name][chunk] += start.elapsed_time(end)
    return {name: {**sample_stats(values.values()), "chunk_indices": list(values)}
            for name, values in chunks.items()}


def measure_stages(pipe, generate, chunk):
    import torch

    events = []

    def timed(target, name):
        return TimingProxy(target, name, events, chunk, lambda: torch.cuda.Event(enable_timing=True))

    with wrapped_stages(pipe, timed):
        torch.cuda.synchronize()
        output = generate()
        torch.cuda.synchronize()
        del output
        return {"method": "CUDA events outside stage call objects, one extra generation",
                "stages": summarize_events(events),
                "scope": "Sum calls per chunk; DiT includes first-chunk clean pass. Refiner omits chunk 0. "
                         "dit.begin_chunk (camera branch) is timed separately from dit. "
                         "Excludes start/encoding, camera rays, scheduler and output conversion. "
                         "Events bracket warmed graph replay on the current stream, not graph capture bodies. "
                         "Diagnostic overhead is excluded from benchmark fps/VRAM; no stage repeatability claim."}
