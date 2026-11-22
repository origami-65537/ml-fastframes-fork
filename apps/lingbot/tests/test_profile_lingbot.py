"""CPU tests for the stage timers of scripts/profile_lingbot.py."""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from apps.lingbot import dit

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "profile_lingbot.py"


def load_profiler():
    saved = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location("profile_lingbot", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved
    return module


class Stage(torch.nn.Module):
    def forward(self, x):
        return x + 1


def test_pixel_frames_per_chunk_matches_original_rate():
    profiler = load_profiler()
    assert profiler.PIXEL_FRAMES_PER_CHUNK == 16
    assert profiler.PIXEL_FRAMES_PER_CHUNK / 0.5 == 4 / 0.5 * 4


def test_stage_timers_time_calls_and_restore_patches():
    profiler = load_profiler()
    stats = {"stage": 0.0, "attn": 0.0}
    forward, attention = Stage.forward, dit.attention
    with patch.object(torch.cuda, "synchronize"):
        with profiler.stage_timers(stats, {"stage": Stage}, time_attn=True):
            assert Stage()(torch.zeros(1)).item() == 1
            assert "__call__" in Stage.__dict__
            assert dit.attention is not attention
    assert stats["stage"] > 0
    assert Stage.forward is forward
    assert "__call__" not in Stage.__dict__
    assert Stage.__call__ is torch.nn.Module.__call__
    assert dit.attention is attention


def test_stage_timers_restore_on_error():
    profiler = load_profiler()
    forward = Stage.forward
    with pytest.raises(RuntimeError), patch.object(torch.cuda, "synchronize"):
        with profiler.stage_timers({"stage": 0.0}, {"stage": Stage}):
            raise RuntimeError
    assert Stage.forward is forward
    assert "__call__" not in Stage.__dict__
