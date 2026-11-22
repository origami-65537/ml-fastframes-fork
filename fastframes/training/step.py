#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The optimizer-update protocol shared by every training stage.

A non-finite loss is reported collectively (``RuntimeError`` on every rank, so that
no peer waits at a later collective). A non-finite gradient norm raises
``FloatingPointError`` locally: the norm is global, so every rank stops there.
"""
from __future__ import annotations

import torch

from .distributed import raise_if_any_failed

__all__ = ["check_finite_loss", "clip_and_step", "is_due", "step_if_finite"]


def check_finite_loss(loss: torch.Tensor, what: str, label: str = "loss") -> None:
    """Stop every rank when ``loss`` is not finite on any of them."""
    raise_if_any_failed(None if torch.isfinite(loss) else FloatingPointError(f"{label} {loss.item()}"), what)


def step_if_finite(optimizer, grad_norm: torch.Tensor, message: str) -> torch.Tensor:
    """``optimizer.step()`` unless the (global) ``grad_norm`` is not finite; returns ``grad_norm``."""
    if not torch.isfinite(grad_norm):  # the norm is global, so every rank stops here
        raise FloatingPointError(message)
    optimizer.step()
    return grad_norm


def clip_and_step(model, optimizer, max_norm: float, message: str) -> torch.Tensor:
    """Clip the global gradient norm of an FSDP ``model``, then :func:`step_if_finite`."""
    return step_if_finite(optimizer, model.clip_grad_norm_(max_norm), message)


def is_due(step: int, every: int, final: bool = False) -> bool:
    """Whether a periodic action (export, save) runs after update ``step``."""
    return step % every == 0 or final
