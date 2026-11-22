"""Opt-in dynamic INT8 W8A8 DiT GEMMs and BF16 diagnostics.

Mirrors fp8.py: INT8 has uniform quantization levels, unlike E4M3's
3-bit mantissa, which the FP8 bench isolation identified as the cause
of the quality loss (weight rounding, not activations). torch._int_mm
runs ~3x faster than BF16 cutlass GEMMs at LingBot shapes on sm_120.
"""
import warnings

import torch
import torch.nn.functional as F
from torch import nn

_FFN = {"ffn_up", "ffn_down"}
_ATTENTION = {"self_qkv", "self_out", "cross_q", "cross_out"}
_PROJECTIONS = _FFN | _ATTENTION
_SCOPES = {"1": _PROJECTIONS, "ffn": _FFN, "attention": _ATTENTION}

_LIMIT = 127.0


def quantize(value):
    """Per-row symmetric INT8: returns (m, k) int8 and (m, 1) fp32 scale."""
    value = value.float()
    maximum = value.abs().amax(dim=-1, keepdim=True)
    scale = maximum.clamp_min(1e-12) / _LIMIT
    encoded = (value / scale).round_().clamp_(-_LIMIT, _LIMIT).to(torch.int8)
    return encoded, scale


@torch.no_grad()
def int8_supported(device):
    if torch.device(device).type != "cuda":
        raise RuntimeError("LB_INT8 requires CUDA torch._int_mm support")
    value = torch.zeros(64, 64, device=device, dtype=torch.int8)
    try:
        torch._int_mm(value, value.t())
        return True
    except torch.cuda.OutOfMemoryError:
        raise
    except (RuntimeError, NotImplementedError) as error:
        raise RuntimeError(f"LB_INT8 requires torch._int_mm support ({error})")


class Int8Linear(nn.Module):
    """W8A8: per-output-channel int8 weights, per-token int8 activations."""

    def __init__(self, weight, bias, *, mode="scaled", direct_input=False):
        super().__init__()
        self.mode, self.direct_input = mode, direct_input
        with torch.no_grad():
            if mode == "activation":
                encoded, scale = weight.detach().to(torch.bfloat16), None
            else:
                encoded, scale = quantize(weight)  # (out, 1) scale per output channel
                if mode != "scaled":
                    encoded, scale = (encoded.float() * scale).to(torch.bfloat16), None
        self.register_buffer("weight", encoded)
        self.register_buffer("weight_scale", scale)
        self.register_buffer("bias", None if bias is None else bias.detach().to(torch.bfloat16))

    def forward(self, value):
        shape = value.shape[:-1]
        if value.numel() == 0:
            return value.new_empty((*shape, self.weight.shape[0]), dtype=torch.bfloat16)
        with torch.autocast("cuda", enabled=False):
            flat = value.reshape(-1, value.shape[-1])
            if self.mode != "weight":
                encoded, scale = quantize(flat if self.direct_input else flat.to(torch.bfloat16))
            if self.mode == "scaled":
                acc = torch._int_mm(encoded, self.weight.t()).float()
                output = acc * scale * self.weight_scale.t()
                if self.bias is not None:
                    output += self.bias.float()
                output = output.to(torch.bfloat16)
            else:
                activation = flat.to(torch.bfloat16) if self.mode == "weight" else (encoded.float() * scale).bfloat16()
                output = F.linear(activation, self.weight, self.bias)
        return output.reshape(*shape, self.weight.shape[0])


@torch.no_grad()
def quantize_dit(model, *, scope="1", mode="scaled", direct_input=False):
    selected = _SCOPES.get(scope, set(scope.split(",")))
    if not selected <= _PROJECTIONS:
        raise ValueError("LB_INT8 must be 0, 1, ffn, attention, or a comma-separated list of "
                         f"{', '.join(sorted(_PROJECTIONS))}")
    if mode not in ("scaled", "weight", "activation", "both"):
        raise ValueError("LB_INT8_DIAG must be scaled, weight, activation, or both")
    int8_supported(model.patch_embedding.weight.device)
    options = dict(mode=mode, direct_input=direct_input)
    fused = nn.ModuleList() if "self_qkv" in selected else None
    for block in model.blocks:
        attention = block.self_attn
        if fused is not None:
            projections = (attention.q, attention.k, attention.v)
            weight = torch.cat([projection.weight for projection in projections])
            bias = torch.cat([projection.bias for projection in projections])
            fused.append(Int8Linear(weight, bias, **options))
            del weight, bias
            for projection in projections:
                projection.register_parameter("weight", None)
        targets = (("self_out", attention, "o"), ("cross_q", block.cross_attn, "q"),
                   ("cross_out", block.cross_attn, "o"), ("ffn_up", block.ffn, "0"), ("ffn_down", block.ffn, "2"))
        for projection, owner, name in targets:
            if projection in selected:
                linear = getattr(owner, name)
                setattr(owner, name, Int8Linear(linear.weight, linear.bias, **options))
    if fused is not None:
        model.add_module("_lb_int8_qkv", fused)
    return fused
