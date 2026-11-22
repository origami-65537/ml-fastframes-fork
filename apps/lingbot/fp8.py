"""Opt-in dynamic E4M3 DiT GEMMs and BF16 diagnostics."""
import warnings

import torch
import torch.nn.functional as F
from torch import nn

_FFN = {"ffn_up", "ffn_down"}
_ATTENTION = {"self_qkv", "self_out", "cross_q", "cross_out"}
_PROJECTIONS = _FFN | _ATTENTION
_SCOPES = {"1": _PROJECTIONS, "ffn": _FFN, "attention": _ATTENTION}


def quantize(value, rowwise):
    value = value.float()
    maximum = value.abs().amax(dim=-1, keepdim=True) if rowwise else value.abs().amax()
    limit = torch.finfo(torch.float8_e4m3fn).max
    scale = maximum.clamp_min(1e-12) / limit
    encoded = (value / scale).clamp(-limit, limit).to(torch.float8_e4m3fn)
    return encoded, scale


@torch.no_grad()
def rowwise_supported(device):
    if torch.device(device).type != "cuda":
        raise RuntimeError("LB_FP8=1 requires CUDA torch._scaled_mm support")
    value = torch.zeros(16, 16, device=device, dtype=torch.float8_e4m3fn)
    scale = torch.ones(16, 1, device=device, dtype=torch.float32)
    bias = torch.zeros(16, device=device, dtype=torch.bfloat16)
    try:
        torch._scaled_mm(value, value.t(), scale_a=scale, scale_b=scale.t(), bias=bias, out_dtype=torch.bfloat16)
        return True
    except torch.cuda.OutOfMemoryError:
        raise
    except (RuntimeError, NotImplementedError) as error:
        scalar = torch.ones((), device=device, dtype=torch.float32)
        torch._scaled_mm(value, value.t(), scale_a=scalar, scale_b=scalar, bias=bias, out_dtype=torch.bfloat16)
        warnings.warn(f"LB_FP8: rowwise _scaled_mm unavailable; using tensorwise scales ({error})", stacklevel=2)
        return False


class FP8Linear(nn.Module):
    def __init__(self, weight, bias, *, rowwise, mode="scaled", direct_input=False):
        super().__init__()
        self.rowwise, self.mode = rowwise, mode
        self.direct_input = direct_input
        with torch.no_grad():
            if mode == "activation":
                encoded, scale = weight.detach().to(torch.bfloat16), None
            else:
                encoded, scale = quantize(weight, rowwise)
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
                encoded, scale = quantize(flat if self.direct_input else flat.to(torch.bfloat16), self.rowwise)
            if self.mode == "scaled":
                output = torch._scaled_mm(encoded, self.weight.t(), scale_a=scale,
                                          scale_b=self.weight_scale.t() if self.rowwise else self.weight_scale,
                                          bias=self.bias, out_dtype=torch.bfloat16)
            else:
                activation = flat.to(torch.bfloat16) if self.mode == "weight" else (encoded.float() * scale).bfloat16()
                output = F.linear(activation, self.weight, self.bias)
        return output.reshape(*shape, self.weight.shape[0])


@torch.no_grad()
def quantize_dit(model, *, scope="1", mode="scaled", direct_input=False):
    selected = _SCOPES.get(scope, set(scope.split(",")))
    if not selected <= _PROJECTIONS:
        raise ValueError("LB_FP8 must be 0, 1, ffn, attention, or a comma-separated list of "
                         f"{', '.join(sorted(_PROJECTIONS))}")
    if mode not in ("scaled", "weight", "activation", "both"):
        raise ValueError("LB_FP8_DIAG must be scaled, weight, activation, or both")
    rowwise = rowwise_supported(model.patch_embedding.weight.device)
    options = dict(rowwise=rowwise, mode=mode, direct_input=direct_input)
    fused = nn.ModuleList() if "self_qkv" in selected else None
    for block in model.blocks:
        attention = block.self_attn
        if fused is not None:
            projections = (attention.q, attention.k, attention.v)
            weight = torch.cat([projection.weight for projection in projections])
            bias = torch.cat([projection.bias for projection in projections])
            fused.append(FP8Linear(weight, bias, **options))
            del weight, bias
            for projection in projections:
                projection.register_parameter("weight", None)
        targets = (("self_out", attention, "o"), ("cross_q", block.cross_attn, "q"),
                   ("cross_out", block.cross_attn, "o"), ("ffn_up", block.ffn, "0"), ("ffn_down", block.ffn, "2"))
        for projection, owner, name in targets:
            if projection in selected:
                linear = getattr(owner, name)
                setattr(owner, name, FP8Linear(linear.weight, linear.bias, **options))
    if fused is not None:
        model.add_module("_lb_fp8_qkv", fused)
    return fused
