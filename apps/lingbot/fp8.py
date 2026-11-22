"""Opt-in dynamic E4M3 DiT GEMMs; no BF16 weight shadow is retained."""
import warnings

import torch
from torch import nn


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
    def __init__(self, weight, bias, *, rowwise):
        super().__init__()
        self.rowwise = rowwise
        with torch.no_grad():
            encoded, scale = quantize(weight, rowwise)
        self.register_buffer("weight", encoded)
        self.register_buffer("weight_scale", scale)
        self.register_buffer("bias", None if bias is None else bias.detach().to(torch.bfloat16))

    def forward(self, value):
        shape = value.shape[:-1]
        if value.numel() == 0:
            return value.new_empty((*shape, self.weight.shape[0]), dtype=torch.bfloat16)
        with torch.autocast("cuda", enabled=False):
            encoded, scale = quantize(value.reshape(-1, value.shape[-1]).to(torch.bfloat16), self.rowwise)
            output = torch._scaled_mm(encoded, self.weight.t(), scale_a=scale,
                                      scale_b=self.weight_scale.t() if self.rowwise else self.weight_scale,
                                      bias=self.bias, out_dtype=torch.bfloat16)
        return output.reshape(*shape, self.weight.shape[0])


@torch.no_grad()
def quantize_dit(model, *, scope="1"):
    if scope not in ("1", "ffn", "attention"):
        raise ValueError("LB_FP8 must be 0, 1, ffn, or attention")
    rowwise = rowwise_supported(model.patch_embedding.weight.device)
    fused = nn.ModuleList() if scope != "ffn" else None
    for block in model.blocks:
        targets = []
        if fused is not None:
            attention = block.self_attn
            projections = (attention.q, attention.k, attention.v)
            weight = torch.cat([projection.weight for projection in projections])
            bias = torch.cat([projection.bias for projection in projections])
            fused.append(FP8Linear(weight, bias, rowwise=rowwise))
            del weight, bias
            for projection in projections:
                projection.register_parameter("weight", None)
            targets.extend(((attention, "o"), (block.cross_attn, "q"), (block.cross_attn, "o")))
        if scope != "attention":
            targets.extend(((block.ffn, "0"), (block.ffn, "2")))
        for owner, name in targets:
            linear = getattr(owner, name)
            setattr(owner, name, FP8Linear(linear.weight, linear.bias, rowwise=rowwise))
    if fused is not None:
        model.add_module("_lb_fp8_qkv", fused)
    return fused
