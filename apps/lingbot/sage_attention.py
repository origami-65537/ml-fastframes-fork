"""Optional SageAttention 1.0.6 boundary for compiled, graph-captured inference."""
from functools import cache
from importlib.metadata import PackageNotFoundError, version

import torch


@cache
def load_sage():
    try:
        installed = version("sageattention")
    except PackageNotFoundError:
        installed = None
    if installed != "1.0.6":
        raise RuntimeError("LB_ATTN=sage requires sageattention==1.0.6")
    from sageattention import sageattn

    return sageattn


@cache
def load_sage_hnd():
    # Use the same pinned v1 kernels while combining the value cast and layout conversion.
    load_sage()
    from sageattention.attn_qk_int8_per_block import forward
    from sageattention.quant_per_block import per_block_int8

    return per_block_int8, forward


def use_sage(backend, device, head_dim, *, eligible):
    if backend not in ("auto", "sage") or not eligible:
        return False
    supported = (device.type == "cuda" and torch.version.hip is None and head_dim in (64, 96, 128)
                 and torch.cuda.get_device_capability(device)[0] >= 8)
    if not supported:
        if backend == "sage":
            raise RuntimeError("SageAttention requires supported head dimensions and NVIDIA Ampere or newer")
        return False
    try:
        load_sage()
    except (RuntimeError, ImportError):
        if backend == "sage":
            raise
        return False
    return True


@torch.library.custom_op("fastframes_lingbot::sage_attention", mutates_args=())
def sage_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                   precenter: bool = False) -> torch.Tensor:
    # v1 smooth_k subtracts the key mean in place; never mutate the rolling/text cache.
    prepared = key - key.mean(dim=1, keepdim=True) if precenter else key.clone()
    return load_sage()(query, prepared, value, tensor_layout="NHD", is_causal=False, smooth_k=not precenter,
                       sm_scale=query.shape[-1] ** -0.5)


@sage_attention.register_fake
def _(query, key, value, precenter=False):
    return torch.empty(query.shape, device=query.device, dtype=query.dtype)


@torch.library.custom_op("fastframes_lingbot::sage_attention_hnd", mutates_args=())
def sage_attention_hnd(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Run the Sage v1 kernels with contiguous per-head INT8 Q/K and FP16 V."""
    quantize, forward = load_sage_hnd()
    centered = key - key.mean(dim=1, keepdim=True)
    q, q_scale, k, k_scale = quantize(query.transpose(1, 2), centered.transpose(1, 2),
                                     sm_scale=query.shape[-1] ** -0.5, tensor_layout="HND")
    v = value.transpose(1, 2).to(dtype=torch.float16, memory_format=torch.contiguous_format)
    return forward(q, k, v, q_scale, k_scale, tensor_layout="HND", output_dtype=query.dtype).transpose(1, 2)


@sage_attention_hnd.register_fake
def _(query, key, value):
    return torch.empty_like(query.transpose(1, 2), memory_format=torch.contiguous_format).transpose(1, 2)
