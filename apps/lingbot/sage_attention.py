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


@cache
def _fp16_v_kernels(head_dim):
    load_sage()
    from sageattention.core import attn_false, attn_h96_false, per_block_int8, per_block_int8_hd96

    return (per_block_int8_hd96, attn_h96_false) if head_dim == 96 else (per_block_int8, attn_false)


@torch.library.custom_op("fastframes_lingbot::sage_attention", mutates_args=())
def sage_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                   precenter: bool = False) -> torch.Tensor:
    # v1 smooth_k subtracts the key mean in place; never mutate the rolling/text cache.
    prepared = key - key.mean(dim=1, keepdim=True) if precenter else key.clone()
    scale = query.shape[-1] ** -0.5
    if value.dtype == torch.float16 and query.dtype != value.dtype:
        # Mirror Sage v1's noncausal path, bypassing only its full-cache BF16 -> FP16 V cast.
        if not precenter:
            prepared.sub_(prepared.mean(dim=1, keepdim=True))
        quantize, attend = _fp16_v_kernels(query.shape[-1])
        q, k, q_scale, k_scale = quantize(query, prepared, sm_scale=scale, tensor_layout="NHD")
        return attend(q, k, value, q_scale, k_scale, tensor_layout="NHD", output_dtype=query.dtype)
    return load_sage()(query, prepared, value, tensor_layout="NHD", is_causal=False, smooth_k=not precenter,
                       sm_scale=scale)


@sage_attention.register_fake
def _(query, key, value, precenter=False):
    return torch.empty(query.shape, device=query.device, dtype=query.dtype)
