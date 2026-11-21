"""Optional SageAttention 1.0.6 boundary for compiled, graph-captured inference."""
from importlib.metadata import version

import torch


def load_sage():
    if version("sageattention") != "1.0.6":
        raise RuntimeError("LB_ATTN=sage requires sageattention==1.0.6")
    from sageattention import sageattn

    return sageattn


@torch.library.custom_op("fastframes_lingbot::sage_attention", mutates_args=())
def sage_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    # v1 smooth_k subtracts the key mean in place; never mutate the rolling/text cache.
    return load_sage()(query, key.clone(), value, tensor_layout="NHD", is_causal=False,
                       sm_scale=query.shape[-1] ** -0.5)


@sage_attention.register_fake
def _(query, key, value):
    return torch.empty(query.shape, device=query.device, dtype=query.dtype)
