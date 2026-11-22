from unittest.mock import Mock, call, patch

import pytest
import torch
from PIL import Image

from apps.lingbot import conditioning, dit, pipeline, refiner


@pytest.mark.parametrize("allocator_env", [None, "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"])
def test_only_configure_expandable_segments_by_default(monkeypatch, allocator_env):
    for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "LB_EXPANDABLE_SEGMENTS"):
        monkeypatch.delenv(name, raising=False)
    if allocator_env:
        monkeypatch.setenv(allocator_env, "expandable_segments:False")
    with (patch("torch.cuda.memory.get_allocator_backend", return_value="native"),
          patch("torch.cuda.memory._set_allocator_settings") as settings):
        pipeline._configure_allocator("cuda", 8)
        settings.assert_not_called()
        pipeline._configure_allocator("cuda", 1)
        if allocator_env:
            settings.assert_not_called()
        else:
            settings.assert_called_once_with("expandable_segments:True")


def test_repeated_image_encoding_moves_vae_back_to_device(monkeypatch):
    for name in ("HEIGHT", "WIDTH", "LATENT_HEIGHT", "LATENT_WIDTH"):
        monkeypatch.setattr(conditioning, name, 2)
    device = torch.device("cpu")
    image = Image.new("RGB", (2, 2))
    vae = Mock()
    vae.encode.return_value = [torch.zeros(16, 1, 2, 2)]
    for _ in range(2):
        encoded = conditioning.image_condition(vae, image, 1, device)
        assert encoded.shape == (20, 1, 2, 2)
    assert [entry[0] for entry in vae.mock_calls] == ["model.to", "encode", "model.cpu"] * 2
    assert vae.model.to.call_args == call(device)

    vae.encode.side_effect = RuntimeError("encode failed")
    with pytest.raises(RuntimeError, match="encode failed"):
        conditioning.image_condition(vae, image, 1, device)
    assert vae.mock_calls[-1] == call.model.cpu()


def test_graphs_cleared_before_park():
    pipe = pipeline.FastFramesLingBot.__new__(pipeline.FastFramesLingBot)
    mock = Mock()
    pipe.dit, pipe.refiner = mock.dit, mock.refiner
    pipe.dit.camera_params = torch.empty(1)
    pipe.cache_k = pipe.cache_v = torch.empty(1)
    with patch("torch.cuda.empty_cache"):
        pipe._park_cache()
    assert mock.mock_calls == [call.dit.clear_graphs(), call.refiner.clear_graphs(), call.dit.move_weights("cpu")]


@pytest.mark.parametrize("fp8", [False, True])
def test_packed_weights_replace_unused_projections_and_move_with_dit(fp8, monkeypatch):
    env = {"LB_ATTN": "cudnn", "LB_ATTN_CROSS": "cudnn", "LB_DIT_PACKED_ONLY": "1", "LB_FP8": str(int(fp8))}
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    model = torch.nn.Module()
    model.dim = 16
    model.num_heads = 1
    model.patch_embedding = torch.nn.Linear(16, 16)
    block = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([block])
    block.self_attn = torch.nn.Module()
    block.cross_attn = torch.nn.Module()
    for name in ("q", "k", "v", "o"):
        setattr(block.self_attn, name, torch.nn.Linear(16, 16, dtype=torch.bfloat16))
        setattr(block.cross_attn, name, torch.nn.Linear(16, 16, dtype=torch.bfloat16))
    block.ffn = torch.nn.Sequential(
        torch.nn.Linear(16, 32, dtype=torch.bfloat16), torch.nn.GELU(), torch.nn.Linear(32, 16, dtype=torch.bfloat16)
    )
    layers = block.cam_scale_layer, block.cam_shift_layer = tuple(torch.nn.Linear(16, 16) for _ in range(2))
    camera = torch.randn(2, 16)
    expected = torch.cat([layer(camera) for layer in layers], dim=-1)
    cache = torch.empty(1)
    with (patch.object(dit, "TOKENS", 2), patch.object(dit.LingBotDiT, "set_context"),
          patch.object(dit.LingBotDiT, "_time_bank", return_value={1: (torch.zeros(1, 1, 16),
                                                                       torch.zeros(1, 1, 6, 16))}),
          patch("apps.lingbot.fp8.rowwise_supported", return_value=True)):
        runner = dit.LingBotDiT(model, cache, cache, None, [1], compile=False, graphs=False)
    sources = (*layers, block.self_attn.q, block.self_attn.k, block.self_attn.v)
    assert all(layer.weight is None and layer.bias is None for layer in sources)
    assert block.cross_attn.k.weight is not None
    weight, bias = runner.cam[0]
    called = []
    original_to = torch.Tensor.to

    def track_to(tensor, *args, **kwargs):
        called.append(tensor.data_ptr())
        return original_to(tensor, *args, **kwargs)

    with patch.object(torch.Tensor, "to", track_to):
        runner.move_weights("cpu")
    assert {weight.data_ptr(), bias.data_ptr()} <= set(called)
    torch.testing.assert_close(torch.nn.functional.linear(camera, weight, bias), expected)


def test_direct_refiner_commit_matches_staged_commit(monkeypatch):
    monkeypatch.setattr(refiner, "TOKENS", 4)
    monkeypatch.setattr(refiner, "FRAME_TOKENS", 2)
    r = refiner.KVRefiner.__new__(refiner.KVRefiner)
    r.layers, r.local_tokens, r.dim = 3, 4, 4
    r.cache_k = torch.zeros(3, 1, 8, 2, 2)
    r.cache_v = torch.zeros_like(r.cache_k)
    r.local = torch.empty(2, 3, 4, 4)
    r.k, r.v = torch.randn(3, 4, 4), torch.randn(3, 4, 4)
    r.rope = torch.randn(4, 1, 1, 2)
    r.rotate, r.write_key = refiner.rope_real, refiner._write_refined_key

    def run_heads():
        for lo, hi in ((0, 2), (2, 3)):
            r._refine(lambda k, v, _: (k + 1, v - 1), lo, hi, None, 0, 4, 3)

    r.direct_kv = False
    run_heads()
    r._commit(*r.local, 3)
    expected_k, expected_v = r.cache_k.clone(), r.cache_v.clone()
    r.cache_k.zero_()
    r.cache_v.zero_()
    r.local = None
    r.direct_kv = True
    run_heads()
    torch.testing.assert_close(r.cache_k, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(r.cache_v, expected_v, rtol=0, atol=0)
