from unittest.mock import Mock, call, patch

import pytest
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from apps.lingbot import conditioning, dit, pipeline, refiner
from fastframes.models.kv_refinement import KVRefinementNetwork


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
    assert mock.mock_calls == [call.dit.clear_graphs(), call.refiner.clear_graphs(), call.dit.model.cpu()]


@pytest.mark.parametrize("value_dtype", [torch.bfloat16, torch.float16])
def test_direct_refiner_commit_matches_staged_commit(monkeypatch, value_dtype):
    monkeypatch.setattr(refiner, "TOKENS", 4)
    monkeypatch.setattr(refiner, "FRAME_TOKENS", 2)
    r = refiner.KVRefiner.__new__(refiner.KVRefiner)
    r.layers, r.local_tokens, r.dim = 3, 4, 4
    r.cache_k = torch.zeros(3, 1, 8, 2, 2)
    r.cache_v = torch.zeros_like(r.cache_k, dtype=value_dtype)
    r.local = torch.empty(2, 3, 4, 4, dtype=torch.bfloat16)
    r.k, r.v = (torch.randn(3, 4, 4, dtype=torch.bfloat16) for _ in range(2))
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


def test_clean_pass_can_skip_last_attention_and_prediction(monkeypatch):
    monkeypatch.setattr(dit, "TOKENS", 2)
    monkeypatch.setattr(dit, "FRAME_TOKENS", 2)
    runner = dit.LingBotDiT.__new__(dit.LingBotDiT)
    runner.layers, runner.world, runner.split_kv = 2, 1, 1
    runner.heads, runner.head_dim, runner.dim = 2, 4, 8
    runner.local_tokens, runner.start, runner.end = 2, 0, 2
    runner.sage_self = runner.sage_cross = runner.sage_precenter = False
    runner.rope = torch.randn(2, 1, 2, 2)
    runner.time_e0 = torch.randn(1, 1, 6, 8)
    runner.cache_k = torch.zeros(2, 1, 4, 2, 4)
    runner.cache_v = torch.zeros_like(runner.cache_k)
    runner.text_k = torch.randn(2, 1, 3, 2, 4)
    runner.text_v = torch.randn_like(runner.text_k)
    runner.camera_params = torch.randn(2, 2, 1, 2, 8)
    runner.gather = [None, None]
    runner.result = Mock(key=torch.zeros_like(runner.cache_k), value=torch.zeros_like(runner.cache_v))
    runner._patch_run = Mock(return_value=torch.randn(1, 2, 8))
    runner._block_run = torch.compile(runner._block, backend="eager", fullgraph=True)
    runner._post_run = Mock()
    runner.qkv = [(torch.randn(24, 8), torch.randn(24)) for _ in range(2)]
    blocks = []
    for _ in range(2):
        block = nn.Module()
        block.modulation = torch.randn(1, 6, 8)
        block.norm1 = block.norm2 = block.norm3 = nn.LayerNorm(8)
        block.ffn = nn.Linear(8, 8)
        block.self_attn, block.cross_attn = nn.Module(), nn.Module()
        block.self_attn.norm_q = block.self_attn.norm_k = block.cross_attn.norm_q = nn.LayerNorm(8)
        block.self_attn.o = block.cross_attn.q = block.cross_attn.o = nn.Linear(8, 8)
        blocks.append(block)
    runner.model = Mock(blocks=blocks)

    def cpu_attention(q, k, v, *_, **__):
        return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                              v.transpose(1, 2)).transpose(1, 2)

    monkeypatch.setattr(dit, "attention", cpu_attention)
    runner._run(2, False, False)
    expected_k, expected_v = runner.cache_k.clone(), runner.cache_v.clone()
    runner.cache_k.zero_()
    runner.cache_v.zero_()
    runner._post_run.reset_mock()
    with patch.object(blocks[-1].self_attn.o, "forward", side_effect=AssertionError("unused projection")):
        runner._run(2, False, True)
    runner._post_run.assert_not_called()
    assert runner.result.key.count_nonzero() == runner.result.value.count_nonzero() == 0
    torch.testing.assert_close(runner.cache_k, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(runner.cache_v, expected_v, rtol=0, atol=0)


@torch.no_grad()
def test_refiner_sage_stem_preserves_independent_frame_batches(monkeypatch):
    net = KVRefinementNetwork(8, 2, 4, 1, d_a=8, n_stem=2, stem_n_heads=2)
    for block in net.stem_blocks:
        block.mod.lin.bias.fill_(1)
    tokens, embedding = torch.randn(4, 7, 8), torch.randn(4, 8)
    expected = net.compute_phi(tokens, embedding)
    calls = []

    def cpu_sage(q, k, v, precenter):
        calls.append((q.shape, precenter))
        return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                              v.transpose(1, 2)).transpose(1, 2)

    monkeypatch.setattr(refiner, "sage_attention", cpu_sage)
    output = net.compute_phi(tokens, embedding, attend=refiner._sage_stem_attention)
    assert calls == [((4, 7, 2, 4), True)] * 2
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
