from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest
import torch
from PIL import Image

from apps.lingbot import conditioning, pipeline, refiner


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


@pytest.fixture
def upstream():
    grid = torch.arange(1000)
    scheduler = SimpleNamespace(timesteps=grid, sigmas=grid, set_timesteps=Mock())
    dit = SimpleNamespace(patch_embedding=torch.nn.Linear(1, 1), blocks=[None], dim=1, num_heads=1)
    return SimpleNamespace(dit=dit, scheduler=scheduler, config=SimpleNamespace(num_train_timesteps=1000))


@pytest.mark.parametrize(
    ("world", "compile", "overrides", "expected"),
    [
        (1, True, {}, (True, True)),
        (1, False, {}, (False, True)),
        (1, True, {"LB_COMPILE_REFINER": "0", "LB_REFINER_DIRECT_KV": "0"}, (False, False)),
        (8, True, {}, (True, False)),
    ],
)
def test_refiner_defaults_respect_eager_world_and_overrides(world, compile, overrides, expected, monkeypatch, upstream):
    for flag in ("LB_COMPILE_REFINER", "LB_REFINER_DIRECT_KV"):
        monkeypatch.delenv(flag, raising=False)
    for flag, value in overrides.items():
        monkeypatch.setenv(flag, value)
    monkeypatch.setenv("LB_STEPS", "2")
    with (patch.object(pipeline, "LingBotDiT"), patch.object(pipeline, "KVRefiner") as refiner,
          patch.object(pipeline, "CapturedDecoder"), patch.object(pipeline, "SpatialDecoder"),
          patch.object(pipeline, "rope_frequencies")):
        pipeline.FastFramesLingBot(pipeline.MODELS["lingbot-1.3b"], upstream, None, None,
                                   world=world, compile=compile, graphs=False)
    args = refiner.call_args.kwargs
    assert (args["compile"], args["direct_kv"]) == expected


@pytest.mark.parametrize(("world", "graphs", "override", "expected"), [
    (1, True, None, False),
    (1, True, "1", True),
    (8, True, None, True),
    (8, True, "0", False),
    (8, False, None, False),
])
def test_dit_graphs_default_and_overrides(world, graphs, override, expected, monkeypatch, upstream):
    for flag in ("LB_GRAPHS_DIT", "LB_GRAPHS_REFINER", "LB_GRAPHS_DECODER", "LB_COMPILE_DIT"):
        monkeypatch.delenv(flag, raising=False)
    if override is not None:
        monkeypatch.setenv("LB_GRAPHS_DIT", override)
    decoder_cls = "CapturedDecoder" if world == 1 else "SpatialDecoder"
    with (patch.object(pipeline, "LingBotDiT") as dit, patch.object(pipeline, "KVRefiner") as refiner,
          patch.object(pipeline, decoder_cls) as decoder, patch.object(pipeline, "rope_frequencies")):
        pipeline.FastFramesLingBot(pipeline.MODELS["lingbot-1.3b"], upstream, None, None, world=world, graphs=graphs)
    assert dit.call_args.kwargs["graphs"] is expected
    assert dit.call_args.kwargs["compile"] is True
    assert refiner.call_args.kwargs["graphs"] is graphs
    assert decoder.call_args.kwargs["graphs"] is graphs


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
