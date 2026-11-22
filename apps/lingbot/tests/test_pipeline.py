from types import SimpleNamespace
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


def test_cache_frames_must_keep_sink_frames():
    minimum = pipeline.SINK_FRAMES + pipeline.CHUNK
    pipeline._check_cache_frames(minimum)
    with pytest.raises(ValueError, match="LB_CACHE_FRAMES"):
        pipeline._check_cache_frames(minimum - 1)


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


@torch.no_grad()
def test_sliced_camera_matches_full_bank_and_tracks_storage(monkeypatch):
    for name, size in (("CHUNK", 2), ("GRID_H", 1), ("GRID_W", 2), ("TOKENS", 4)):
        monkeypatch.setattr(dit, name, size)
    runner = dit.LingBotDiT.__new__(dit.LingBotDiT)
    m = runner.model = torch.nn.Module()
    m.patch_embedding_wancamctrl = torch.nn.Linear(384 * 4, 8)
    m.c2ws_hidden_states_layer1 = torch.nn.Linear(8, 8)
    m.c2ws_hidden_states_layer2 = torch.nn.Linear(8, 8)
    m.blocks = torch.nn.ModuleList(torch.nn.Module() for _ in range(3))
    for block in m.blocks:
        block.cam_injector_layer1 = torch.nn.Linear(8, 8)
        block.cam_injector_layer2 = torch.nn.Linear(8, 8)
    runner.cam = [(torch.randn(16, 8), torch.randn(16)) for _ in m.blocks]
    runner.start, runner.end = 1, 3
    runner.camera = torch.randn(1, 384, 2, 2, 4, dtype=torch.bfloat16)
    runner.camera_params = torch.empty(3, 2, 1, 2, 8, dtype=torch.bfloat16)
    runner._camera_project_run = torch.compile(runner._project_camera, backend="aot_eager", fullgraph=True)
    write_layer = torch.compile(dit._write_camera_layer, backend="aot_eager", fullgraph=True)
    destinations = []

    def write_slice(camera, block, params, dst):
        destinations.append(dst)
        write_layer(camera, block, params, dst)

    runner._camera_layer_run = write_slice
    with torch.autocast("cpu", dtype=torch.bfloat16):
        runner._prepare_camera()
        old_storage = runner.camera_params.detach()
        expected = old_storage.clone()
        runner.camera_params.data = torch.zeros_like(old_storage)
        runner._prepare_camera_sliced()
    assert [dst.data_ptr() for dst in destinations] == [layer.data_ptr() for layer in runner.camera_params]
    assert [dst.numel() for dst in destinations] == [expected[0].numel()] * len(m.blocks)
    torch.testing.assert_close(old_storage, expected, rtol=0, atol=0)
    torch.testing.assert_close(runner.camera_params, expected, rtol=0, atol=0)


@pytest.fixture
def upstream():
    grid = torch.arange(1000)
    scheduler = SimpleNamespace(timesteps=grid, sigmas=grid, set_timesteps=Mock())
    dit = SimpleNamespace(patch_embedding=torch.nn.Linear(1, 1), blocks=[None], dim=1, num_heads=1)
    return SimpleNamespace(dit=dit, scheduler=scheduler, config=SimpleNamespace(num_train_timesteps=1000))


@pytest.mark.parametrize(("world", "compile", "env", "expected"), [
    (1, True, {}, True),
    (1, False, {}, False),
    (8, True, {}, False),
    (1, True, {"LB_COMPILE_DIT": "0"}, False),
    (1, False, {"LB_COMPILE_DIT": "1"}, True),
    (1, True, {"LB_DIT_CAMERA_SLICED": "0"}, False),
])
def test_sliced_camera_defaults_to_single_gpu_with_compiled_dit(world, compile, env, expected, monkeypatch, upstream):
    for flag in ("LB_DIT_CAMERA_SLICED", "LB_COMPILE_DIT"):
        monkeypatch.delenv(flag, raising=False)
    for flag, value in env.items():
        monkeypatch.setenv(flag, value)
    with (patch.object(pipeline, "LingBotDiT") as runner, patch.object(pipeline, "KVRefiner"),
          patch.object(pipeline, "CapturedDecoder"), patch.object(pipeline, "SpatialDecoder"),
          patch.object(pipeline, "rope_frequencies")):
        pipeline.FastFramesLingBot(pipeline.MODELS["lingbot-1.3b"], upstream, None, None,
                                   world=world, compile=compile, graphs=False)
    assert runner.call_args.kwargs["camera_sliced"] is expected


def test_explicitly_sliced_camera_requires_dit_compilation():
    with pytest.raises(ValueError, match="requires DiT compilation"):
        dit.LingBotDiT(None, None, None, None, (), compile=False, camera_sliced=True)


@pytest.mark.parametrize(
    "tma,epilogue,expected",
    [
        ("0", "0", {}),
        ("1", "0", {"triton.enable_persistent_tma_matmul": True}),
        ("0", "1", {"max_epilogue_benchmarked_choices": 4}),
        ("1", "1", {"triton.enable_persistent_tma_matmul": True, "max_epilogue_benchmarked_choices": 4}),
    ],
)
def test_gemm_options_only_apply_to_the_dit_block(monkeypatch, tma, epilogue, expected):
    from torch._inductor import list_mode_options

    monkeypatch.setenv("LB_DIT_TMA_GEMM", tma)
    monkeypatch.setenv("LB_DIT_GEMM_EPILOGUE_TUNE", epilogue)
    monkeypatch.setenv("LB_FP8", "0")
    monkeypatch.setenv("LB_ATTN", "cudnn")
    monkeypatch.setenv("LB_ATTN_CROSS", "cudnn")
    monkeypatch.setattr(dit, "TOKENS", 4)
    m = torch.nn.Module()
    m.dim, m.num_heads = 8, 1
    m.patch_embedding = torch.nn.Linear(8, 8)
    block = torch.nn.Module()
    m.blocks = torch.nn.ModuleList([block])
    block.self_attn = torch.nn.Module()
    block.self_attn.q = torch.nn.Linear(8, 8)
    block.self_attn.k = torch.nn.Linear(8, 8)
    block.self_attn.v = torch.nn.Linear(8, 8)
    block.cam_scale_layer = torch.nn.Linear(8, 8)
    block.cam_shift_layer = torch.nn.Linear(8, 8)
    time_bank = {1: (torch.empty(1, 1, 8), torch.empty(1, 1, 6, 8))}

    with (patch.object(dit.LingBotDiT, "set_context"),
          patch.object(dit.LingBotDiT, "_time_bank", return_value=time_bank),
          patch.object(torch, "compile") as compile_fn):
        dit.LingBotDiT(m, None, None, None, (1,), graphs=False, camera_sliced=True)

    mode = "max-autotune-no-cudagraphs"
    calls = {c.args[0].__name__: c.kwargs for c in compile_fn.call_args_list}
    block_config = calls.pop("_block")
    base_config = dict(dynamic=False, fullgraph=True, mode=mode)
    assert set(calls) == {"_project_camera", "_write_camera_layer", "_patch", "_post"}
    assert all(kwargs == base_config for kwargs in calls.values())
    if expected:
        options = {**list_mode_options(mode), **expected}
        assert block_config == dict(dynamic=False, fullgraph=True, options=options)
        torch.compile(torch.neg, **block_config)
    else:
        assert block_config == base_config


@pytest.mark.parametrize("flag", ["LB_DIT_TMA_GEMM", "LB_DIT_GEMM_EPILOGUE_TUNE"])
def test_dit_gemm_options_require_compilation(monkeypatch, flag):
    monkeypatch.setenv(flag, "1")
    with pytest.raises(ValueError, match=flag):
        dit.LingBotDiT(None, None, None, None, (), compile=False)


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
