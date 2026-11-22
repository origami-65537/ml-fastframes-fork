from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest
import torch

from fastframes import pipeline
from fastframes.pipeline import FF_S, FastFrames


@pytest.mark.parametrize(("device_name", "vram_gib", "requested", "offload"), [
    ("cuda:1", 16, None, True),
    ("cuda:1", 24, None, False),
    ("cuda:1", 16, False, False),
    ("cuda:1", 48, True, True),
    ("cpu", 0, None, False),
])
def test_from_pretrained_places_t5_on_the_resolved_device(device_name, vram_gib, requested, offload):
    device = torch.device(device_name)
    dit = SimpleNamespace(patch_embedding=SimpleNamespace(weight=SimpleNamespace(device=device)))
    with (patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(total_memory=vram_gib << 30))
          as properties, patch.object(pipeline, "build_wan", return_value=dit),
          patch.object(pipeline, "KVRefinementNetwork"), patch.object(pipeline, "PixelDecoder"),
          patch.object(pipeline, "TextEncoder") as text_encoder):
        pipe = FastFrames.from_pretrained(FF_S, ".", {}, {"stem_in.weight": torch.empty(1, 1)}, {},
                                          device=device, offload_text_encoder=requested)
    assert pipe.offload_text_encoder is offload
    text_encoder.return_value.to.assert_called_once_with("cpu" if offload else device, dtype=torch.bfloat16)
    if requested is None and device.type == "cuda":
        properties.assert_called_once_with(device)
    else:
        properties.assert_not_called()


def test_encode_prompt_leaves_text_encoder_on_device_by_default():
    device = torch.device("cuda:0")
    dit = SimpleNamespace(patch_embedding=SimpleNamespace(weight=SimpleNamespace(device=device)))
    embeds = torch.empty(1, 2, 3)
    encoder = Mock(device=device, return_value={"prompt_embeds": embeds})
    with (patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(total_memory=48 << 30)),
          patch("torch.cuda.empty_cache") as empty_cache):
        pipe = FastFrames(FF_S, dit, None, None, encoder)
        assert pipe.encode_prompt("first") is embeds
        assert pipe.encode_prompt("second") is embeds
    assert encoder.mock_calls == [call(["first"]), call(["second"])]
    empty_cache.assert_not_called()


def test_encode_prompt_moves_an_injected_cpu_encoder_to_the_gpu_once():
    device = torch.device("cuda:1")
    dit = SimpleNamespace(patch_embedding=SimpleNamespace(weight=SimpleNamespace(device=device)))
    encoder = Mock(device=torch.device("cpu"), return_value={"prompt_embeds": torch.empty(1, 2, 3)})

    def move(target):
        encoder.device = target

    encoder.to.side_effect = move
    pipe = FastFrames(FF_S, dit, None, None, encoder, offload_text_encoder=False)
    pipe.encode_prompt("first")
    pipe.encode_prompt("second")
    assert encoder.mock_calls == [call.to(device), call(["first"]), call(["second"])]


@pytest.mark.parametrize("encode_fails", [False, True])
@pytest.mark.parametrize("requested", [None, True])
def test_encode_prompt_offloads_text_encoder(encode_fails, requested):
    device = torch.device("cuda:1")
    dit = SimpleNamespace(patch_embedding=SimpleNamespace(weight=SimpleNamespace(device=device)))
    embeds = torch.empty(1, 2, 3)
    mock = Mock()
    mock.encoder.return_value = {"prompt_embeds": embeds}
    with patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(total_memory=16 << 30)):
        pipe = FastFrames(FF_S, dit, None, None, mock.encoder, offload_text_encoder=requested)
    if encode_fails:
        mock.encoder.side_effect = RuntimeError("encode failed")
    with patch("torch.cuda.empty_cache", mock.empty_cache):
        if encode_fails:
            with pytest.raises(RuntimeError, match="encode failed"):
                pipe.encode_prompt("prompt")
        else:
            assert pipe.encode_prompt("prompt") is embeds
    assert mock.mock_calls == [
        call.encoder.to(device), call.encoder(["prompt"]), call.encoder.cpu(), call.empty_cache(),
    ]
