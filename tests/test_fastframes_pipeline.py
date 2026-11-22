from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest
import torch

from fastframes.pipeline import FF_S, FastFrames


def test_encode_prompt_leaves_text_encoder_on_device_by_default():
    device = torch.device("cuda:0")
    dit = SimpleNamespace(patch_embedding=SimpleNamespace(weight=SimpleNamespace(device=device)))
    embeds = torch.empty(1, 2, 3)
    encoder = Mock(device=device, return_value={"prompt_embeds": embeds})
    pipe = FastFrames(FF_S, dit, None, None, encoder)
    with patch("torch.cuda.empty_cache") as empty_cache:
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
    pipe = FastFrames(FF_S, dit, None, None, encoder)
    pipe.encode_prompt("first")
    pipe.encode_prompt("second")
    assert encoder.mock_calls == [call.to(device), call(["first"]), call(["second"])]


@pytest.mark.parametrize("encode_fails", [False, True])
def test_encode_prompt_offloads_text_encoder_when_requested(encode_fails):
    device = torch.device("cuda:1")
    dit = SimpleNamespace(patch_embedding=SimpleNamespace(weight=SimpleNamespace(device=device)))
    embeds = torch.empty(1, 2, 3)
    mock = Mock()
    mock.encoder.return_value = {"prompt_embeds": embeds}
    pipe = FastFrames(FF_S, dit, None, None, mock.encoder, offload_text_encoder=True)
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
