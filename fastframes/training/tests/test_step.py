import pytest
import torch

from fastframes.training.kv_refinement import MasterAdamW
from fastframes.training.step import check_finite_loss, clip_and_step, is_due, step_if_finite


class CountingOptimizer:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


class ClippedModel:
    def __init__(self, norm):
        self.norm, self.max_norm = norm, None

    def clip_grad_norm_(self, max_norm):
        self.max_norm = max_norm
        return self.norm


def test_check_finite_loss():
    check_finite_loss(torch.tensor(1.5), "forward of update 1")
    with pytest.raises(RuntimeError) as info:
        check_finite_loss(torch.tensor(float("nan")), "forward of update 3")
    assert str(info.value) == "forward of update 3 failed on ranks [(0, \"FloatingPointError('loss nan')\")]"
    assert isinstance(info.value.__cause__, FloatingPointError)
    with pytest.raises(RuntimeError, match="critic loss inf"):
        check_finite_loss(torch.tensor(float("inf")), "critic update 2", "critic loss")


def test_step_if_finite():
    optimizer = CountingOptimizer()
    norm = torch.tensor(0.5)
    assert step_if_finite(optimizer, norm, "unused") is norm
    assert optimizer.steps == 1
    with pytest.raises(FloatingPointError, match="^non-finite gradient norm at update 4$"):
        step_if_finite(optimizer, torch.tensor(float("inf")), "non-finite gradient norm at update 4")
    assert optimizer.steps == 1


def test_clip_and_step():
    optimizer, model = CountingOptimizer(), ClippedModel(torch.tensor(2.0))
    assert clip_and_step(model, optimizer, 1.0, "unused").item() == 2.0
    assert model.max_norm == 1.0 and optimizer.steps == 1
    with pytest.raises(FloatingPointError, match="at critic update 7"):
        clip_and_step(ClippedModel(torch.tensor(float("nan"))), optimizer, 1.0,
                      "non-finite critic gradient norm at critic update 7")
    assert optimizer.steps == 1


def test_is_due():
    assert [step for step in range(1, 8) if is_due(step, 3)] == [3, 6]
    assert is_due(7, 3, final=True)
    assert not is_due(7, 3, final=False)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_master_adamw_skips_non_finite_gradients(dtype):
    torch.manual_seed(0)
    network = torch.nn.Linear(4, 2).to(dtype)
    optimizer = MasterAdamW(network, lr=1e-2, betas=(0.9, 0.99), weight_decay=0.0, max_grad_norm=1.0)
    network(torch.randn(3, 4, dtype=dtype)).square().sum().backward()
    before = [p.detach().clone() for p in network.parameters()]
    assert torch.isfinite(optimizer.step(1))
    assert any(not torch.equal(b, p) for b, p in zip(before, network.parameters()))

    optimizer.zero_grad()
    network(torch.randn(3, 4, dtype=dtype)).sum().backward()
    network.weight.grad[0, 0] = float("nan")
    before = [p.detach().clone() for p in network.parameters()]
    masters = [p.detach().clone() for p in optimizer.master.parameters()]
    with pytest.raises(FloatingPointError, match="^non-finite gradient norm at update 2$"):
        optimizer.step(2)
    assert all(torch.equal(b, p) for b, p in zip(before, network.parameters()))
    assert all(torch.equal(m, p) for m, p in zip(masters, optimizer.master.parameters()))
