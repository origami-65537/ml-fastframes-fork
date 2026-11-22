"""CPU math/ownership checks; actual scaled GEMMs require CUDA."""
import copy
import unittest
import weakref
from unittest.mock import patch

import torch
from torch import nn

from apps.lingbot import dit
from apps.lingbot.fp8 import FP8Linear, quantize, quantize_dit, rowwise_supported


def emulated_scaled_mm(a, b, *, scale_a, scale_b, bias=None, out_dtype=None):
    output = (a.float() * scale_a) @ (b.float() * scale_b)
    if bias is not None:
        output = output + bias.float()
    return output.to(out_dtype)


class FP8Tests(unittest.TestCase):
    def test_quantized_matmul(self):
        generator = torch.Generator().manual_seed(0)
        value = torch.randn(2, 32, 128, generator=generator).to(torch.bfloat16)
        weight = torch.randn(96, 128, generator=generator).to(torch.bfloat16)
        bias = torch.randn(96, generator=generator).to(torch.bfloat16)
        reference = torch.nn.functional.linear(value, weight, bias).float()
        for rowwise in (True, False):
            with self.subTest(rowwise=rowwise):
                linear = FP8Linear(weight, bias, rowwise=rowwise)
                with patch("torch._scaled_mm", emulated_scaled_mm):
                    output = torch.compile(linear, fullgraph=True, backend="eager")(value)
                error = ((output.float() - reference).norm() / reference.norm()).item()
                print(f"FP8 {'rowwise' if rowwise else 'tensorwise'} relative L2 error vs BF16: {error:.6f}")
                self.assertLess(error, 0.06)
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertEqual(output.shape, reference.shape)
                self.assertEqual(linear.weight.dtype, torch.float8_e4m3fn)

    def test_zero_and_empty(self):
        for rowwise in (True, False):
            encoded, scale = quantize(torch.zeros(16, 32), rowwise)
            self.assertTrue(torch.isfinite(scale).all())
            self.assertEqual(encoded.float().count_nonzero().item(), 0)
            linear = FP8Linear(torch.zeros(16, 32), None, rowwise=rowwise)
            with patch("torch._scaled_mm", emulated_scaled_mm):
                self.assertEqual(linear(torch.zeros(1, 32)).count_nonzero().item(), 0)
            self.assertEqual(linear(torch.empty(0, 32)).shape, (0, 16))

    def test_real_scaled_mm_exports(self):
        for rowwise in (True, False):
            linear = FP8Linear(torch.zeros(96, 128), torch.zeros(96), rowwise=rowwise)
            exported = torch.export.export(linear, (torch.zeros(2, 32, 128),), strict=True)
            graphs = (module.graph for module in exported.graph_module.modules()
                      if isinstance(module, torch.fx.GraphModule))
            self.assertTrue(any(node.target == torch.ops.aten._scaled_mm.default
                                for graph in graphs for node in graph.nodes))

    def test_unsupported_device(self):
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            rowwise_supported("cpu")

    def test_scale_fallback(self):
        empty = torch.zeros(16, 16, dtype=torch.float8_e4m3fn)
        scale = torch.ones(16, 1)
        scalar = torch.ones(())
        with patch("torch.zeros", return_value=empty), patch("torch.ones", side_effect=[scale, scalar]):
            with patch("torch._scaled_mm", side_effect=[RuntimeError("rowwise unsupported"), empty]) as mm:
                with self.assertWarnsRegex(UserWarning, "using tensorwise scales"):
                    self.assertFalse(rowwise_supported("cuda"))
                self.assertEqual(mm.call_count, 2)

    def test_selected_weights_released(self):
        model = nn.Module()
        model.patch_embedding = nn.Linear(16, 16)
        block = nn.Module()
        block.self_attn = nn.Module()
        block.cross_attn = nn.Module()
        for name in ("q", "k", "v", "o"):
            setattr(block.self_attn, name, nn.Linear(16, 16))
            setattr(block.cross_attn, name, nn.Linear(16, 16))
        block.ffn = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16))
        model.blocks = nn.ModuleList([block])
        for scope, has_attention, has_ffn in (("ffn", False, True), ("attention", True, False)):
            with self.subTest(scope=scope), patch("apps.lingbot.fp8.rowwise_supported", return_value=True):
                scoped = copy.deepcopy(model)
                self.assertEqual(quantize_dit(scoped, scope=scope) is not None, has_attention)
                self.assertEqual(isinstance(scoped.blocks[0].self_attn.o, FP8Linear), has_attention)
                self.assertEqual(isinstance(scoped.blocks[0].ffn[0], FP8Linear), has_ffn)
        weights = [weakref.ref(module.weight) for module in
                   (block.self_attn.q, block.self_attn.k, block.self_attn.v, block.self_attn.o,
                    block.cross_attn.q, block.cross_attn.o, block.ffn[0], block.ffn[2])]
        untouched = (block.cross_attn.k.weight, block.cross_attn.v.weight, model.patch_embedding.weight)
        with patch("apps.lingbot.fp8.rowwise_supported", return_value=True):
            fused = quantize_dit(model)
        self.assertTrue(all(weight() is None for weight in weights))
        self.assertIs(fused, model._lb_fp8_qkv)
        self.assertIsNone(block.self_attn.q.weight)
        self.assertTrue(all(weight.dtype == torch.float32 for weight in untouched))
        self.assertEqual(sum(isinstance(module, FP8Linear) for module in model.modules()), 6)

    def test_block_applies_self_attention_output_scales(self):
        block = nn.Module()
        block.modulation = torch.zeros(1, 6, 16)
        block.modulation[:, 2] = 1
        block.norm1 = block.norm2 = block.norm3 = nn.LayerNorm(16)
        block.ffn = nn.Linear(16, 16)
        block.self_attn = a = nn.Module()
        block.cross_attn = cross = nn.Module()
        a.norm_q = a.norm_k = cross.norm_q = nn.LayerNorm(16)
        a.o = nn.Linear(16, 16)
        cross.q = cross.o = nn.Linear(16, 16)
        runner = dit.LingBotDiT.__new__(dit.LingBotDiT)
        runner.world = runner.heads = runner.split_kv = 1
        runner.dim = runner.head_dim = 16
        runner.local_tokens = runner.end = 2
        runner.start = 0
        runner.attn_self = runner.attn_cross = "cudnn"
        runner.sage_precenter = False
        runner.time_e0 = torch.zeros(1, 1, 6, 16)
        runner.rope = torch.zeros(2, 1, 8, 2)
        runner.rope[..., 0] = 1
        generator = torch.Generator().manual_seed(4)
        value = torch.randn(1, 2, 16, generator=generator)
        cache = torch.zeros(1, 2, 1, 16)
        args = (block, value, cache.clone(), cache.clone(), cache, cache, torch.zeros(2, 1, 2, 16),
                torch.randn(48, 16, generator=generator), torch.zeros(48), None, None, None, 1, False)
        with (patch.object(dit, "TOKENS", 2), patch.object(dit, "FRAME_TOKENS", 2),
              patch.object(dit, "attention", side_effect=lambda q, k, v, *_, **__: v),
              patch("torch._scaled_mm", side_effect=emulated_scaled_mm) as mm):
            expected = runner._block(*args)
            a.o = FP8Linear(a.o.weight, a.o.bias, rowwise=True)
            actual = runner._block(*args)
        mm.assert_called_once()
        self.assertLess((actual - expected).norm() / expected.norm(), 0.06)


if __name__ == "__main__":
    unittest.main()
