"""CPU coverage of the opaque attention boundary; kernels require CUDA."""
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from apps.lingbot.dit import _attention_backend, attention
from apps.lingbot.sage_attention import sage_attention, sage_attention_hnd, use_sage


class SageAttentionTests(unittest.TestCase):
    def test_layout_and_cache_ownership(self):
        query = torch.randn(1, 7, 2, 64)
        key = torch.randn(1, 11, 2, 64)
        original = key.clone()
        value = torch.randn_like(key)

        def sage(q, k, v, **kwargs):
            expected = dict(tensor_layout="NHD", is_causal=False, sm_scale=64 ** -0.5)
            if kwargs.pop("smooth_k"):
                k.sub_(k.mean(dim=1, keepdim=True))
            self.assertEqual(kwargs, expected)
            return k[:, :q.shape[1]].clone()

        with patch("apps.lingbot.sage_attention.load_sage", return_value=sage):
            compiled = torch.compile(sage_attention, backend="eager", fullgraph=True)
            result = compiled(query, key, value)
            precentered = compiled(query, key, value, True)
        torch.testing.assert_close(result, precentered, rtol=0, atol=0)
        torch.testing.assert_close(key, original)

    def test_auto_prefers_sage_when_available(self):
        device = torch.device("cuda")
        with (patch("torch.cuda.get_device_capability", return_value=(12, 0)),
              patch("apps.lingbot.sage_attention.load_sage") as load):
            self.assertTrue(use_sage("auto", device, 128, eligible=True))
            self.assertFalse(use_sage("cudnn", device, 128, eligible=True))
            self.assertFalse(use_sage("auto", device, 128, eligible=False))
            load.assert_called_once()
            load.side_effect = RuntimeError("missing backend")
            self.assertFalse(use_sage("auto", device, 128, eligible=True))
            with self.assertRaisesRegex(RuntimeError, "missing backend"):
                use_sage("sage", device, 128, eligible=True)

    def test_fake_and_empty(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            query = torch.empty(1, 0, 2, 64)
            result = sage_attention(query, query, query)
            self.assertEqual(result.shape, query.shape)
            self.assertEqual(result.stride(), query.stride())

    def test_backend_error_propagates(self):
        query = torch.zeros(1, 1, 2, 64)
        with patch("apps.lingbot.sage_attention.load_sage", side_effect=RuntimeError("missing backend")):
            with self.assertRaisesRegex(RuntimeError, "missing backend"):
                sage_attention(query, query, query)

    def test_hnd_preserves_rounded_inputs_and_cache_ownership(self):
        query = torch.randn(1, 7, 2, 64, dtype=torch.bfloat16)
        key = torch.randn(1, 15, 2, 64, dtype=query.dtype)
        value = torch.randn_like(key)
        saved_key, saved_value = key.clone(), value.clone()
        # Use a live prefix of the rolling cache, not the whole backing allocation.
        live_key, live_value = key[:, :11], value[:, :11]
        centered = live_key - live_key.mean(dim=1, keepdim=True)
        q_scale, k_scale = object(), object()

        def quantize(q, k, *, sm_scale, tensor_layout):
            self.assertEqual(tensor_layout, "HND")
            self.assertEqual(sm_scale, 64 ** -0.5)
            torch.testing.assert_close(q, query.transpose(1, 2), rtol=0, atol=0)
            torch.testing.assert_close(k, centered.transpose(1, 2), rtol=0, atol=0)
            return q.contiguous(), q_scale, k.contiguous(), k_scale

        def forward(q, k, v, qs, ks, *, tensor_layout, output_dtype):
            self.assertEqual(tensor_layout, "HND")
            self.assertEqual(output_dtype, query.dtype)
            self.assertIs(qs, q_scale)
            self.assertIs(ks, k_scale)
            self.assertTrue(v.is_contiguous())
            torch.testing.assert_close(v, live_value.transpose(1, 2).half(), rtol=0, atol=0)
            return q.clone()

        with patch("apps.lingbot.sage_attention.load_sage_hnd", return_value=(quantize, forward)):
            compiled = torch.compile(attention, backend="aot_eager", fullgraph=True)
            result = compiled(query, live_key, live_value, backend="sage_hnd")
        torch.testing.assert_close(result, query, rtol=0, atol=0)
        self.assertTrue(result.transpose(1, 2).is_contiguous())
        torch.testing.assert_close(key, saved_key, rtol=0, atol=0)
        torch.testing.assert_close(value, saved_value, rtol=0, atol=0)

        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            q = torch.empty(query.shape, dtype=query.dtype)
            self.assertEqual(sage_attention_hnd(q, q, q).stride(), result.stride())

    def test_new_backend_selection_does_not_fall_back(self):
        cuda = torch.device("cuda")
        with (patch("torch.backends.cuda.is_flash_attention_available", return_value=True),
              patch("torch.cuda.get_device_capability", return_value=(12, 0)),
              patch("apps.lingbot.sage_attention.load_sage"),
              patch("apps.lingbot.sage_attention.load_sage_hnd") as load_hnd):
            self.assertEqual(_attention_backend("flash", cuda, 128, sage_eligible=True), "flash")
            self.assertEqual(_attention_backend("sage_hnd", cuda, 128, sage_eligible=True), "sage_hnd")
            self.assertEqual(_attention_backend("auto", cuda, 128, sage_eligible=True), "sage")
            self.assertEqual(_attention_backend("cudnn", cuda, 128, sage_eligible=False), "cudnn")
            with self.assertRaisesRegex(ValueError, "split-KV"):
                _attention_backend("flash", cuda, 128, sage_eligible=False, split=2)
            with self.assertRaisesRegex(ValueError, "one GPU"):
                _attention_backend("sage_hnd", cuda, 128, sage_eligible=False)
            with self.assertRaisesRegex(ValueError, "head dim"):
                _attention_backend("sage_hnd", cuda, 96, sage_eligible=True)
            load_hnd.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "PyTorch CUDA FlashAttention"):
            _attention_backend("flash", torch.device("cpu"), 128, sage_eligible=False)

    def test_flash_attention_compiled_non_square(self):
        generator = torch.Generator().manual_seed(7)
        query = torch.randn(1, 7, 2, 64, generator=generator, dtype=torch.bfloat16)
        key = torch.randn(1, 11, 2, 64, generator=generator, dtype=query.dtype)
        value = torch.randn(1, 11, 2, 64, generator=generator, dtype=query.dtype)
        q, k, v = (t.transpose(1, 2) for t in (query, key, value))
        with sdpa_kernel(SDPBackend.MATH):
            expected = F.scaled_dot_product_attention(q, k, v).transpose(1, 2)
        compiled = torch.compile(attention, backend="aot_eager", fullgraph=True)
        actual = compiled(query, key, value, backend="flash")
        torch.testing.assert_close(actual, expected, rtol=0.016, atol=0.008)


if __name__ == "__main__":
    unittest.main()
