"""CPU coverage of the opaque attention boundary; kernels require CUDA."""
import unittest
from unittest.mock import patch

import torch

from apps.lingbot.sage_attention import sage_attention, use_sage


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

    def test_fp16_v_cache_matches_sage_conversion_and_centering(self):
        query = torch.randn(1, 7, 2, 64, dtype=torch.bfloat16)
        key = torch.randn(1, 11, 2, 64, dtype=query.dtype)
        original = key.clone()
        value = torch.randn_like(key)
        fp16_cache = torch.empty_like(value, dtype=torch.float16)
        fp16_cache[:, :6].copy_(value[:, :6])
        fp16_cache[:, 6:].copy_(value[:, 6:])
        centered = key - key.mean(dim=1, keepdim=True)
        q_int8, k_int8 = query.to(torch.int8), centered.to(torch.int8)
        expected_q_scale, expected_k_scale = torch.ones(1), torch.ones(2)

        def quantize(q, k, *, sm_scale, tensor_layout):
            self.assertEqual(sm_scale, 64 ** -0.5)
            self.assertEqual(tensor_layout, "NHD")
            torch.testing.assert_close(q, query, rtol=0, atol=0)
            torch.testing.assert_close(k, centered, rtol=0, atol=0)
            return q_int8, expected_q_scale, k_int8, expected_k_scale

        def attend(q, k, v, q_scale, k_scale, *, tensor_layout, output_dtype):
            self.assertEqual(tensor_layout, "NHD")
            self.assertIs(q, q_int8)
            self.assertIs(k, k_int8)
            self.assertIs(q_scale, expected_q_scale)
            self.assertIs(k_scale, expected_k_scale)
            self.assertEqual(v.data_ptr(), fp16_cache.data_ptr())
            torch.testing.assert_close(v, value.to(torch.float16), rtol=0, atol=0)
            return v[:, :q.shape[1]].to(output_dtype)

        with patch("apps.lingbot.sage_attention._fp16_v_kernels", return_value=(quantize, attend)):
            compiled = torch.compile(sage_attention, backend="eager", fullgraph=True)
            for precenter in (False, True):
                output = compiled(query, key, fp16_cache, precenter)
                torch.testing.assert_close(output, value[:, :7].to(torch.float16).to(query.dtype))
        torch.testing.assert_close(key, original, rtol=0, atol=0)

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


if __name__ == "__main__":
    unittest.main()
