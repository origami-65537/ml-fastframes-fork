"""CPU coverage of the opaque attention boundary; kernels require CUDA."""
import unittest
from unittest.mock import patch

import torch

from apps.lingbot.sage_attention import sage_attention


class SageAttentionTests(unittest.TestCase):
    def test_layout_and_cache_ownership(self):
        query = torch.randn(1, 7, 2, 64)
        key = torch.randn(1, 11, 2, 64)
        original = key.clone()
        value = torch.randn_like(key)

        def sage(q, k, v, **kwargs):
            self.assertEqual(kwargs, dict(tensor_layout="NHD", is_causal=False, sm_scale=64 ** -0.5))
            k.zero_()
            return q.clone()

        with patch("apps.lingbot.sage_attention.load_sage", return_value=sage):
            result = torch.compile(sage_attention, backend="eager", fullgraph=True)(query, key, value)
        torch.testing.assert_close(result, query)
        torch.testing.assert_close(key, original)

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
