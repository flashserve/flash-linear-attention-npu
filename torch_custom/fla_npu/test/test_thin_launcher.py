"""M1 thin-launcher tests: dispatch gating and ctypes parity.

Only meaningful on an NPU host with the optional extension built
(FLA_NPU_BUILD_THIN=1). Skipped elsewhere.
"""
from __future__ import annotations

import os
import unittest

import pytest
import torch


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401

        return bool(torch.npu.is_available())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _npu_available(), reason="NPU is required")


def _have_thin() -> bool:
    try:
        import fla_npu._C_thin  # noqa: F401

        return True
    except Exception:
        return False


def _state_pair(batch=8):
    num_value_heads, dim = 16, 128
    gap, offset = 16384, 12288
    inner = num_value_heads * dim * dim
    block_stride = inner + gap

    def make():
        backing = torch.empty(batch * block_stride * 4, dtype=torch.int8, device="npu")
        typed = backing.view(torch.float32)
        state = torch.as_strided(
            typed,
            size=(batch, num_value_heads, dim, dim),
            stride=(block_stride, dim * dim, dim, 1),
            storage_offset=offset,
        )
        state.zero_()
        return state, backing

    def norm(t):
        return torch.nn.functional.normalize(t, p=2, dim=-1)

    query = norm(torch.randn(batch, 8, dim, device="npu")).to(torch.bfloat16)
    key = norm(torch.randn(batch, 8, dim, device="npu")).to(torch.bfloat16)
    value = torch.randn(batch, num_value_heads, dim, dtype=torch.bfloat16, device="npu")
    beta = torch.rand(batch, num_value_heads, dtype=torch.bfloat16, device="npu")
    g = torch.rand(batch, num_value_heads, dtype=torch.float32, device="npu")
    actual_seq_lengths = torch.tensor([0] + [1] * batch, dtype=torch.int32, device="npu")
    ssm_state_indices = torch.arange(batch, dtype=torch.int32, device="npu")
    return {
        "query": query,
        "key": key,
        "value": value,
        "beta": beta,
        "g": g,
        "scale": dim ** -0.5,
        "actual_seq_lengths": actual_seq_lengths,
        "ssm_state_indices": ssm_state_indices,
        "make_state": make,
    }


@pytest.mark.skipif(not _have_thin(), reason="thin launcher extension not built")
class TestThinLauncher(unittest.TestCase):
    def test_dispatch_gating(self):
        from fla_npu.ops.ascendc import _get_thin_op

        os.environ.pop("FLA_NPU_THIN_LAUNCHER", None)
        self.assertIsNotNone(_get_thin_op("recurrent_gated_delta_rule"))
        os.environ["FLA_NPU_THIN_LAUNCHER"] = "1"
        self.assertIsNotNone(_get_thin_op("recurrent_gated_delta_rule"))
        self.assertIsNotNone(_get_thin_op("npu_causal_conv1d_update"))
        # legacy npu_causal_conv1d stays on ctypes; only the PR #390 update
        # form (npu_causal_conv1d_update) is thin-enabled.
        self.assertIsNone(_get_thin_op("npu_causal_conv1d"))
        os.environ["FLA_NPU_THIN_LAUNCHER"] = "0"
        self.assertIsNone(_get_thin_op("recurrent_gated_delta_rule"))
        self.assertIsNone(_get_thin_op("npu_chunk_fwd_o"))
        os.environ.pop("FLA_NPU_THIN_LAUNCHER", None)

    def test_recurrent_parity_with_ctypes(self):
        import fla_npu._C_thin as thin_ext
        from fla_npu.ops.ascendc import _aclnn_ctypes

        torch.manual_seed(1024)
        inputs = _state_pair()
        ctypes_op = _aclnn_ctypes.npu_recurrent_gated_delta_rule

        state_c, backing_c = inputs["make_state"]()
        out_c = ctypes_op(
            inputs["query"], inputs["key"], inputs["value"], state_c,
            beta=inputs["beta"], g=inputs["g"], scale=inputs["scale"],
            actual_seq_lengths=inputs["actual_seq_lengths"],
            ssm_state_indices=inputs["ssm_state_indices"],
            num_accepted_tokens=None,
        )

        state_t, backing_t = inputs["make_state"]()
        stream = int(torch.npu.current_stream().npu_stream)
        out_t = thin_ext.npu_recurrent_gated_delta_rule(
            inputs["query"], inputs["key"], inputs["value"], state_t,
            inputs["beta"], inputs["scale"], inputs["actual_seq_lengths"],
            inputs["ssm_state_indices"], None, inputs["g"], None, stream)
        torch.npu.synchronize()

        self.assertEqual(
            float((out_c.float() - out_t.float()).abs().max().item()), 0.0)
        self.assertEqual(
            float((state_c.float() - state_t.float()).abs().max().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
