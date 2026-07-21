"""Explicit torch.ops.npu route for chunk_scaled_dot_kkt."""

import os

import pytest

from tests.operators._shared.route_requirements import require_legacy_route

require_legacy_route()

from tests.operators.chunk_scaled_dot_kkt.common.case_matrix import case_ids


@pytest.mark.npu
def test_legacy_route_case():
    if os.environ.get("FLA_NPU_RUN_OPERATOR_TESTS") != "1":
        pytest.skip("set FLA_NPU_RUN_OPERATOR_TESTS=1 on an NPU test host")
    assert case_ids(route="torch_ops_npu")
    import torch
    import fla_npu

    fla_npu.load_legacy_torch_ops()

    B, H_k, H_v, T, K, C = 1, 2, 4, 129, 128, 64
    k = torch.randn(B, H_k, T, K, device="npu", dtype=torch.float16)
    g = -torch.rand(B, H_v, T, device="npu", dtype=torch.float32).cumsum(-1)
    beta = torch.rand(B, H_v, T, device="npu", dtype=torch.float32)
    A = torch.ops.npu.npu_chunk_scaled_dot_kkt(k, g, beta, chunk_size=C)
    torch.npu.synchronize()
    assert A.shape == (B, H_k, T, C) and A.dtype == torch.float32
