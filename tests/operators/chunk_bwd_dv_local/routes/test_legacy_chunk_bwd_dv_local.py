"""Explicit torch.ops.npu route for chunk_bwd_dv_local."""

import os

import pytest

from tests.operators._shared.route_requirements import require_legacy_route

require_legacy_route()

from tests.operators.chunk_bwd_dv_local.common.case_matrix import case_ids


@pytest.mark.npu
def test_legacy_route_case():
    if os.environ.get("FLA_NPU_RUN_OPERATOR_TESTS") != "1":
        pytest.skip("set FLA_NPU_RUN_OPERATOR_TESTS=1 on an NPU test host")
    assert case_ids(route="torch_ops_npu")
    import torch
    import fla_npu

    fla_npu.load_legacy_torch_ops()

    B, H_k, H_v, T, K, V, C = 1, 2, 4, 129, 128, 128, 64
    q = torch.randn(B, H_k, T, K, device="npu", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    d_o = torch.randn(B, H_v, T, V, device="npu", dtype=torch.bfloat16)
    g = -torch.rand(B, H_v, T, device="npu", dtype=torch.float32).cumsum(-1)
    d_v = torch.ops.npu.npu_chunk_bwd_dv_local(q, k, d_o, g, K ** -0.5, C)
    torch.npu.synchronize()
    assert d_v.shape == d_o.shape
