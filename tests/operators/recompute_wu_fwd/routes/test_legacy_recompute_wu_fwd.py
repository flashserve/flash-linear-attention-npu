"""Explicit torch.ops.npu route for recompute_wu_fwd."""

import os

import pytest

from tests.operators._shared.route_requirements import require_legacy_route

require_legacy_route()

from tests.operators.recompute_wu_fwd.common.case_matrix import case_ids


@pytest.mark.npu
def test_legacy_route_case():
    if os.environ.get("FLA_NPU_RUN_OPERATOR_TESTS") != "1":
        pytest.skip("set FLA_NPU_RUN_OPERATOR_TESTS=1 on an NPU test host")
    assert case_ids(route="torch_ops_npu")
    import torch
    import fla_npu

    fla_npu.load_legacy_torch_ops()

    B, H_k, H_v, T, K, V, C = 1, 2, 4, 128, 128, 256, 64
    k = torch.randn(B, H_k, T, K, device="npu", dtype=torch.bfloat16)
    v = torch.randn(B, H_v, T, V, device="npu", dtype=torch.bfloat16)
    beta = torch.rand(B, H_v, T, device="npu", dtype=torch.float32)
    A = torch.randn(B, H_v, T, C, device="npu", dtype=torch.bfloat16)
    g = -torch.rand(B, H_v, T, device="npu", dtype=torch.float32).cumsum(-1)
    w, u = torch.ops.npu.npu_recompute_w_u_fwd(k, v, beta, A, C, g=g)
    torch.npu.synchronize()
    assert w.shape == (B, H_v, T, K) and u.shape == v.shape
