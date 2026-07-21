"""Explicit torch.ops.npu route for chunk_gated_delta_rule_bwd_dhu."""

import os

import pytest

from tests.operators._shared.route_requirements import require_legacy_route

require_legacy_route()

from tests.operators.chunk_gated_delta_rule_bwd_dhu.common.case_matrix import case_ids


@pytest.mark.npu
def test_legacy_route_case():
    if os.environ.get("FLA_NPU_RUN_OPERATOR_TESTS") != "1":
        pytest.skip("set FLA_NPU_RUN_OPERATOR_TESTS=1 on an NPU test host")
    assert case_ids(route="torch_ops_npu")
    import torch
    import fla_npu

    fla_npu.load_legacy_torch_ops()

    B, H_k, H_v, T, K, V, C = 1, 2, 4, 128, 128, 128, 64
    q = torch.randn(B, H_k, T, K, device="npu", dtype=torch.float16)
    k = torch.randn_like(q)
    w = torch.randn(B, H_v, T, K, device="npu", dtype=torch.float16)
    d_o = torch.randn(B, H_v, T, V, device="npu", dtype=torch.float16)
    dv = torch.randn_like(d_o)
    g = -torch.rand(B, H_v, T, device="npu", dtype=torch.float32).cumsum(-1)
    dh, dh0, dv2 = torch.ops.npu.npu_chunk_gated_delta_rule_bwd_dhu(q, k, w, d_o, dv, K ** -0.5, C, g=g)
    torch.npu.synchronize()
    assert dv2.shape == dv.shape
