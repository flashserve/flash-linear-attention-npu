"""Explicit torch.ops.npu route for causal_conv1d_bwd."""

import os

import pytest

from tests.operators._shared.route_requirements import require_legacy_route

require_legacy_route()

from tests.operators.causal_conv1d_bwd.common.case_matrix import case_ids


@pytest.mark.npu
def test_legacy_route_case():
    if os.environ.get("FLA_NPU_RUN_OPERATOR_TESTS") != "1":
        pytest.skip("set FLA_NPU_RUN_OPERATOR_TESTS=1 on an NPU test host")
    assert case_ids(route="torch_ops_npu")
    import torch
    import fla_npu

    fla_npu.load_legacy_torch_ops()

    B, T, D, W = 2, 64, 128, 3
    x = torch.randn(B, T, D, device="npu", dtype=torch.float16)
    weight = torch.randn(W, D, device="npu", dtype=torch.float16)
    y = torch.randn(B, 4, T, D // 4, device="npu", dtype=torch.float16)
    dy = torch.randn_like(y)
    dx, dw, db, dh0 = torch.ops.npu.npu_causal_conv1d_bwd(x, y, weight, dy, activation=1, input_layout="BNSD")
    torch.npu.synchronize()
    assert dx.shape == x.shape and dw.shape == weight.shape
