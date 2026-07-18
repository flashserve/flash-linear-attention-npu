"""Explicit torch.ops.npu route for causal_conv1d."""

import os

import pytest

from tests.operators._shared.route_requirements import require_legacy_route

require_legacy_route()

from tests.operators.causal_conv1d.common.case_matrix import case_ids


@pytest.mark.npu
def test_legacy_route_case():
    if os.environ.get("FLA_NPU_RUN_OPERATOR_TESTS") != "1":
        pytest.skip("set FLA_NPU_RUN_OPERATOR_TESTS=1 on an NPU test host")
    assert case_ids(route="torch_ops_npu")
    import torch
    import fla_npu

    fla_npu.load_legacy_torch_ops()

    B, T, D, W = 2, 64, 128, 3
    x = torch.randn(B, T, D, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(W, D, device="npu", dtype=torch.bfloat16)
    bias = torch.randn(D, device="npu", dtype=torch.bfloat16)
    state = torch.zeros(B, W, D, device="npu", dtype=torch.bfloat16)
    y = torch.ops.npu.npu_causal_conv1d(x, weight, bias, state, activation_mode=1, run_mode=0)
    torch.npu.synchronize()
    assert y.shape == x.shape
