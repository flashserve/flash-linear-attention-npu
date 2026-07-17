"""Explicit torch.ops.npu route for solve_tri."""

import os

import pytest

from tests.operators.solve_tri.common.case_matrix import case_ids


@pytest.mark.npu
def test_legacy_route_case():
    if os.environ.get("FLA_NPU_RUN_OPERATOR_TESTS") != "1":
        pytest.skip("set FLA_NPU_RUN_OPERATOR_TESTS=1 on an NPU test host")
    assert case_ids(route="torch_ops_npu")
    import torch
    import fla_npu

    fla_npu.load_legacy_torch_ops()

    B, T, H, C = 1, 128, 4, 64
    x = torch.randn(B, T, H, C, device="npu", dtype=torch.float16)
    row = torch.arange(C, device="npu").view(1, 1, 1, C)
    pos = torch.arange(T, device="npu").view(1, T, 1, 1) % C
    x = torch.where(row < pos, x * 0.01, torch.zeros_like(x))
    y = torch.ops.npu.npu_solve_tri(x, layout="bsnd")
    torch.npu.synchronize()
    assert y.shape == x.shape
