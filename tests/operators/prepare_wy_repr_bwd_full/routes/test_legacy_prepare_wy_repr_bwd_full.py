"""Explicit torch.ops.npu route for prepare_wy_repr_bwd_full."""

import os

import pytest

from tests.operators.prepare_wy_repr_bwd_full.common.case_matrix import case_ids


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
    dA = torch.randn_like(A)
    dw = torch.randn(B, H_v, T, K, device="npu", dtype=torch.bfloat16)
    du = torch.randn_like(v)
    g = -torch.rand(B, H_v, T, device="npu", dtype=torch.float32).cumsum(-1)
    dk, dv, dbeta, dg = torch.ops.npu.npu_prepare_wy_repr_bwd_full(k, v, beta, A, dA, dw, du, g, C)
    torch.npu.synchronize()
    assert dk.shape == k.shape and dv.shape == v.shape
