"""Explicit torch.ops.npu route for chunk_fwd_o."""

import os

import pytest

from tests.operators.chunk_fwd_o.common.case_matrix import case_ids


@pytest.mark.npu
def test_legacy_route_case():
    if os.environ.get("FLA_NPU_RUN_OPERATOR_TESTS") != "1":
        pytest.skip("set FLA_NPU_RUN_OPERATOR_TESTS=1 on an NPU test host")
    assert case_ids(route="torch_ops_npu")
    import torch
    import fla_npu

    fla_npu.load_legacy_torch_ops()

    B, H_k, H_v, T, K, V, C = 1, 2, 4, 129, 128, 128, 64
    N_c = (T + C - 1) // C
    q = torch.randn(B, H_k, T, K, device="npu", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn(B, H_v, T, V, device="npu", dtype=torch.float16)
    h = torch.randn(B, H_v, N_c, K, V, device="npu", dtype=torch.float16)
    g = -torch.rand(B, H_v, T, device="npu", dtype=torch.float32).cumsum(-1)
    o = torch.ops.npu.npu_chunk_fwd_o(q, k, v, h, K ** -0.5, g=g, chunk_size=C)
    torch.npu.synchronize()
    assert o.shape == v.shape
