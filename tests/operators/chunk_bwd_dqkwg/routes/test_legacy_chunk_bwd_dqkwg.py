"""Explicit torch.ops.npu route for chunk_bwd_dqkwg."""

import os

import pytest

from tests.operators._shared.route_requirements import require_legacy_route

require_legacy_route()

from tests.operators.chunk_bwd_dqkwg.common.case_matrix import case_ids


@pytest.mark.npu
def test_legacy_route_case():
    if os.environ.get("FLA_NPU_RUN_OPERATOR_TESTS") != "1":
        pytest.skip("set FLA_NPU_RUN_OPERATOR_TESTS=1 on an NPU test host")
    assert case_ids(route="torch_ops_npu")
    import torch
    import fla_npu

    fla_npu.load_legacy_torch_ops()

    B, H_k, H_v, T, K, V, C = 1, 2, 4, 128, 128, 128, 64
    N_c = (T + C - 1) // C
    q = torch.randn(B, H_k, T, K, device="npu", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn(B, H_v, T, V, device="npu", dtype=torch.float16)
    g = -torch.rand(B, H_v, T, device="npu", dtype=torch.float32).cumsum(-1)
    h = torch.randn(B, H_v, N_c, K, V, device="npu", dtype=torch.float16)
    dox, dv = torch.randn_like(v), torch.randn_like(v)
    dh = torch.randn_like(h)
    dq, dk, dw, dg = torch.ops.npu.npu_chunk_bwd_dqkwg(
        q, k, v, g, h, dox, dh, dv, C,
        scale=K ** -0.5, w=None, g_gamma=None)
    torch.npu.synchronize()
    assert dq.shape == q.shape and dk.shape == k.shape
