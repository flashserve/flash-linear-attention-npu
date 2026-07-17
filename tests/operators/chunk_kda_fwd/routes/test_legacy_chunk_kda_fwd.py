"""Explicit torch.ops.npu route for chunk_kda_fwd."""

import os

import pytest

from tests.operators.chunk_kda_fwd.common.case_matrix import case_ids


@pytest.mark.npu
def test_legacy_route_case():
    if os.environ.get("FLA_NPU_RUN_OPERATOR_TESTS") != "1":
        pytest.skip("set FLA_NPU_RUN_OPERATOR_TESTS=1 on an NPU test host")
    assert case_ids(route="torch_ops_npu")
    import torch
    import fla_npu

    fla_npu.load_legacy_torch_ops()

    B, H_k, H_v, T, K, V, C = 1, 2, 4, 128, 128, 128, 64
    q = torch.randn(B, H_k, T, K, device="npu", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(B, H_v, T, V, device="npu", dtype=torch.bfloat16)
    gk = -torch.rand(B, H_v, T, K, device="npu", dtype=torch.float32).cumsum(2)
    beta = torch.sigmoid(torch.randn(B, H_v, T, device="npu", dtype=torch.float32))
    outputs = torch.ops.npu.npu_chunk_kda_fwd(q, k, v, gk, beta, K ** -0.5, C,
                            layout="BNSD", output_final_state=True)
    o, final_state = outputs[:2]
    torch.npu.synchronize()
    assert o.shape == v.shape and final_state.dtype == torch.float32
