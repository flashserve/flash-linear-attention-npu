"""Explicit torch.ops.npu route for chunk_gated_delta_rule_fwd_h."""

import os

import pytest

from tests.operators.chunk_gated_delta_rule_fwd_h.common.case_matrix import case_ids


@pytest.mark.npu
def test_legacy_route_case():
    if os.environ.get("FLA_NPU_RUN_OPERATOR_TESTS") != "1":
        pytest.skip("set FLA_NPU_RUN_OPERATOR_TESTS=1 on an NPU test host")
    assert case_ids(route="torch_ops_npu")
    import torch
    import fla_npu

    fla_npu.load_legacy_torch_ops()

    B, H_k, H_v, T, K, V, C = 1, 2, 4, 128, 128, 128, 64
    k = torch.randn(B, H_k, T, K, device="npu", dtype=torch.bfloat16)
    w = torch.randn(B, H_v, T, K, device="npu", dtype=torch.bfloat16)
    u = torch.randn(B, H_v, T, V, device="npu", dtype=torch.bfloat16)
    g = -torch.rand(B, H_v, T, device="npu", dtype=torch.float32).cumsum(-1)
    h, v_new, final_state = torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h(
        k, w, u, g, chunk_size=C, output_final_state=True)
    torch.npu.synchronize()
    assert v_new.shape == u.shape and final_state.shape == (B, H_v, K, V)
