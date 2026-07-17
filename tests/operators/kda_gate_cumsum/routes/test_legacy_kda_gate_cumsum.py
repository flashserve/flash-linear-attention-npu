"""Explicit torch.ops.npu route for kda_gate_cumsum."""

import os

import pytest

from tests.operators.kda_gate_cumsum.common.case_matrix import case_ids


@pytest.mark.npu
def test_legacy_route_case():
    if os.environ.get("FLA_NPU_RUN_OPERATOR_TESTS") != "1":
        pytest.skip("set FLA_NPU_RUN_OPERATOR_TESTS=1 on an NPU test host")
    assert case_ids(route="torch_ops_npu")
    import torch
    import fla_npu

    fla_npu.load_legacy_torch_ops()

    B, T, H_v, K, C = 1, 128, 4, 128, 64
    raw = torch.randn(B, T, H_v, K, device="npu", dtype=torch.bfloat16)
    A_log = torch.randn(H_v, device="npu", dtype=torch.float32)
    dt_bias = torch.randn(H_v, K, device="npu", dtype=torch.float32)
    gk = torch.ops.npu.npu_kda_gate_cumsum(raw, C, A_log=A_log, dt_bias=dt_bias,
                         use_gate_in_kernel=True, safe_gate=True,
                         lower_bound=-5.0, layout="BSND")
    torch.npu.synchronize()
    assert gk.shape == raw.shape and gk.dtype == torch.float32
