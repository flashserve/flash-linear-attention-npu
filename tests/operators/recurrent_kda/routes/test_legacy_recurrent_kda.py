"""Explicit torch.ops.npu route for recurrent_kda."""

from __future__ import annotations

import os

import pytest

from tests.operators._shared.route_requirements import require_legacy_route

require_legacy_route()

from tests.operators.recurrent_kda.common.case_matrix import case_ids  # noqa: E402


@pytest.mark.npu
def test_legacy_route_case():
    if os.environ.get("FLA_NPU_RUN_OPERATOR_TESTS") != "1":
        pytest.skip("set FLA_NPU_RUN_OPERATOR_TESTS=1 on an NPU test host")
    assert case_ids(route="torch_ops_npu")
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    import fla_npu

    fla_npu.load_legacy_torch_ops()

    device_id = int(os.environ.get("TEST_DEVICE_ID", "0"))
    device = torch.device(f"npu:{device_id}")
    torch.npu.set_device(device)
    batch, seq_len, heads, value_heads, key_dim, value_dim = 2, 2, 2, 4, 128, 128
    q = torch.randn(batch, seq_len, heads, key_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(batch, seq_len, value_heads, value_dim, device=device, dtype=torch.bfloat16)
    g = torch.randn(batch, seq_len, value_heads, key_dim, device=device, dtype=torch.float32)
    beta = torch.randn(batch, seq_len, value_heads, device=device, dtype=torch.float32)
    A_log = torch.randn(value_heads, device=device, dtype=torch.float32)
    actual_seq_lengths = torch.tensor([0] + [seq_len] * batch, device=device, dtype=torch.int32)
    state = torch.zeros(batch, value_heads, value_dim, key_dim, device=device, dtype=torch.float32)
    out, final_state = torch.ops.npu.npu_recurrent_kda(
        q,
        k,
        v,
        g,
        beta,
        state,
        actual_seq_lengths=actual_seq_lengths,
        A_log=A_log,
        layout="BSND",
        output_final_state=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
    )
    torch.npu.synchronize()
    assert out.shape == v.shape
    assert final_state.shape == (batch, value_heads, value_dim, key_dim)
    assert final_state.data_ptr() == state.data_ptr()
