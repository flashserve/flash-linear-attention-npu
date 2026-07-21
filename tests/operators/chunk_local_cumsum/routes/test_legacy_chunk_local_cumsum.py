"""Explicit torch.ops.npu route for chunk_local_cumsum."""

import os

import pytest

from tests.operators._shared.route_requirements import require_legacy_route

require_legacy_route()

from tests.operators.chunk_local_cumsum.common.case_matrix import case_ids


@pytest.mark.npu
def test_legacy_route_case():
    if os.environ.get("FLA_NPU_RUN_OPERATOR_TESTS") != "1":
        pytest.skip("set FLA_NPU_RUN_OPERATOR_TESTS=1 on an NPU test host")
    assert case_ids(route="torch_ops_npu")
    import torch
    import fla_npu

    fla_npu.load_legacy_torch_ops()

    B, H_v, T, C = 1, 4, 129, 64
    g = torch.randn(B, H_v, T, device="npu", dtype=torch.float32)
    out = torch.ops.npu.npu_chunk_local_cumsum(g, C, reverse=False, scale=1.0, head_first=True)
    torch.npu.synchronize()
    assert out.shape == g.shape and out.dtype == torch.float32
