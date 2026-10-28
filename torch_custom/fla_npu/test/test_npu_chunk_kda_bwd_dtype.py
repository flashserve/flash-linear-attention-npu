"""Backward dtype regression using equal-valued FP32 inputs as the control."""

import itertools

import pytest
import torch

pytest.importorskip("torch_npu")
from fla_npu.ops.ascendc import npu_chunk_kda_bwd, npu_chunk_kda_fwd


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("dtypes", list(itertools.product([torch.bfloat16, torch.float32], repeat=3)))
def test_gate_gradient_dtypes(packed, dtypes):
    torch.manual_seed(993)
    device = torch.device("npu", torch.npu.current_device())
    shape = (3, 65, 128) if packed else (1, 3, 65, 128)
    q, k, v, do = [(torch.randn(shape) * 0.02).to(device, torch.bfloat16) for _ in range(4)]
    raw = (torch.randn(shape) * 0.1).to(device, dtypes[0])
    beta = torch.sigmoid(torch.randn(shape[:-1])).to(device, dtypes[1])
    a_log = (torch.randn(3) * 0.1 - 0.5).to(device, dtypes[2])
    bias = torch.zeros((3, 128), device=device)
    cu = [0, 1, 1, 65] if packed else None
    layout = "NTD" if packed else "BNSD"
    scale = 128 ** -0.5
    saved = npu_chunk_kda_fwd(
        q, k, v, raw.float(), beta.float(), scale, layout=layout,
        safe_gate=True, lower_bound=-5.0, use_gate_in_kernel=True,
        A_log=a_log.float(), dt_bias=bias.flatten(), disable_recompute=True,
        return_intermediate_states=True, cu_seqlens=cu,
    )
    _, _, gk, aqk, akk, w, _, qg, kg, vn, h, _ = saved

    def backward(g, b, a, output_layout):
        return npu_chunk_kda_bwd(
            q, k, v, b, gk, aqk, akk, w, qg, kg, vn, h, do, scale,
            raw_g=g, A_log=a, dt_bias=bias, cu_seqlens=cu,
            safe_gate=True, lower_bound=-5.0, use_gate_in_kernel=True,
            implementation="optimized", input_layout=layout, output_layout=output_layout,
        )

    control = backward(raw.float(), beta.float(), a_log.float(), layout)
    layouts = [layout]
    if "950" not in torch.npu.get_device_name(device.index):
        layouts.append("TND" if packed else "BSND")
    for output_layout in layouts:
        actual = backward(raw, beta, a_log, output_layout)
        torch.npu.synchronize()
        for index, dtype in ((3, beta.dtype), (4, raw.dtype), (6, a_log.dtype)):
            expected = control[index]
            if output_layout != layout and index in (3, 4):
                expected = expected.transpose(-2, -1) if index == 3 else expected.transpose(-3, -2)
            assert actual[index].dtype == dtype
            torch.testing.assert_close(actual[index], expected.to(dtype), rtol=0, atol=0)
        assert actual[7].dtype == torch.float32
