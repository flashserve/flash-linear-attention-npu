"""Backward dtype regression using equal-valued FP32 inputs as the control."""

import itertools

import pytest
import torch

pytest.importorskip("torch_npu")
from fla_npu.ops.ascendc import (
    npu_chunk_kda_bwd, npu_chunk_kda_fwd, npu_chunk_kda_bwd_prepare,
    npu_chunk_kda_bwd_finalize, npu_chunk_gated_delta_rule_bwd_dhu,
)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("dtypes", list(itertools.product([torch.bfloat16, torch.float32], repeat=4)))
def test_gate_gradient_dtypes(packed, dtypes):
    torch.manual_seed(993)
    device = torch.device("npu", torch.npu.current_device())
    shape = (3, 65, 128) if packed else (1, 3, 65, 128)
    q, k, v, do = [(torch.randn(shape) * 0.02).to(device, torch.bfloat16) for _ in range(4)]
    raw = (torch.randn(shape) * 0.1).to(device, dtypes[0])
    beta = torch.sigmoid(torch.randn(shape[:-1])).to(device, dtypes[1])
    a_log = (torch.randn(3) * 0.1 - 0.5).to(device, dtypes[2])
    bias = (torch.randn((3, 128)) * 0.1).to(device, dtypes[3])
    cu = [0, 1, 1, 65] if packed else None
    layout = "NTD" if packed else "BNSD"
    scale = 128 ** -0.5
    saved = npu_chunk_kda_fwd(
        q, k, v, raw.float(), beta.float(), scale, layout=layout,
        safe_gate=True, lower_bound=-5.0, use_gate_in_kernel=True,
        A_log=a_log.float(), dt_bias=bias.float().flatten(), disable_recompute=True,
        return_intermediate_states=True, cu_seqlens=cu,
    )
    _, _, gk, aqk, akk, w, _, qg, kg, vn, h, _ = saved

    def backward(g, b, a, bias_input, output_layout):
        return npu_chunk_kda_bwd(
            q, k, v, b, gk, aqk, akk, w, qg, kg, vn, h, do, scale,
            raw_g=g, A_log=a, dt_bias=bias_input, cu_seqlens=cu,
            safe_gate=True, lower_bound=-5.0, use_gate_in_kernel=True,
            implementation="optimized", input_layout=layout, output_layout=output_layout,
        )

    daqk, dv_local, dq_raw = npu_chunk_kda_bwd_prepare(
        aqk, vn, do, h, scale=scale, layout=layout, cu_seqlens=cu,
    )
    scan = (qg, kg, w, do, dv_local, gk)
    if packed:
        scan = tuple(x.unsqueeze(0) for x in scan)
    dh, _, dv_scan = npu_chunk_gated_delta_rule_bwd_dhu(
        *scan[:5], scale, 64, gK=scan[5], cu_seqlens=cu,
        chunk_indices=[0, 0, 2, 0] if packed else None, use_exp2=True,
    )
    if packed:
        dh, dv_scan = dh.squeeze(0), dv_scan.squeeze(0)

    control = backward(raw.float(), beta.float(), a_log.float(), bias.float(), layout)
    layouts = [layout]
    if "950" not in torch.npu.get_device_name(device.index):
        layouts.append("TND" if packed else "BSND")
    for output_layout in layouts:
        actual = backward(raw, beta, a_log, bias, output_layout)
        torch.npu.synchronize()
        for index, dtype in ((3, beta.dtype), (4, raw.dtype), (6, a_log.dtype), (7, bias.dtype)):
            expected = control[index]
            if output_layout != layout and index in (3, 4):
                expected = expected.transpose(-2, -1) if index == 3 else expected.transpose(-3, -2)
            assert actual[index].dtype == dtype
            torch.testing.assert_close(actual[index], expected.to(dtype), rtol=0, atol=0)

        direct = npu_chunk_kda_bwd_finalize(
            q, k, v, gk, raw, beta, a_log, bias, akk, vn, h, dh,
            dv_scan, daqk, dq_raw, scale=scale, cu_seqlens=cu,
            output_layout=output_layout,
        )
        for got, expected in zip(direct, (*actual[:5], *actual[6:])):
            torch.testing.assert_close(got, expected, rtol=0, atol=0)
