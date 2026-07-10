# Copyright (c) 2026 Tianjin University, Ltd.

import pathlib
import sys

import torch

try:
    import torch_npu  # noqa: F401
except Exception:  # pragma: no cover - CPU fallback for syntax/smoke only
    torch_npu = None

import fla_npu  # noqa: F401


ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tests.reference.chunk_kda_reference import chunk_kda_forward_reference  # noqa: E402


def _device():
    if torch_npu is not None and hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    return torch.device("cpu")


def _make_inputs(device, b=1, h=2, hv=2, t=64, kdim=32, vdim=64, dtype=torch.float32):
    torch.manual_seed(1234 + b + h + hv + t + kdim + vdim)
    q = (torch.randn(b, t, h, kdim, device=device, dtype=dtype) * 0.08).requires_grad_(True)
    k = (torch.randn(b, t, h, kdim, device=device, dtype=dtype) * 0.08).requires_grad_(True)
    v = (torch.randn(b, t, hv, vdim, device=device, dtype=dtype) * 0.08).requires_grad_(True)
    gk = (torch.randn(b, t, hv, kdim, device=device, dtype=torch.float32).cumsum(dim=1) * 0.001).requires_grad_(True)
    beta = torch.sigmoid(torch.randn(b, t, hv, device=device, dtype=torch.float32)).requires_grad_(True)
    initial_state = (torch.randn(b, hv, kdim, vdim, device=device, dtype=torch.float32) * 0.02).requires_grad_(True)
    return q, k, v, gk, beta, initial_state


def _assert_close(name, actual, expected, rtol=2e-3, atol=2e-3):
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=rtol, atol=atol, msg=name)


def _kda_gate_cumsum_reference(g, chunk_size, A_log=None, dt_bias=None, use_gate_in_kernel=False,
                               safe_gate=False, lower_bound=-5.0):
    rcp_ln2 = 1.4426950408889634
    g_float = g.to(torch.float32)
    if use_gate_in_kernel:
        if not safe_gate:
            raise ValueError("test reference only covers safe_gate raw path")
        x = g_float
        if dt_bias is not None:
            bias = dt_bias.reshape(g.shape[-2], g.shape[-1]).to(torch.float32)
            if g.dim() == 4:
                x = x + bias[None, None, :, :]
            else:
                x = x + bias[None, :, :]
        a = torch.exp(A_log.to(torch.float32))
        if g.dim() == 4:
            x = x * a[None, None, :, None]
        else:
            x = x * a[None, :, None]
        gate = float(lower_bound) * torch.sigmoid(x)
    else:
        gate = g_float

    out = torch.empty_like(gate, dtype=torch.float32)
    if g.dim() == 4:
        for b in range(g.shape[0]):
            for start in range(0, g.shape[1], chunk_size):
                end = min(start + chunk_size, g.shape[1])
                out[b, start:end] = torch.cumsum(gate[b, start:end] * rcp_ln2, dim=0)
    else:
        for start in range(0, g.shape[0], chunk_size):
            end = min(start + chunk_size, g.shape[0])
            out[start:end] = torch.cumsum(gate[start:end] * rcp_ln2, dim=0)
    return out


def test_chunk_kda_fwd_matches_reference():
    device = _device()
    q, k, v, gk, beta, initial_state = _make_inputs(device, h=1, hv=1, t=8, kdim=8, vdim=8)
    scale = q.shape[-1] ** -0.5

    got = torch.ops.npu.npu_chunk_kda_fwd(
        q,
        k,
        v,
        gk,
        beta,
        scale,
        64,
        initial_state=initial_state,
        output_final_state=True,
        return_intermediate=True,
    )
    ref = chunk_kda_forward_reference(
        q.detach().cpu(),
        k.detach().cpu(),
        v.detach().cpu(),
        gk.detach().cpu(),
        beta.detach().cpu(),
        scale=scale,
        chunk_size=64,
        initial_state=initial_state.detach().cpu(),
        output_final_state=True,
    )

    _assert_close("o", got[0], ref.o)
    _assert_close("final_state", got[1], ref.final_state)
    _assert_close("g", got[2], gk)
    _assert_close("Aqk", got[3], ref.Aqk)
    _assert_close("Akk", got[4], ref.Akk)
    _assert_close("w", got[5], ref.w)
    _assert_close("u", got[6], ref.u)
    _assert_close("qg", got[7], ref.qg)
    _assert_close("kg", got[8], ref.kg)
    _assert_close("v_new", got[9], ref.v_new)
    _assert_close("initial_state", got[11], initial_state)


def test_chunk_kda_fwd_chunk128_v128_gva_varlen():
    device = _device()
    q, k, v, gk, beta, _ = _make_inputs(device, h=1, hv=2, t=16, kdim=8, vdim=128)
    scale = q.shape[-1] ** -0.5
    cu_seqlens = [0, 6, 16]

    got = torch.ops.npu.npu_chunk_kda_fwd(
        q,
        k,
        v,
        gk,
        beta,
        scale,
        128,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        return_intermediate=False,
    )
    ref = chunk_kda_forward_reference(
        q.detach().cpu(),
        k.detach().cpu(),
        v.detach().cpu(),
        gk.detach().cpu(),
        beta.detach().cpu(),
        scale=scale,
        chunk_size=128,
        output_final_state=True,
        cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int64),
    )
    _assert_close("o chunk128 v128 gva varlen", got[0], ref.o, rtol=3e-3, atol=3e-3)
    _assert_close("final_state chunk128 v128 gva varlen", got[1], ref.final_state, rtol=3e-3, atol=3e-3)
    _assert_close("g chunk128 v128 gva varlen", got[2], gk)
    assert got[11].numel() == 0


def test_chunk_kda_fwd_bf16_chunk32_matches_reference():
    device = _device()
    if device.type == "cpu":
        return
    q, k, v, gk, beta, initial_state = _make_inputs(
        device, h=1, hv=1, t=8, kdim=8, vdim=32, dtype=torch.bfloat16
    )
    scale = q.shape[-1] ** -0.5

    got = torch.ops.npu.npu_chunk_kda_fwd(
        q,
        k,
        v,
        gk,
        beta,
        scale,
        32,
        initial_state=initial_state,
        output_final_state=True,
        return_intermediate=False,
    )
    ref = chunk_kda_forward_reference(
        q.detach().cpu(),
        k.detach().cpu(),
        v.detach().cpu(),
        gk.detach().cpu(),
        beta.detach().cpu(),
        scale=scale,
        chunk_size=32,
        initial_state=initial_state.detach().cpu(),
        output_final_state=True,
    )
    _assert_close("o bf16 chunk32", got[0], ref.o, rtol=2e-2, atol=2e-2)
    _assert_close("final_state bf16 chunk32", got[1], ref.final_state, rtol=2e-2, atol=2e-2)
    _assert_close("g bf16 chunk32", got[2], gk)
    _assert_close("initial_state bf16 chunk32", got[11], initial_state)


def test_chunk_kda_fwd_bf16_gate_matches_reference():
    device = _device()
    if device.type == "cpu":
        return
    q, k, v, gk, beta, initial_state = _make_inputs(
        device, h=1, hv=1, t=8, kdim=8, vdim=32, dtype=torch.float16
    )
    gk_bf16 = gk.detach().to(torch.bfloat16).requires_grad_(True)
    beta_bf16 = beta.detach().to(torch.bfloat16).requires_grad_(True)
    scale = q.shape[-1] ** -0.5

    got = torch.ops.npu.npu_chunk_kda_fwd(
        q,
        k,
        v,
        gk_bf16,
        beta_bf16,
        scale,
        64,
        initial_state=initial_state,
        output_final_state=True,
        return_intermediate=False,
    )
    ref = chunk_kda_forward_reference(
        q.detach().cpu(),
        k.detach().cpu(),
        v.detach().cpu(),
        gk_bf16.detach().cpu().float(),
        beta_bf16.detach().cpu().float(),
        scale=scale,
        chunk_size=64,
        initial_state=initial_state.detach().cpu(),
        output_final_state=True,
    )
    _assert_close("o bf16 gate", got[0], ref.o, rtol=2e-2, atol=2e-2)
    _assert_close("final_state bf16 gate", got[1], ref.final_state, rtol=2e-2, atol=2e-2)
    _assert_close("g bf16 gate", got[2], gk)
    _assert_close("initial_state bf16 gate", got[11], initial_state)


def test_chunk_kda_fwd_fp16_matches_reference():
    device = _device()
    if device.type == "cpu":
        return
    q, k, v, gk, beta, initial_state = _make_inputs(
        device, h=1, hv=1, t=8, kdim=8, vdim=32, dtype=torch.float16
    )
    scale = q.shape[-1] ** -0.5

    got = torch.ops.npu.npu_chunk_kda_fwd(
        q,
        k,
        v,
        gk,
        beta,
        scale,
        64,
        initial_state=initial_state,
        output_final_state=True,
        return_intermediate=True,
    )
    ref = chunk_kda_forward_reference(
        q.detach().cpu(),
        k.detach().cpu(),
        v.detach().cpu(),
        gk.detach().cpu(),
        beta.detach().cpu(),
        scale=scale,
        chunk_size=64,
        initial_state=initial_state.detach().cpu(),
        output_final_state=True,
    )
    _assert_close("o fp16", got[0], ref.o, rtol=2e-2, atol=2e-2)
    _assert_close("final_state fp16", got[1], ref.final_state, rtol=2e-2, atol=2e-2)
    _assert_close("g fp16", got[2], gk)
    _assert_close("Aqk fp16", got[3], ref.Aqk, rtol=2e-2, atol=2e-2)
    _assert_close("Akk fp16", got[4], ref.Akk, rtol=2e-2, atol=2e-2)
    _assert_close("w fp16", got[5], ref.w, rtol=2e-2, atol=2e-2)
    _assert_close("u fp16", got[6], ref.u, rtol=2e-2, atol=2e-2)
    _assert_close("qg fp16", got[7], ref.qg, rtol=2e-2, atol=2e-2)
    _assert_close("kg fp16", got[8], ref.kg, rtol=2e-2, atol=2e-2)
    _assert_close("v_new fp16", got[9], ref.v_new, rtol=2e-2, atol=2e-2)
    _assert_close("h fp16", got[10], ref.h, rtol=2e-2, atol=2e-2)
    _assert_close("initial_state fp16", got[11], initial_state)


def test_chunk_kda_fwd_fp16_large_return_intermediate_matches_reference():
    device = _device()
    if device.type == "cpu":
        return
    torch.manual_seed(2026)
    q_cpu = (torch.randn(1, 128, 1, 128, dtype=torch.float16) * 0.08).contiguous()
    k_cpu = (torch.randn(1, 128, 1, 128, dtype=torch.float16) * 0.08).contiguous()
    v_cpu = (torch.randn(1, 128, 2, 64, dtype=torch.float16) * 0.08).contiguous()
    gk_cpu = (torch.randn(1, 128, 2, 128, dtype=torch.float32).cumsum(dim=1) * 0.001).contiguous()
    beta_cpu = torch.sigmoid(torch.randn(1, 128, 2, dtype=torch.float32)).contiguous()
    q = q_cpu.to(device)
    k = k_cpu.to(device)
    v = v_cpu.to(device)
    gk = gk_cpu.to(device)
    beta = beta_cpu.to(device)
    scale = q.shape[-1] ** -0.5

    got = torch.ops.npu.npu_chunk_kda_fwd(
        q,
        k,
        v,
        gk,
        beta,
        scale,
        64,
        output_final_state=True,
        return_intermediate=True,
    )
    ref = chunk_kda_forward_reference(
        q_cpu,
        k_cpu,
        v_cpu,
        gk_cpu,
        beta_cpu,
        scale=scale,
        chunk_size=64,
        output_final_state=True,
    )

    expected_shapes = [
        v.shape,
        (q.shape[0], v.shape[2], q.shape[3], v.shape[3]),
        gk.shape,
        (q.shape[0], q.shape[1], v.shape[2], 64),
        (q.shape[0], q.shape[1], v.shape[2], 64),
        (q.shape[0], q.shape[1], v.shape[2], q.shape[3]),
        v.shape,
        (q.shape[0], q.shape[1], v.shape[2], q.shape[3]),
        (q.shape[0], q.shape[1], v.shape[2], q.shape[3]),
        v.shape,
        (q.shape[0], 2, v.shape[2], q.shape[3], v.shape[3]),
    ]
    for idx, shape in enumerate(expected_shapes):
        assert tuple(got[idx].shape) == tuple(shape)
        assert torch.isfinite(got[idx].detach().cpu().float()).all().item()
    assert got[1].dtype == torch.float32
    assert got[11].numel() == 0
    _assert_close("o large fp16", got[0], ref.o, rtol=2e-2, atol=2e-2)
    _assert_close("final_state large fp16", got[1], ref.final_state, rtol=2e-2, atol=2e-2)
    _assert_close("g large fp16", got[2], gk_cpu)
    _assert_close("Aqk large fp16", got[3], ref.Aqk, rtol=2e-2, atol=2e-2)
    _assert_close("Akk large fp16", got[4], ref.Akk, rtol=2e-2, atol=2e-2)
    _assert_close("w large fp16", got[5], ref.w, rtol=2e-2, atol=2e-2)
    _assert_close("u large fp16", got[6], ref.u, rtol=2e-2, atol=2e-2)
    _assert_close("qg large fp16", got[7], ref.qg, rtol=2e-2, atol=2e-2)
    _assert_close("kg large fp16", got[8], ref.kg, rtol=2e-2, atol=2e-2)
    _assert_close("v_new large fp16", got[9], ref.v_new, rtol=2e-2, atol=2e-2)
    _assert_close("h large fp16", got[10], ref.h, rtol=2e-2, atol=2e-2)


def test_chunk_kda_fwd_tnd_matches_reference():
    device = _device()
    q, k, v, gk, beta, initial_state = _make_inputs(device, b=1, h=1, hv=2, t=8, kdim=8, vdim=16)
    scale = q.shape[-1] ** -0.5

    got = torch.ops.npu.npu_chunk_kda_fwd(
        q.squeeze(0),
        k.squeeze(0),
        v.squeeze(0),
        gk.squeeze(0),
        beta.squeeze(0),
        scale,
        64,
        initial_state=initial_state,
        output_final_state=True,
        return_intermediate=True,
    )
    ref = chunk_kda_forward_reference(
        q.detach().cpu(),
        k.detach().cpu(),
        v.detach().cpu(),
        gk.detach().cpu(),
        beta.detach().cpu(),
        scale=scale,
        chunk_size=64,
        initial_state=initial_state.detach().cpu(),
        output_final_state=True,
    )

    _assert_close("o tnd", got[0], ref.o.squeeze(0))
    _assert_close("final_state tnd", got[1], ref.final_state)
    _assert_close("g tnd", got[2], gk.squeeze(0))
    _assert_close("Aqk tnd", got[3], ref.Aqk.squeeze(0))
    _assert_close("Akk tnd", got[4], ref.Akk.squeeze(0))
    _assert_close("w tnd", got[5], ref.w.squeeze(0))
    _assert_close("u tnd", got[6], ref.u.squeeze(0))
    _assert_close("qg tnd", got[7], ref.qg.squeeze(0))
    _assert_close("kg tnd", got[8], ref.kg.squeeze(0))
    _assert_close("v_new tnd", got[9], ref.v_new.squeeze(0))
    _assert_close("h tnd", got[10], ref.h.squeeze(0))
    _assert_close("initial_state tnd", got[11], initial_state)


def test_chunk_kda_fwd_bnsd_direct_matches_reference():
    device = _device()
    if device.type == "cpu":
        return
    q, k, v, gk, beta, initial_state = _make_inputs(device, b=1, h=1, hv=2, t=16, kdim=16, vdim=32,
                                                    dtype=torch.float16)
    scale = q.shape[-1] ** -0.5
    q_bnsd = q.permute(0, 2, 1, 3).contiguous()
    k_bnsd = k.permute(0, 2, 1, 3).contiguous()
    v_bnsd = v.permute(0, 2, 1, 3).contiguous()
    gk_bnsd = gk.permute(0, 2, 1, 3).contiguous()
    beta_bns = beta.permute(0, 2, 1).contiguous()

    got = torch.ops.npu.npu_chunk_kda_fwd(
        q_bnsd,
        k_bnsd,
        v_bnsd,
        gk_bnsd,
        beta_bns,
        scale,
        64,
        initial_state=initial_state,
        output_final_state=True,
        return_intermediate=True,
    )
    ref = chunk_kda_forward_reference(
        q.detach().cpu(),
        k.detach().cpu(),
        v.detach().cpu(),
        gk.detach().cpu(),
        beta.detach().cpu(),
        scale=scale,
        chunk_size=64,
        initial_state=initial_state.detach().cpu(),
        output_final_state=True,
    )

    _assert_close("o bnsd", got[0], ref.o.permute(0, 2, 1, 3), rtol=2e-2, atol=2e-2)
    _assert_close("final_state bnsd", got[1], ref.final_state, rtol=2e-2, atol=2e-2)
    _assert_close("g bnsd", got[2], gk_bnsd, rtol=2e-2, atol=2e-2)
    _assert_close("Aqk bnsd", got[3], ref.Aqk.permute(0, 2, 1, 3), rtol=2e-2, atol=2e-2)
    _assert_close("Akk bnsd", got[4], ref.Akk.permute(0, 2, 1, 3), rtol=2e-2, atol=2e-2)
    _assert_close("w bnsd", got[5], ref.w.permute(0, 2, 1, 3), rtol=2e-2, atol=2e-2)
    _assert_close("u bnsd", got[6], ref.u.permute(0, 2, 1, 3), rtol=2e-2, atol=2e-2)
    _assert_close("qg bnsd", got[7], ref.qg.permute(0, 2, 1, 3), rtol=2e-2, atol=2e-2)
    _assert_close("kg bnsd", got[8], ref.kg.permute(0, 2, 1, 3), rtol=2e-2, atol=2e-2)
    _assert_close("v_new bnsd", got[9], ref.v_new.permute(0, 2, 1, 3), rtol=2e-2, atol=2e-2)
    _assert_close("h bnsd", got[10], ref.h.permute(0, 2, 1, 3, 4), rtol=2e-2, atol=2e-2)
    _assert_close("initial_state bnsd", got[11], initial_state)


def test_chunk_kda_fwd_ntd_direct_matches_reference():
    device = _device()
    if device.type == "cpu":
        return
    q, k, v, gk, beta, initial_state = _make_inputs(device, b=1, h=1, hv=2, t=16, kdim=16, vdim=32,
                                                    dtype=torch.float16)
    scale = q.shape[-1] ** -0.5
    q_ntd = q.squeeze(0).permute(1, 0, 2).contiguous()
    k_ntd = k.squeeze(0).permute(1, 0, 2).contiguous()
    v_ntd = v.squeeze(0).permute(1, 0, 2).contiguous()
    gk_ntd = gk.squeeze(0).permute(1, 0, 2).contiguous()
    beta_nt = beta.squeeze(0).permute(1, 0).contiguous()

    got = torch.ops.npu.npu_chunk_kda_fwd(
        q_ntd,
        k_ntd,
        v_ntd,
        gk_ntd,
        beta_nt,
        scale,
        64,
        initial_state=initial_state,
        output_final_state=True,
        return_intermediate=True,
    )
    ref = chunk_kda_forward_reference(
        q.detach().cpu(),
        k.detach().cpu(),
        v.detach().cpu(),
        gk.detach().cpu(),
        beta.detach().cpu(),
        scale=scale,
        chunk_size=64,
        initial_state=initial_state.detach().cpu(),
        output_final_state=True,
    )

    _assert_close("o ntd", got[0], ref.o.squeeze(0).permute(1, 0, 2), rtol=2e-2, atol=2e-2)
    _assert_close("final_state ntd", got[1], ref.final_state, rtol=2e-2, atol=2e-2)
    _assert_close("g ntd", got[2], gk_ntd, rtol=2e-2, atol=2e-2)
    _assert_close("Aqk ntd", got[3], ref.Aqk.squeeze(0).permute(1, 0, 2), rtol=2e-2, atol=2e-2)
    _assert_close("Akk ntd", got[4], ref.Akk.squeeze(0).permute(1, 0, 2), rtol=2e-2, atol=2e-2)
    _assert_close("w ntd", got[5], ref.w.squeeze(0).permute(1, 0, 2), rtol=2e-2, atol=2e-2)
    _assert_close("u ntd", got[6], ref.u.squeeze(0).permute(1, 0, 2), rtol=2e-2, atol=2e-2)
    _assert_close("qg ntd", got[7], ref.qg.squeeze(0).permute(1, 0, 2), rtol=2e-2, atol=2e-2)
    _assert_close("kg ntd", got[8], ref.kg.squeeze(0).permute(1, 0, 2), rtol=2e-2, atol=2e-2)
    _assert_close("v_new ntd", got[9], ref.v_new.squeeze(0).permute(1, 0, 2), rtol=2e-2, atol=2e-2)
    _assert_close("h ntd", got[10], ref.h.squeeze(0).permute(1, 0, 2, 3), rtol=2e-2, atol=2e-2)
    _assert_close("initial_state ntd", got[11], initial_state)


def test_kda_gate_cumsum_default_and_fwd_integration():
    device = _device()
    if device.type == "cpu":
        return
    q, k, v, _, beta, initial_state = _make_inputs(
        device, h=1, hv=2, t=40, kdim=8, vdim=16, dtype=torch.float16
    )
    g_step = (torch.randn(1, 40, 2, 8, device=device, dtype=torch.bfloat16) * 0.001)
    gk = torch.ops.npu.npu_kda_gate_cumsum(g_step, 32)
    ref_gk = _kda_gate_cumsum_reference(g_step.detach().cpu(), 32)
    _assert_close("gate cumsum default", gk, ref_gk, rtol=2e-3, atol=2e-3)

    scale = q.shape[-1] ** -0.5
    got = torch.ops.npu.npu_chunk_kda_fwd(
        q,
        k,
        v,
        gk,
        beta,
        scale,
        32,
        initial_state=initial_state,
        output_final_state=True,
        return_intermediate=False,
    )
    ref = chunk_kda_forward_reference(
        q.detach().cpu(),
        k.detach().cpu(),
        v.detach().cpu(),
        ref_gk,
        beta.detach().cpu(),
        scale=scale,
        chunk_size=32,
        initial_state=initial_state.detach().cpu(),
        output_final_state=True,
    )
    _assert_close("gate cumsum fwd o", got[0], ref.o, rtol=2e-2, atol=2e-2)
    _assert_close("gate cumsum fwd state", got[1], ref.final_state, rtol=2e-2, atol=2e-2)
    _assert_close("gate cumsum fwd g", got[2], gk, rtol=2e-2, atol=2e-2)
    _assert_close("gate cumsum fwd initial_state", got[11], initial_state)


def test_kda_gate_cumsum_bnsd_direct_matches_reference():
    device = _device()
    if device.type == "cpu":
        return
    torch.manual_seed(6789)
    g_bsnd = (torch.randn(1, 40, 2, 8, device=device, dtype=torch.bfloat16) * 0.001)
    g_bnsd = g_bsnd.permute(0, 2, 1, 3).contiguous()
    got = torch.ops.npu.npu_kda_gate_cumsum(g_bnsd, 32)
    ref = _kda_gate_cumsum_reference(g_bsnd.detach().cpu(), 32).permute(0, 2, 1, 3)
    _assert_close("gate cumsum bnsd", got, ref, rtol=2e-3, atol=2e-3)


def test_kda_gate_cumsum_ntd_direct_matches_reference():
    device = _device()
    if device.type == "cpu":
        return
    torch.manual_seed(7890)
    g_bsnd = (torch.randn(1, 40, 2, 8, device=device, dtype=torch.bfloat16) * 0.001)
    g_ntd = g_bsnd.squeeze(0).permute(1, 0, 2).contiguous()
    got = torch.ops.npu.npu_kda_gate_cumsum(g_ntd, 32)
    ref = _kda_gate_cumsum_reference(g_bsnd.squeeze(0).detach().cpu(), 32).permute(1, 0, 2)
    _assert_close("gate cumsum ntd", got, ref, rtol=2e-3, atol=2e-3)


def test_kda_gate_cumsum_safe_gate_matches_reference():
    device = _device()
    if device.type == "cpu":
        return
    torch.manual_seed(5678)
    raw = (torch.randn(1, 40, 2, 8, device=device, dtype=torch.bfloat16) * 0.5)
    a_log = torch.randn(2, device=device, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(2, 8, device=device, dtype=torch.float32) * 0.1
    got = torch.ops.npu.npu_kda_gate_cumsum(
        raw,
        32,
        A_log=a_log,
        dt_bias=dt_bias,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
    )
    ref = _kda_gate_cumsum_reference(
        raw.detach().cpu(),
        32,
        A_log=a_log.detach().cpu(),
        dt_bias=dt_bias.detach().cpu(),
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
    )
    _assert_close("gate cumsum safe", got, ref, rtol=2e-3, atol=2e-3)


def test_kda_gate_cumsum_safe_gate_multitask_last_row_matches_reference():
    device = _device()
    if device.type == "cpu":
        return
    torch.manual_seed(20260707)
    chunk_size = 64
    raw = torch.randn(1, 1536, 2, 128, device=device, dtype=torch.bfloat16)
    a_log = torch.log(torch.empty(2, device=device, dtype=torch.float32).uniform_(1, 16))
    dt_bias = torch.randn(2 * 128, device=device, dtype=torch.float32)
    got = torch.ops.npu.npu_kda_gate_cumsum(
        raw,
        chunk_size,
        A_log=a_log,
        dt_bias=dt_bias,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
    )
    ref = _kda_gate_cumsum_reference(
        raw.detach().cpu(),
        chunk_size,
        A_log=a_log.detach().cpu(),
        dt_bias=dt_bias.detach().cpu(),
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
    )
    _assert_close("gate cumsum safe multitask", got, ref, rtol=2e-3, atol=2e-3)


if __name__ == "__main__":
    test_chunk_kda_fwd_matches_reference()
    test_chunk_kda_fwd_chunk128_v128_gva_varlen()
    test_chunk_kda_fwd_bf16_chunk32_matches_reference()
    test_chunk_kda_fwd_bf16_gate_matches_reference()
    test_chunk_kda_fwd_fp16_matches_reference()
    test_chunk_kda_fwd_tnd_matches_reference()
    test_kda_gate_cumsum_default_and_fwd_integration()
    test_kda_gate_cumsum_bnsd_direct_matches_reference()
    test_kda_gate_cumsum_ntd_direct_matches_reference()
    test_kda_gate_cumsum_safe_gate_matches_reference()
    test_kda_gate_cumsum_safe_gate_multitask_last_row_matches_reference()
