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
    initial_state = (torch.randn(b, hv, kdim, vdim, device=device, dtype=dtype) * 0.02).requires_grad_(True)
    return q, k, v, gk, beta, initial_state


def _assert_close(name, actual, expected, rtol=2e-3, atol=2e-3):
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=rtol, atol=atol, msg=name)


def _lower_inverse_autograd(mat):
    eye = torch.eye(mat.shape[-1], device=mat.device, dtype=torch.float32)
    lhs = eye + torch.tril(mat.to(torch.float32), diagonal=-1)
    return torch.linalg.solve_triangular(lhs, eye, upper=False)


def _chunk_kda_forward_autograd(q, k, v, gk, beta, scale, chunk_size, initial_state):
    bsz, total_t, hq, kdim = q.shape
    _, _, hv, vdim = v.shape
    group = hv // hq
    out_batches = []
    state_out = []
    for b in range(bsz):
        hv_outputs = [[] for _ in range(hv)]
        hv_states = [initial_state[b, ihv].to(torch.float32) for ihv in range(hv)]
        for start in range(0, total_t, chunk_size):
            end = min(start + chunk_size, total_t)
            cur_t = end - start
            for ihv in range(hv):
                ih = ihv // group
                q_blk = q[b, start:end, ih].to(torch.float32)
                k_blk = k[b, start:end, ih].to(torch.float32)
                v_blk = v[b, start:end, ihv].to(torch.float32)
                g_blk = gk[b, start:end, ihv].to(torch.float32)
                beta_blk = beta[b, start:end, ihv].to(torch.float32)

                rel = g_blk[:, None, :] - g_blk[None, :, :]
                qk = torch.einsum("ik,jk,ijk->ij", q_blk, k_blk, torch.exp2(rel)) * float(scale)
                kk = torch.einsum("ik,jk,ijk->ij", k_blk, k_blk, torch.exp2(rel))
                tril_qk = torch.tril(qk, diagonal=0)
                tril_kk = torch.tril(kk * beta_blk[:, None], diagonal=-1)
                inv_akk = _lower_inverse_autograd(tril_kk)

                k_beta_g = k_blk * beta_blk[:, None] * torch.exp2(g_blk)
                v_beta = v_blk * beta_blk[:, None]
                w_blk = inv_akk @ k_beta_g
                u_blk = inv_akk @ v_beta

                last_g = g_blk[cur_t - 1]
                qg_blk = q_blk * torch.exp2(g_blk)
                kg_blk = k_blk * torch.exp2(last_g[None, :] - g_blk)

                h_prev = hv_states[ihv]
                v_new_blk = u_blk - w_blk @ h_prev
                hv_states[ihv] = torch.exp2(last_g)[:, None] * h_prev + kg_blk.T @ v_new_blk

                o_inter = qg_blk @ h_prev * float(scale)
                hv_outputs[ihv].append(o_inter + tril_qk @ v_new_blk)
        out_batches.append(torch.stack([torch.cat(hv_outputs[ihv], dim=0) for ihv in range(hv)], dim=1))
        state_out.append(torch.stack(hv_states, dim=0))
    return torch.stack(out_batches, dim=0).to(v.dtype), torch.stack(state_out, dim=0).to(initial_state.dtype)


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
    _assert_close("Aqk", got[2], ref.Aqk)
    _assert_close("Akk", got[3], ref.Akk)
    _assert_close("w", got[4], ref.w)
    _assert_close("u", got[5], ref.u)
    _assert_close("qg", got[6], ref.qg)
    _assert_close("kg", got[7], ref.kg)
    _assert_close("v_new", got[8], ref.v_new)


def test_chunk_kda_fwd_chunk128_v256_gva_varlen():
    device = _device()
    q, k, v, gk, beta, _ = _make_inputs(device, h=1, hv=2, t=16, kdim=8, vdim=256)
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
    _assert_close("o chunk128 v256 gva varlen", got[0], ref.o, rtol=3e-3, atol=3e-3)
    _assert_close("final_state chunk128 v256 gva varlen", got[1], ref.final_state, rtol=3e-3, atol=3e-3)


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
    _assert_close("Aqk fp16", got[2], ref.Aqk, rtol=2e-2, atol=2e-2)
    _assert_close("Akk fp16", got[3], ref.Akk, rtol=2e-2, atol=2e-2)
    _assert_close("w fp16", got[4], ref.w, rtol=2e-2, atol=2e-2)
    _assert_close("u fp16", got[5], ref.u, rtol=2e-2, atol=2e-2)
    _assert_close("qg fp16", got[6], ref.qg, rtol=2e-2, atol=2e-2)
    _assert_close("kg fp16", got[7], ref.kg, rtol=2e-2, atol=2e-2)
    _assert_close("v_new fp16", got[8], ref.v_new, rtol=2e-2, atol=2e-2)


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
    _assert_close("Aqk tnd", got[2], ref.Aqk.squeeze(0))
    _assert_close("Akk tnd", got[3], ref.Akk.squeeze(0))
    _assert_close("w tnd", got[4], ref.w.squeeze(0))
    _assert_close("u tnd", got[5], ref.u.squeeze(0))
    _assert_close("qg tnd", got[6], ref.qg.squeeze(0))
    _assert_close("kg tnd", got[7], ref.kg.squeeze(0))
    _assert_close("v_new tnd", got[8], ref.v_new.squeeze(0))
    _assert_close("h tnd", got[9], ref.h.squeeze(0))


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


def test_chunk_kda_bwd_matches_autograd():
    device = _device()
    q, k, v, gk, beta, initial_state = _make_inputs(device, h=1, hv=1, t=8, kdim=8, vdim=8)
    scale = q.shape[-1] ** -0.5
    fwd = torch.ops.npu.npu_chunk_kda_fwd(
        q,
        k,
        v,
        gk,
        beta,
        scale,
        64,
        initial_state=initial_state,
        output_final_state=True,
        return_intermediate=False,
    )
    d_o = torch.randn_like(fwd[0])
    dht = torch.randn_like(fwd[1])

    got = torch.ops.npu.npu_chunk_kda_bwd(
        q,
        k,
        v,
        gk,
        beta,
        fwd[2],
        fwd[3],
        d_o,
        scale,
        64,
        initial_state=initial_state,
        dht=dht,
    )

    q_ref = q.detach().cpu().requires_grad_(True)
    k_ref = k.detach().cpu().requires_grad_(True)
    v_ref = v.detach().cpu().requires_grad_(True)
    gk_ref = gk.detach().cpu().requires_grad_(True)
    beta_ref = beta.detach().cpu().requires_grad_(True)
    initial_state_ref = initial_state.detach().cpu().requires_grad_(True)
    ref_o, ref_final_state = _chunk_kda_forward_autograd(
        q_ref,
        k_ref,
        v_ref,
        gk_ref,
        beta_ref,
        scale,
        64,
        initial_state_ref,
    )
    ref_grads = torch.autograd.grad(
        [ref_o, ref_final_state],
        [q_ref, k_ref, v_ref, gk_ref, beta_ref, initial_state_ref],
        grad_outputs=[d_o.detach().cpu(), dht.detach().cpu()],
        allow_unused=False,
        retain_graph=False,
    )
    expected = (ref_grads[0], ref_grads[1], ref_grads[2], ref_grads[4].float(), ref_grads[3].float(), ref_grads[5])
    for name, actual, ref in zip(("dq", "dk", "dv", "dbeta", "dgk", "dh0"), got, expected):
        _assert_close(name, actual, ref, rtol=3e-3, atol=3e-3)


def test_chunk_kda_bwd_fp16_matches_autograd():
    device = _device()
    if device.type == "cpu":
        return
    q, k, v, gk, beta, initial_state = _make_inputs(
        device, h=1, hv=1, t=8, kdim=8, vdim=32, dtype=torch.float16
    )
    scale = q.shape[-1] ** -0.5
    fwd = torch.ops.npu.npu_chunk_kda_fwd(
        q,
        k,
        v,
        gk,
        beta,
        scale,
        64,
        initial_state=initial_state,
        output_final_state=True,
        return_intermediate=False,
    )
    d_o = torch.randn_like(fwd[0])
    dht = torch.randn_like(fwd[1])

    got = torch.ops.npu.npu_chunk_kda_bwd(
        q,
        k,
        v,
        gk,
        beta,
        fwd[2],
        fwd[3],
        d_o,
        scale,
        64,
        initial_state=initial_state,
        dht=dht,
    )

    q_ref = q.detach().cpu().requires_grad_(True)
    k_ref = k.detach().cpu().requires_grad_(True)
    v_ref = v.detach().cpu().requires_grad_(True)
    gk_ref = gk.detach().cpu().requires_grad_(True)
    beta_ref = beta.detach().cpu().requires_grad_(True)
    initial_state_ref = initial_state.detach().cpu().requires_grad_(True)
    ref_o, ref_final_state = _chunk_kda_forward_autograd(
        q_ref,
        k_ref,
        v_ref,
        gk_ref,
        beta_ref,
        scale,
        64,
        initial_state_ref,
    )
    ref_grads = torch.autograd.grad(
        [ref_o, ref_final_state],
        [q_ref, k_ref, v_ref, gk_ref, beta_ref, initial_state_ref],
        grad_outputs=[d_o.detach().cpu(), dht.detach().cpu()],
        allow_unused=False,
        retain_graph=False,
    )
    expected = (ref_grads[0], ref_grads[1], ref_grads[2], ref_grads[4].float(), ref_grads[3].float(), ref_grads[5])
    for name, actual, ref in zip(("dq", "dk", "dv", "dbeta", "dgk", "dh0"), got, expected):
        _assert_close(f"{name} fp16", actual, ref, rtol=2e-2, atol=2e-2)


def test_chunk_kda_bwd_bf16_gate_matches_autograd():
    device = _device()
    if device.type == "cpu":
        return
    q, k, v, gk, beta, initial_state = _make_inputs(
        device, h=1, hv=1, t=8, kdim=8, vdim=32, dtype=torch.float16
    )
    gk_bf16 = gk.detach().to(torch.bfloat16).requires_grad_(True)
    beta_bf16 = beta.detach().to(torch.bfloat16).requires_grad_(True)
    scale = q.shape[-1] ** -0.5
    fwd = torch.ops.npu.npu_chunk_kda_fwd(
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
    d_o = torch.randn_like(fwd[0])
    dht = torch.randn_like(fwd[1])

    got = torch.ops.npu.npu_chunk_kda_bwd(
        q,
        k,
        v,
        gk_bf16,
        beta_bf16,
        fwd[2],
        fwd[3],
        d_o,
        scale,
        64,
        initial_state=initial_state,
        dht=dht,
    )

    q_ref = q.detach().cpu().requires_grad_(True)
    k_ref = k.detach().cpu().requires_grad_(True)
    v_ref = v.detach().cpu().requires_grad_(True)
    gk_ref = gk_bf16.detach().cpu().float().requires_grad_(True)
    beta_ref = beta_bf16.detach().cpu().float().requires_grad_(True)
    initial_state_ref = initial_state.detach().cpu().requires_grad_(True)
    ref_o, ref_final_state = _chunk_kda_forward_autograd(
        q_ref,
        k_ref,
        v_ref,
        gk_ref,
        beta_ref,
        scale,
        64,
        initial_state_ref,
    )
    ref_grads = torch.autograd.grad(
        [ref_o, ref_final_state],
        [q_ref, k_ref, v_ref, gk_ref, beta_ref, initial_state_ref],
        grad_outputs=[d_o.detach().cpu(), dht.detach().cpu()],
        allow_unused=False,
        retain_graph=False,
    )
    expected = (ref_grads[0], ref_grads[1], ref_grads[2], ref_grads[4].float(), ref_grads[3].float(), ref_grads[5])
    for name, actual, ref in zip(("dq", "dk", "dv", "dbeta", "dgk", "dh0"), got, expected):
        _assert_close(f"{name} bf16 gate", actual, ref, rtol=2e-2, atol=2e-2)


def test_chunk_kda_bwd_chunk128_v256_gva_matches_autograd():
    device = _device()
    q, k, v, gk, beta, initial_state = _make_inputs(device, h=1, hv=2, t=8, kdim=8, vdim=256)
    scale = q.shape[-1] ** -0.5
    fwd = torch.ops.npu.npu_chunk_kda_fwd(
        q,
        k,
        v,
        gk,
        beta,
        scale,
        128,
        initial_state=initial_state,
        output_final_state=True,
        return_intermediate=False,
    )
    d_o = torch.randn_like(fwd[0])
    dht = torch.randn_like(fwd[1])

    got = torch.ops.npu.npu_chunk_kda_bwd(
        q,
        k,
        v,
        gk,
        beta,
        fwd[2],
        fwd[3],
        d_o,
        scale,
        128,
        initial_state=initial_state,
        dht=dht,
    )

    q_ref = q.detach().cpu().requires_grad_(True)
    k_ref = k.detach().cpu().requires_grad_(True)
    v_ref = v.detach().cpu().requires_grad_(True)
    gk_ref = gk.detach().cpu().requires_grad_(True)
    beta_ref = beta.detach().cpu().requires_grad_(True)
    initial_state_ref = initial_state.detach().cpu().requires_grad_(True)
    ref_o, ref_final_state = _chunk_kda_forward_autograd(
        q_ref,
        k_ref,
        v_ref,
        gk_ref,
        beta_ref,
        scale,
        128,
        initial_state_ref,
    )
    ref_grads = torch.autograd.grad(
        [ref_o, ref_final_state],
        [q_ref, k_ref, v_ref, gk_ref, beta_ref, initial_state_ref],
        grad_outputs=[d_o.detach().cpu(), dht.detach().cpu()],
        allow_unused=False,
        retain_graph=False,
    )
    expected = (ref_grads[0], ref_grads[1], ref_grads[2], ref_grads[4].float(), ref_grads[3].float(), ref_grads[5])
    for name, actual, ref in zip(("dq", "dk", "dv", "dbeta", "dgk", "dh0"), got, expected):
        _assert_close(f"{name} chunk128 v256 gva", actual, ref, rtol=3e-3, atol=3e-3)


def test_chunk_kda_bwd_tnd_matches_autograd():
    device = _device()
    q, k, v, gk, beta, initial_state = _make_inputs(device, b=1, h=1, hv=2, t=8, kdim=8, vdim=16)
    scale = q.shape[-1] ** -0.5
    fwd = torch.ops.npu.npu_chunk_kda_fwd(
        q.squeeze(0),
        k.squeeze(0),
        v.squeeze(0),
        gk.squeeze(0),
        beta.squeeze(0),
        scale,
        64,
        initial_state=initial_state,
        output_final_state=True,
        return_intermediate=False,
    )
    d_o = torch.randn_like(fwd[0])
    dht = torch.randn_like(fwd[1])

    got = torch.ops.npu.npu_chunk_kda_bwd(
        q.squeeze(0),
        k.squeeze(0),
        v.squeeze(0),
        gk.squeeze(0),
        beta.squeeze(0),
        fwd[2],
        fwd[3],
        d_o,
        scale,
        64,
        initial_state=initial_state,
        dht=dht,
    )

    q_ref = q.detach().cpu().requires_grad_(True)
    k_ref = k.detach().cpu().requires_grad_(True)
    v_ref = v.detach().cpu().requires_grad_(True)
    gk_ref = gk.detach().cpu().requires_grad_(True)
    beta_ref = beta.detach().cpu().requires_grad_(True)
    initial_state_ref = initial_state.detach().cpu().requires_grad_(True)
    ref_o, ref_final_state = _chunk_kda_forward_autograd(
        q_ref,
        k_ref,
        v_ref,
        gk_ref,
        beta_ref,
        scale,
        64,
        initial_state_ref,
    )
    ref_grads = torch.autograd.grad(
        [ref_o, ref_final_state],
        [q_ref, k_ref, v_ref, gk_ref, beta_ref, initial_state_ref],
        grad_outputs=[d_o.detach().cpu().unsqueeze(0), dht.detach().cpu()],
        allow_unused=False,
        retain_graph=False,
    )
    expected = (
        ref_grads[0].squeeze(0),
        ref_grads[1].squeeze(0),
        ref_grads[2].squeeze(0),
        ref_grads[4].float().squeeze(0),
        ref_grads[3].float().squeeze(0),
        ref_grads[5],
    )
    for name, actual, ref in zip(("dq", "dk", "dv", "dbeta", "dgk", "dh0"), got, expected):
        _assert_close(f"{name} tnd", actual, ref, rtol=3e-3, atol=3e-3)


if __name__ == "__main__":
    test_chunk_kda_fwd_matches_reference()
    test_chunk_kda_fwd_chunk128_v256_gva_varlen()
    test_chunk_kda_fwd_bf16_chunk32_matches_reference()
    test_chunk_kda_fwd_bf16_gate_matches_reference()
    test_chunk_kda_fwd_fp16_matches_reference()
    test_chunk_kda_fwd_tnd_matches_reference()
    test_kda_gate_cumsum_default_and_fwd_integration()
    test_kda_gate_cumsum_safe_gate_matches_reference()
    test_chunk_kda_bwd_matches_autograd()
    test_chunk_kda_bwd_fp16_matches_autograd()
    test_chunk_kda_bwd_bf16_gate_matches_autograd()
    test_chunk_kda_bwd_chunk128_v256_gva_matches_autograd()
    test_chunk_kda_bwd_tnd_matches_autograd()
