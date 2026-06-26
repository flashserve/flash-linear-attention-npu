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
    q = (torch.randn(b, h, t, kdim, device=device, dtype=dtype) * 0.08).requires_grad_(True)
    k = (torch.randn(b, h, t, kdim, device=device, dtype=dtype) * 0.08).requires_grad_(True)
    v = (torch.randn(b, hv, t, vdim, device=device, dtype=dtype) * 0.08).requires_grad_(True)
    gk = (torch.randn(b, hv, t, kdim, device=device, dtype=dtype).cumsum(dim=2) * 0.001).requires_grad_(True)
    beta = torch.sigmoid(torch.randn(b, hv, t, device=device, dtype=dtype)).requires_grad_(True)
    initial_state = (torch.randn(b, hv, kdim, vdim, device=device, dtype=dtype) * 0.02).requires_grad_(True)
    return q, k, v, gk, beta, initial_state


def _assert_close(name, actual, expected, rtol=2e-3, atol=2e-3):
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=rtol, atol=atol, msg=name)


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

    ref_grads = torch.autograd.grad(
        [fwd[0], fwd[1]],
        [q, k, v, gk, beta, initial_state],
        grad_outputs=[d_o, dht],
        allow_unused=False,
        retain_graph=False,
    )
    expected = (ref_grads[0], ref_grads[1], ref_grads[2], ref_grads[4].float(), ref_grads[3].float(), ref_grads[5])
    for name, actual, ref in zip(("dq", "dk", "dv", "dbeta", "dgk", "dh0"), got, expected):
        _assert_close(name, actual, ref, rtol=3e-3, atol=3e-3)


if __name__ == "__main__":
    test_chunk_kda_fwd_matches_reference()
    test_chunk_kda_fwd_chunk128_v256_gva_varlen()
    test_chunk_kda_bwd_matches_autograd()
