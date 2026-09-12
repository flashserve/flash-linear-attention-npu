#!/usr/bin/env python3
"""Smoke test for npu_chunk_kda_bwd_recompute against an inline CPU fused ref."""

from __future__ import annotations

import torch

RCP_LN2 = 1.4426950216
LIMITS = {"gk": 0.05, "qg": 0.05, "kg": 0.05, "w": 0.15, "u": 0.15}


def make_inputs(*, batch=1, hk=2, hv=4, tokens=128, dtype=torch.bfloat16):
    q = torch.randn(batch, hk, tokens, 128, dtype=dtype)
    k = torch.randn(batch, hk, tokens, 128, dtype=dtype)
    v = torch.randn(batch, hv, tokens, 128, dtype=dtype)
    g = torch.randn(batch, hv, tokens, 128, dtype=dtype)
    beta = torch.rand(batch, hv, tokens, dtype=dtype)
    a_log = torch.randn(hv, dtype=torch.float32)
    dt_bias = torch.randn(hv, 128, dtype=torch.float32)
    a = torch.randn(batch, hv, tokens, 64, dtype=dtype)
    return {
        "q": q, "k": k, "v": v, "g": g, "beta": beta,
        "A_log": a_log, "dt_bias": dt_bias, "A": a,
    }


def fused_cpu(inputs, chunk_size=64, lower_bound=-5.0):
    calc = torch.float32
    q, k, v, g, beta, a = (inputs[n].to(calc) for n in ("q", "k", "v", "g", "beta", "A"))
    a_log = inputs["A_log"].to(calc)
    dt_bias = inputs["dt_bias"].to(calc)
    hv = g.shape[1]
    tokens = g.shape[2]
    x = g + dt_bias.view(1, hv, 1, -1)
    g_corr = lower_bound * torch.sigmoid(torch.exp(a_log.view(1, hv, 1, 1)) * x)
    gk = torch.empty_like(g_corr)
    for start in range(0, tokens, chunk_size):
        end = min(start + chunk_size, tokens)
        gk[:, :, start:end] = torch.cumsum(g_corr[:, :, start:end], dim=2) * RCP_LN2
    e2 = torch.exp2(gk)
    group = max(int(hv) // int(q.shape[1]), 1)
    q_hv = q.repeat_interleave(group, dim=1)
    k_hv = k.repeat_interleave(group, dim=1)
    beta_k = beta.unsqueeze(-1)
    qg = q_hv * e2
    kbg = k_hv * beta_k * e2
    vb = v * beta_k
    orig = inputs["q"].dtype
    kg = torch.empty_like(k_hv)
    w = torch.empty((q.shape[0], hv, tokens, k.shape[-1]), dtype=calc)
    u = torch.empty((v.shape[0], hv, tokens, v.shape[-1]), dtype=calc)
    for start in range(0, tokens, chunk_size):
        end = min(start + chunk_size, tokens)
        length = end - start
        gk_last = gk[:, :, end - 1 : end, :]
        kg[:, :, start:end] = k_hv[:, :, start:end] * torch.exp2(gk_last - gk[:, :, start:end])
        a_tile = a[:, :, start:end, :length].to(orig)
        w[:, :, start:end] = torch.matmul(a_tile, kbg[:, :, start:end].to(orig)).to(calc)
        u[:, :, start:end] = torch.matmul(a_tile, vb[:, :, start:end].to(orig)).to(calc)
    return {
        "gk": gk, "w": w.to(orig), "u": u.to(orig),
        "qg": qg.to(orig), "kg": kg.to(orig),
    }


def main() -> int:
    if not torch.npu.is_available():
        print("NPU not available, skip.")
        return 0

    from fla_npu.ops.ascendc import chunk_kda_bwd_recompute

    inputs = make_inputs()
    chunk_size = 64
    npu_in = {name: tensor.npu() for name, tensor in inputs.items()}
    gk, w, u, qg, kg = chunk_kda_bwd_recompute(
        npu_in["q"], npu_in["k"], npu_in["v"], npu_in["g"], npu_in["beta"], npu_in["A"],
        chunk_size,
        A_log=npu_in["A_log"],
        dt_bias=npu_in["dt_bias"],
        use_gate_in_kernel=True,
        use_exp2=True,
        lower_bound=-5.0,
    )
    expected = fused_cpu(inputs, chunk_size=chunk_size)

    def report(name, got, ref):
        max_diff = (got.float().cpu() - ref.float()).abs().max().item()
        print(f"{name}: max_diff={max_diff:.6f}")
        return max_diff

    diffs = {
        "gk": report("gk", gk, expected["gk"]),
        "w": report("w", w, expected["w"]),
        "u": report("u", u, expected["u"]),
        "qg": report("qg", qg, expected["qg"]),
        "kg": report("kg", kg, expected["kg"]),
    }
    ok = all(diffs[name] < LIMITS[name] for name in diffs)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
