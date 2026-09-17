#!/usr/bin/env python3
"""CPU fused-ref vs NPU ChunkKdaBwdRecompute. No extra golden files."""

from __future__ import annotations

import argparse
import os

RCP_LN2 = 1.4426950216
LIMITS = {"gk": 0.05, "qg": 0.05, "kg": 0.05, "w": 0.15, "u": 0.15}


def _randn(shape, dtype, seed, scale=1.0):
    import torch

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return (torch.randn(shape, generator=gen, dtype=torch.float32) * scale).to(dtype)


def _rand(shape, dtype, seed, low=0.1, high=0.9):
    import torch

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    data = torch.rand(shape, generator=gen, dtype=torch.float32)
    return (data * (high - low) + low).to(dtype)


def _fused_ref(q, k, v, g, beta, a, a_log, dt_bias, chunk_size: int):
    import torch

    calc = torch.float32
    qf, kf, vf, gf, betaf, af = (t.to(calc) for t in (q, k, v, g, beta, a))
    a_log_f = a_log.to(calc)
    dt_f = dt_bias.to(calc)
    hv = gf.shape[1]
    tokens = gf.shape[2]
    x = gf + dt_f.view(1, hv, 1, -1)
    g_corr = (-5.0) * torch.sigmoid(torch.exp(a_log_f.view(1, hv, 1, 1)) * x)
    gk = torch.empty_like(g_corr)
    for start in range(0, tokens, chunk_size):
        end = min(start + chunk_size, tokens)
        gk[:, :, start:end] = torch.cumsum(g_corr[:, :, start:end], dim=2) * RCP_LN2
    e2 = torch.exp2(gk)
    group = max(int(hv) // int(qf.shape[1]), 1)
    q_hv = qf.repeat_interleave(group, dim=1)
    k_hv = kf.repeat_interleave(group, dim=1)
    beta_k = betaf.unsqueeze(-1)
    qg = q_hv * e2
    kbg = k_hv * beta_k * e2
    vb = vf * beta_k
    orig = q.dtype
    kg = torch.empty_like(k_hv)
    w = torch.empty((q.shape[0], hv, tokens, k.shape[-1]), dtype=calc)
    u = torch.empty((v.shape[0], hv, tokens, v.shape[-1]), dtype=calc)
    for start in range(0, tokens, chunk_size):
        end = min(start + chunk_size, tokens)
        length = end - start
        gk_last = gk[:, :, end - 1 : end, :]
        kg[:, :, start:end] = k_hv[:, :, start:end] * torch.exp2(gk_last - gk[:, :, start:end])
        a_tile = af[:, :, start:end, :length].to(orig)
        w[:, :, start:end] = torch.matmul(a_tile, kbg[:, :, start:end].to(orig)).to(calc)
        u[:, :, start:end] = torch.matmul(a_tile, vb[:, :, start:end].to(orig)).to(calc)
    return gk, w.to(orig), u.to(orig), qg.to(orig), kg.to(orig)


def _run_one(batch: int, hk: int, hv: int, tokens: int, seed: int) -> int:
    import torch
    import torch_npu  # noqa: F401
    from fla_npu.ops.ascendc import chunk_kda_bwd_recompute

    torch.npu.set_device(0)
    chunk_size = 64
    d = torch.bfloat16
    q = _randn((batch, hk, tokens, 128), d, seed + 1)
    k = _randn((batch, hk, tokens, 128), d, seed + 2)
    v = _randn((batch, hv, tokens, 128), d, seed + 3)
    g = _randn((batch, hv, tokens, 128), d, seed + 4, 0.05)
    beta = _rand((batch, hv, tokens), d, seed + 5)
    a = _randn((batch, hv, tokens, chunk_size), d, seed + 6)
    a_log = _randn((hv,), torch.float32, seed + 7, 0.05)
    dt_bias = _randn((hv, 128), torch.float32, seed + 8, 0.05)
    ref = _fused_ref(q, k, v, g, beta, a, a_log, dt_bias, chunk_size)

    def launch():
        return chunk_kda_bwd_recompute(
            q.npu(), k.npu(), v.npu(), g.npu(), beta.npu(), a.npu(), chunk_size,
            A_log=a_log.npu(), dt_bias=dt_bias.npu(),
            use_gate_in_kernel=True, use_exp2=True, lower_bound=-5.0,
        )

    first = launch()
    torch.npu.synchronize()
    second = launch()
    torch.npu.synchronize()
    names = ("gk", "w", "u", "qg", "kg")
    print(f"device={torch.npu.get_device_name(0)} visible={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}")
    print(f"shape B={batch} HK={hk} HV={hv} T={tokens}")
    ok = True
    for name, got, again, cpu in zip(names, first, second, ref):
        g1 = got.float().cpu()
        g2 = again.float().cpu()
        if not torch.isfinite(g1).all():
            print(f"{name}: NON_FINITE")
            ok = False
            continue
        launch_diff = (g1 - g2).abs().max().item()
        max_diff = (g1 - cpu.float()).abs().max().item()
        limit = LIMITS[name]
        status = "ok" if max_diff < limit and launch_diff == 0.0 else "FAIL"
        if status != "ok":
            ok = False
        print(f"{name}: max_diff={max_diff:.6f} launch_diff={launch_diff:.6g} limit={limit} {status}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--hk", type=int, default=8)
    parser.add_argument("--hv", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()
    return _run_one(args.batch, args.hk, args.hv, args.tokens, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
