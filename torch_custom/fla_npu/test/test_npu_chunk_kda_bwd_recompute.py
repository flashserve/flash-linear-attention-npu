#!/usr/bin/env python3
"""Smoke test for npu_chunk_kda_bwd_recompute against CPU golden."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
GOLDEN_DIR = ROOT / "kda_gate_wu_fusion_golden"
sys.path.insert(0, str(GOLDEN_DIR))

from kda_gate_wu_golden import fused_cpu  # noqa: E402


def make_inputs(*, batch=1, hk=2, hv=4, tokens=128, device="npu", dtype=torch.bfloat16):
    q = torch.randn(batch, hk, tokens, 128, device=device, dtype=dtype)
    k = torch.randn(batch, hk, tokens, 128, device=device, dtype=dtype)
    v = torch.randn(batch, hv, tokens, 128, device=device, dtype=dtype)
    g = torch.randn(batch, hv, tokens, 128, device=device, dtype=dtype)
    beta = torch.rand(batch, hv, tokens, device=device, dtype=dtype)
    a_log = torch.randn(hv, device=device, dtype=torch.float32)
    dt_bias = torch.randn(hv, 128, device=device, dtype=torch.float32)
    a = torch.randn(batch, hv, tokens, 64, device=device, dtype=dtype)
    return {
        "q": q, "k": k, "v": v, "g": g, "beta": beta,
        "A_log": a_log, "dt_bias": dt_bias, "A": a,
    }


def main() -> int:
    if not torch.npu.is_available():
        print("NPU not available, skip.")
        return 0

    from fla_npu.ops.ascendc import chunk_kda_bwd_recompute

    inputs = make_inputs()
    chunk_size = 64
    gk, w, u, qg, kg = chunk_kda_bwd_recompute(
        inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"], inputs["A"],
        chunk_size,
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        use_gate_in_kernel=True,
        use_exp2=True,
        lower_bound=-5.0,
    )

    cpu_inputs = {name: tensor.float().cpu() for name, tensor in inputs.items()}
    expected = fused_cpu(
        cpu_inputs,
        chunk_size=chunk_size,
        use_gate=True,
        safe_gate=True,
        lower_bound=-5.0,
        cu_seqlens=None,
    )

    def report(name, got, ref):
        ref_t = ref.to(got.dtype) if ref.dtype != got.dtype else ref
        max_diff = (got.float().cpu() - ref_t.float()).abs().max().item()
        print(f"{name}: max_diff={max_diff:.6f}")
        return max_diff

    diffs = {
        "gk": report("gk", gk, expected["gk"]),
        "w": report("w", w, expected["w"]),
        "u": report("u", u, expected["u"]),
        "qg": report("qg", qg, expected["qg"]),
        "kg": report("kg", kg, expected["kg"]),
    }
    # gk/qg/kg are vector; w/u are bf16 matmul vs fp32 golden.
    limits = {"gk": 0.05, "qg": 0.05, "kg": 0.05, "w": 0.15, "u": 0.15}
    ok = all(diffs[name] < limits[name] for name in diffs)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
