#!/usr/bin/env python3
"""NVIDIA GPU timing for the FLA chain that matches NPU chunk_kda_bwd_recompute.

NPU fused kernel = kda_gate_chunk_cumsum (safe-gate + chunk cumsum, exp2 scale)
                 + recompute_w_u_fwd (w/u/qg/kg).

GPU has no single fused kernel; this script times the two Triton ops as one
chain (CUDA events) and also reports each op. Layout is FLA native [B,T,H,K].

Same model case as ATK perf JSON:
  B=1, HK=HV=4, T=512, K=V=128, chunk_size=64, bf16, lower_bound=-5.0

Run on a CUDA host with flash-linear-attention installed, e.g.:

  export FLA_DISABLE_BACKEND_DISPATCH=1
  python3 bench_gpu_chunk_kda_bwd_recompute.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

RCP_LN2 = 1.4426950216
SHAPE = dict(B=1, HK=4, HV=4, T=512, K=128, V=128, chunk_size=64)
WARMUP = 20
ITERS = 50
SEED = 20260817


def _cuda_ms(fn, iters: int) -> list[float]:
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)  # ms -> us
    return times


def main() -> int:
    try:
        import torch
        import triton
    except ImportError as exc:
        print(f"NEED_TORCH_TRITON: {exc}", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("CUDA_NOT_AVAILABLE", file=sys.stderr)
        return 2

    repo_candidates = [
        Path(__file__).resolve().parents[3] / "flash-linear-attention",
        Path("/data/l00906020/flash-linear-attention"),
        Path.home() / "flash-linear-attention",
    ]
    for root in repo_candidates:
        if (root / "fla" / "ops" / "kda" / "wy_fast.py").is_file():
            sys.path.insert(0, str(root))
            break

    try:
        from fla.ops.kda.gate import kda_gate_chunk_cumsum
        from fla.ops.kda.wy_fast import recompute_w_u_fwd
    except ImportError as exc:
        print(
            "NEED_FLA: pip install flash-linear-attention  "
            f"or clone fla-org/flash-linear-attention onto PYTHONPATH ({exc})",
            file=sys.stderr,
        )
        return 2

    B, HK, HV, T, K, V, BT = (
        SHAPE["B"],
        SHAPE["HK"],
        SHAPE["HV"],
        SHAPE["T"],
        SHAPE["K"],
        SHAPE["V"],
        SHAPE["chunk_size"],
    )
    device = torch.device("cuda")
    g = torch.Generator(device="cpu")
    g.manual_seed(SEED)

    def randn(*shape, dtype=torch.bfloat16, scale=1.0):
        t = torch.randn(*shape, generator=g, dtype=torch.float32) * scale
        return t.to(dtype=dtype, device=device)

    # FLA layout [B, T, H, D]
    q = randn(B, T, HK, K)
    k = randn(B, T, HK, K)
    v = randn(B, T, HV, V)
    gate_in = randn(B, T, HV, K, scale=0.05)
    beta = (torch.rand(B, T, HV, generator=g) * 0.8 + 0.1).to(dtype=torch.bfloat16, device=device)
    A = randn(B, T, HV, BT)
    A_log = randn(HV, dtype=torch.float32, scale=0.05)
    dt_bias = randn(HV, K, dtype=torch.float32, scale=0.05)

    def gate():
        return kda_gate_chunk_cumsum(
            g=gate_in,
            A_log=A_log,
            chunk_size=BT,
            scale=RCP_LN2,
            dt_bias=dt_bias,
            lower_bound=-5.0,
        )

    def wy(gk):
        return recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, gk=gk, q=q)

    def chain():
        gk = gate()
        return wy(gk)

    # compile / autotune
    for _ in range(3):
        chain()
    torch.cuda.synchronize()
    gk = gate()
    w, u, qg, kg = wy(gk)
    torch.cuda.synchronize()

    for _ in range(WARMUP):
        chain()
    torch.cuda.synchronize()

    e2e = _cuda_ms(chain, ITERS)
    gate_us = _cuda_ms(lambda: gate(), ITERS)
    gk_hold = gate()
    torch.cuda.synchronize()
    wy_us = _cuda_ms(lambda: wy(gk_hold), ITERS)

    def stats(xs: list[float]) -> dict:
        xs = sorted(xs)
        return {
            "mean_us": round(statistics.fmean(xs), 3),
            "median_us": round(statistics.median(xs), 3),
            "std_us": round(statistics.pstdev(xs), 3),
            "min_us": round(xs[0], 3),
            "max_us": round(xs[-1], 3),
        }

    report = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "triton": getattr(triton, "__version__", "unknown"),
        "cuda": torch.version.cuda,
        "shape": SHAPE,
        "outputs": {
            "gk": [int(x) for x in gk.shape],
            "w": [int(x) for x in w.shape],
            "u": [int(x) for x in u.shape],
            "qg": None if qg is None else [int(x) for x in qg.shape],
            "kg": [int(x) for x in kg.shape],
        },
        "gpu_chain_e2e": stats(e2e),
        "gpu_kda_gate_chunk_cumsum": stats(gate_us),
        "gpu_recompute_w_u_fwd": stats(wy_us),
        "npu_fused_atk_profiler_us": 359.50,
        "npu_shape_note": "ATK performance_device, Ascend950PR, Mix 1 AIC : 2 AIV",
        "compare_mean_us": {
            "npu_fused": 359.50,
            "gpu_chain": stats(e2e)["mean_us"],
            "speedup_npu_over_gpu": round(stats(e2e)["mean_us"] / 359.50, 3),
        },
        "warmup": WARMUP,
        "iters": ITERS,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
