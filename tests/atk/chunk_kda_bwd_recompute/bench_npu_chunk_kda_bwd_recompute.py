#!/usr/bin/env python3
"""NPU Event timing for fused chunk_kda_bwd_recompute.

Default model shape: B=1, HK=HV=96, T=8192, K=V=128, bf16, chunk_size=64.

Does not D2H outputs each iteration (ATK performance_device would).
Tensors are generated on CPU then copied to NPU.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time

SHAPE = dict(
    B=int(os.environ.get("KDA_BWD_B", "1")),
    HK=int(os.environ.get("KDA_BWD_HK", os.environ.get("KDA_BWD_H", "96"))),
    HV=int(os.environ.get("KDA_BWD_HV", os.environ.get("KDA_BWD_H", "96"))),
    T=int(os.environ.get("KDA_BWD_T", "8192")),
    K=128,
    V=128,
    chunk_size=64,
)
WARMUP = int(os.environ.get("KDA_BWD_WARMUP", "5"))
ITERS = int(os.environ.get("KDA_BWD_ITERS", "10"))
SEED = 20260817


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


def _npu_us(fn, iters: int) -> list[float]:
    import torch

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        torch.npu.synchronize()
        start.record()
        fn()
        end.record()
        torch.npu.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return times


def main() -> int:
    import torch
    import torch_npu  # noqa: F401
    from fla_npu.ops.ascendc import chunk_kda_bwd_recompute

    B, HK, HV, T, K, V, BT = (
        SHAPE["B"],
        SHAPE["HK"],
        SHAPE["HV"],
        SHAPE["T"],
        SHAPE["K"],
        SHAPE["V"],
        SHAPE["chunk_size"],
    )
    device = torch.device("npu:0")
    torch.npu.set_device(device)

    q = _randn((B, HK, T, K), torch.bfloat16, SEED + 1).to(device)
    k = _randn((B, HK, T, K), torch.bfloat16, SEED + 2).to(device)
    v = _randn((B, HV, T, V), torch.bfloat16, SEED + 3).to(device)
    g = _randn((B, HV, T, K), torch.bfloat16, SEED + 4, 0.05).to(device)
    beta = _rand((B, HV, T), torch.bfloat16, SEED + 5).to(device)
    a = _randn((B, HV, T, BT), torch.bfloat16, SEED + 6).to(device)
    a_log = _randn((HV,), torch.float32, SEED + 7, 0.05).to(device)
    dt_bias = _randn((HV, K), torch.float32, SEED + 8, 0.05).to(device)
    torch.npu.synchronize()

    def run():
        return chunk_kda_bwd_recompute(
            q,
            k,
            v,
            g,
            beta,
            a,
            BT,
            A_log=a_log,
            dt_bias=dt_bias,
            use_gate_in_kernel=True,
            use_exp2=True,
            lower_bound=-5.0,
        )

    t0 = time.perf_counter()
    outs = run()
    torch.npu.synchronize()
    first_ms = (time.perf_counter() - t0) * 1000.0
    gk, w, u, qg, kg = outs

    for _ in range(WARMUP):
        run()
    torch.npu.synchronize()

    times = _npu_us(run, ITERS)
    xs = sorted(times)
    report = {
        "device": torch.npu.get_device_name(0),
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "shape": SHAPE,
        "first_launch_ms": round(first_ms, 3),
        "warmup": WARMUP,
        "iters": ITERS,
        "mean_us": round(statistics.fmean(xs), 3),
        "median_us": round(statistics.median(xs), 3),
        "std_us": round(statistics.pstdev(xs), 3),
        "min_us": round(xs[0], 3),
        "max_us": round(xs[-1], 3),
        "outputs": {
            "gk": [int(x) for x in gk.shape],
            "w": [int(x) for x in w.shape],
            "u": [int(x) for x in u.shape],
            "qg": [int(x) for x in qg.shape],
            "kg": [int(x) for x in kg.shape],
        },
        "hbm_allocated_mb": round(torch.npu.memory_allocated(device) / (1024 * 1024), 1),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
