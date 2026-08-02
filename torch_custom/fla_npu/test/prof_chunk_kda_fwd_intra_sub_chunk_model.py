#!/usr/bin/env python3
"""Model-case microbench / msprof target for npu_chunk_kda_fwd_intra_sub_chunk.

Shape: B=1, T=8192, H=HV=32, K=128, BT=64, bf16 (DESIGN / 1.5ms target).
"""

from __future__ import annotations

import math
import os
import statistics
import time

import torch


def main() -> None:
    import fla_npu.ops.ascendc  # noqa: F401 — OPP path before torch_npu
    import torch_npu  # noqa: F401
    from fla_npu.ops.ascendc import npu_chunk_kda_fwd_intra_sub_chunk

    # With ASCEND_RT_VISIBLE_DEVICES remapping, always use logical device 0 unless ASCEND_DEVICE_ID set.
    device = int(os.environ.get("ASCEND_DEVICE_ID", "0"))
    torch.npu.set_device(device)

    B, H, T, K, BT = 1, 32, 8192, 128, 64
    dtype = torch.bfloat16
    warmup = int(os.environ.get("FLA_NPU_WARMUP", "5"))
    iters = int(os.environ.get("FLA_NPU_ITERS", "20"))
    torch.manual_seed(0)

    q = torch.randn(B, H, T, K, dtype=dtype, device="npu")
    k = torch.randn(B, H, T, K, dtype=dtype, device="npu")
    g = (-torch.linspace(0, 8, T).view(1, 1, T, 1).expand(B, H, T, K)).to(dtype=dtype, device="npu").contiguous()
    beta = torch.rand(B, H, T, dtype=dtype, device="npu")
    scale = 1.0 / math.sqrt(K)

    for _ in range(warmup):
        npu_chunk_kda_fwd_intra_sub_chunk(q, k, g, beta, scale, BT)
        torch.npu.synchronize()

    times_ms = []
    for _ in range(iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        npu_chunk_kda_fwd_intra_sub_chunk(q, k, g, beta, scale, BT)
        torch.npu.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_ms.sort()
    med = statistics.median(times_ms)
    print(
        f"[prof-model] B={B} H={H} T={T} K={K} BT={BT} dtype={dtype} "
        f"wall_ms med={med:.3f} min={times_ms[0]:.3f} max={times_ms[-1]:.3f} "
        f"warmup={warmup} iters={iters} device={device}"
    )


if __name__ == "__main__":
    main()
