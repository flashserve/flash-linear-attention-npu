#!/usr/bin/env python3
"""Medium shape for msprof op simulator (T=1024; full 8192 too slow under sim).

Default: B=1, H=2, T=1024, K=128, BT=64 → NT=16, tasks=32 MixBlocks-ish.
Override via env: FLA_SIM_B/H/T/K/BT, FLA_SIM_WARMUP (default 1).
"""
from __future__ import annotations

import math
import os

import fla_npu.ops.ascendc as ascendc_ops  # noqa: F401 — before torch_npu
import torch
import torch_npu  # noqa: F401


def main() -> None:
    device = int(os.environ.get("ASCEND_DEVICE_ID", "0"))
    torch.npu.set_device(device)
    B = int(os.environ.get("FLA_SIM_B", "1"))
    H = int(os.environ.get("FLA_SIM_H", "2"))
    T = int(os.environ.get("FLA_SIM_T", "1024"))
    K = int(os.environ.get("FLA_SIM_K", "128"))
    BT = int(os.environ.get("FLA_SIM_BT", "64"))
    warmup = int(os.environ.get("FLA_SIM_WARMUP", "1"))
    dtype = torch.bfloat16
    torch.manual_seed(0)
    q = torch.randn(B, H, T, K, dtype=dtype, device="npu")
    k = torch.randn(B, H, T, K, dtype=dtype, device="npu")
    g = (-torch.linspace(0, 8, T, device="npu").view(1, 1, T, 1).expand(B, H, T, K)).to(dtype).contiguous()
    beta = torch.rand(B, H, T, dtype=dtype, device="npu")
    scale = 1.0 / math.sqrt(K)
    for _ in range(warmup):
        aqk, akkd = ascendc_ops.npu_chunk_kda_fwd_intra_sub_chunk(q, k, g, beta, scale, BT)
        torch.npu.synchronize()
    aqk, akkd = ascendc_ops.npu_chunk_kda_fwd_intra_sub_chunk(q, k, g, beta, scale, BT)
    torch.npu.synchronize()
    nt = T // BT
    print(
        f"[sim-t1024] shape=({B},{H},{T},{K}) BT={BT} NT={nt} tasks~={B*H*nt} "
        f"aqk={tuple(aqk.shape)} akkd={tuple(akkd.shape)}"
    )


if __name__ == "__main__":
    main()
