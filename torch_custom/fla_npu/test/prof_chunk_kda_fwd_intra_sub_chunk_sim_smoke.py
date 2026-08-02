#!/usr/bin/env python3
"""Tiny shape for msprof op simulator (full 8192 is too slow under sim)."""
from __future__ import annotations

import math
import os

import fla_npu.ops.ascendc as ascendc_ops  # noqa: F401 — before torch_npu
import torch
import torch_npu  # noqa: F401


def main() -> None:
    device = int(os.environ.get("ASCEND_DEVICE_ID", "0"))
    torch.npu.set_device(device)
    B, H, T, K, BT = 1, 2, 64, 128, 64
    dtype = torch.bfloat16
    torch.manual_seed(0)
    q = torch.randn(B, H, T, K, dtype=dtype, device="npu")
    k = torch.randn(B, H, T, K, dtype=dtype, device="npu")
    g = (-torch.linspace(0, 8, T, device="npu").view(1, 1, T, 1).expand(B, H, T, K)).to(dtype).contiguous()
    beta = torch.rand(B, H, T, dtype=dtype, device="npu")
    scale = 1.0 / math.sqrt(K)
    for _ in range(2):
        aqk, akkd = ascendc_ops.npu_chunk_kda_fwd_intra_sub_chunk(q, k, g, beta, scale, BT)
        torch.npu.synchronize()
    print(f"[sim-smoke] aqk={tuple(aqk.shape)} akkd={tuple(akkd.shape)}")


if __name__ == "__main__":
    main()
