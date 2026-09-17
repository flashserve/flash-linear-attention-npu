#!/usr/bin/env python3
"""Single-launch harness for the KernelA A5 instruction simulator."""

import argparse

import torch
import torch_npu  # noqa: F401

from fla_npu.ops.ascendc import npu_chunk_kda_bwd_prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=4)
    args = parser.parse_args()
    device = "npu:0"
    aqk = torch.empty((1, args.heads, 64, 64), dtype=torch.bfloat16, device=device)
    v_new = torch.empty((1, args.heads, 64, 128), dtype=torch.bfloat16, device=device)
    d_o = torch.empty((1, args.heads, 64, 128), dtype=torch.bfloat16, device=device)
    h = torch.empty((1, args.heads, 1, 128, 128), dtype=torch.bfloat16, device=device)
    outputs = npu_chunk_kda_bwd_prepare(
        aqk,
        v_new,
        d_o,
        h,
        scale=0.125,
        chunk_size=64,
        state_v_first=False,
    )
    torch.npu.synchronize()
    del outputs


if __name__ == "__main__":
    main()
