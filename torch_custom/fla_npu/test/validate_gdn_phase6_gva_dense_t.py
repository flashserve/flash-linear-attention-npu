#!/usr/bin/env python3
"""Small A2 smoke for native GVA and non-aligned dense T in Phase6."""

from __future__ import annotations

import argparse
import json

import torch
import torch_npu  # noqa: F401  # registers the NPU backend

from fla_npu.ops import ascendc


def run_case(device: int, key_heads: int, value_heads: int, tokens: int, chunk_size: int) -> dict:
    if value_heads % key_heads:
        raise ValueError("value_heads must be divisible by key_heads")
    torch.manual_seed(20260814 + tokens + key_heads * 17 + value_heads)
    torch.npu.set_device(device)
    dtype = torch.bfloat16
    q = (torch.randn(1, key_heads, tokens, 128, dtype=dtype) * 0.05).npu()
    k = (torch.randn(1, key_heads, tokens, 128, dtype=dtype) * 0.05).npu()
    v = (torch.randn(1, value_heads, tokens, 128, dtype=dtype) * 0.05).npu()
    beta = torch.sigmoid(torch.randn(1, tokens, value_heads, dtype=torch.float32, device="npu"))
    g = -torch.rand(1, tokens, value_heads, dtype=torch.float32, device="npu") * 0.1

    native = ascendc.gdn_core_fwd_phase6(
        q,
        k,
        v,
        g,
        beta,
        output_final_state=True,
        chunk_size=chunk_size,
        scale=128**-0.5,
    )
    expanded_q = q.repeat_interleave(value_heads // key_heads, dim=1).contiguous()
    expanded_k = k.repeat_interleave(value_heads // key_heads, dim=1).contiguous()
    expanded = ascendc.gdn_core_fwd_phase6(
        expanded_q,
        expanded_k,
        v,
        g,
        beta,
        output_final_state=True,
        chunk_size=chunk_size,
        scale=128**-0.5,
    )
    torch.npu.synchronize()

    names = ("output", "final_state", "g_cumsum", "A")
    comparisons = {}
    finite = {}
    for name, left, right in zip(names, native, expanded):
        if left is None or right is None:
            comparisons[name] = left is right
            finite[name] = True
            continue
        comparisons[name] = bool(torch.equal(left, right))
        finite[name] = bool(torch.isfinite(left.float()).all().item())
    return {
        "key_heads": key_heads,
        "value_heads": value_heads,
        "tokens": tokens,
        "chunk_size": chunk_size,
        "native_shapes": [None if item is None else list(item.shape) for item in native],
        "expanded_shapes": [None if item is None else list(item.shape) for item in expanded],
        "native_vs_expanded_bit_exact": comparisons,
        "native_finite": finite,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--key-heads", type=int, default=2)
    parser.add_argument("--value-heads", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=130)
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()
    result = run_case(args.device, args.key_heads, args.value_heads, args.tokens, args.chunk_size)
    print(json.dumps(result, sort_keys=True))
    if not all(result["native_vs_expanded_bit_exact"].values()):
        raise SystemExit("native GVA differs from expanded-H reference")
    if not all(result["native_finite"].values()):
        raise SystemExit("native GVA produced non-finite output")


if __name__ == "__main__":
    main()
