#!/usr/bin/env python3
import argparse
import gc
import importlib
import math
import os
import random
import time

import torch
import torch_npu  # noqa: F401

from fla_npu.ops import ascendc as fla_ascendc


os.environ["TBE_PARALLEL_COMPILE_ENABLE"] = "0"
os.environ["PARALLEL_COMPILE"] = "0"

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)


def release_aclnn_keepalive():
    try:
        runtime_mod = importlib.import_module("fla_npu.ops.ascendc._runtime")
        runtime_mod._RECENT_LAUNCH_STORAGE.clear()
    except Exception:
        pass


def random_cu_seqlens(total: int, seq_num: int, seed: int) -> list[int]:
    if seq_num <= 1:
        return [0, total]
    rng = random.Random(seed)
    cuts = sorted(rng.sample(range(1, total), seq_num - 1))
    return [0, *cuts, total]


def make_chunk_indices(cu_seqlens: list[int], chunk_size: int) -> list[int]:
    chunk_indices: list[int] = []
    for seq_idx in range(len(cu_seqlens) - 1):
        seq_len = cu_seqlens[seq_idx + 1] - cu_seqlens[seq_idx]
        chunk_num = (seq_len + chunk_size - 1) // chunk_size
        for chunk_idx in range(chunk_num):
            chunk_indices.extend((seq_idx, chunk_idx))
    return chunk_indices


def npu_empty(shape: tuple[int, ...], dtype: torch.dtype, device: int) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=f"npu:{device}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=["g", "gK", "both"], default="both")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--device", type=int, default=int(os.environ.get("TEST_DEVICE_ID", "0")))
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument("--HK", type=int, default=None)
    parser.add_argument("--HV", type=int, default=None)
    parser.add_argument("--T", type=int, default=65536)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--V", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--seq-num", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--has-dh0", action="store_true")
    args = parser.parse_args()
    hk = args.H if args.HK is None else args.HK
    hv = args.H if args.HV is None else args.HV
    if hv % hk != 0:
        raise ValueError(f"HV must be divisible by HK, got HK={hk}, HV={hv}")
    if args.repeat < 1:
        raise ValueError(f"repeat must be positive, got {args.repeat}")

    torch.npu.set_device(args.device)
    cu_seqlens = random_cu_seqlens(args.T, args.seq_num, args.seed)
    chunk_indices = make_chunk_indices(cu_seqlens, args.chunk_size)
    seq_lens = [cu_seqlens[i + 1] - cu_seqlens[i] for i in range(len(cu_seqlens) - 1)]
    branches = ["g", "gK"] if args.branch == "both" else [args.branch]

    print(
        "DHU PERF CONFIG "
        f"branches={','.join(branches)} repeat={args.repeat} B={args.B} HK={hk} HV={hv} T={args.T} "
        f"K={args.K} V={args.V} chunk_size={args.chunk_size} dtype=bf16 "
        f"gate_dtype=fp32 seq_num={args.seq_num} seed={args.seed} "
        f"seq_min={min(seq_lens)} seq_max={max(seq_lens)} chunk_pairs={len(chunk_indices) // 2}",
        flush=True,
    )

    with torch.no_grad():
        for branch in branches:
            print(f"DHU PERF CASE branch={branch}", flush=True)
            shape_k = (args.B, hk, args.T, args.K)
            shape_w = (args.B, hv, args.T, args.K)
            shape_v = (args.B, hv, args.T, args.V)
            q = npu_empty(shape_k, torch.bfloat16, args.device)
            k = npu_empty(shape_k, torch.bfloat16, args.device)
            w = npu_empty(shape_w, torch.bfloat16, args.device)
            d_o = npu_empty(shape_v, torch.bfloat16, args.device)
            dv = npu_empty(shape_v, torch.bfloat16, args.device)
            g = npu_empty((args.B, hv, args.T), torch.float32, args.device) if branch == "g" else None
            gk = npu_empty((args.B, hv, args.T, args.K), torch.float32, args.device) if branch == "gK" else None
            chunk_count = len(chunk_indices) // 2
            h0 = npu_empty((args.B, hv, chunk_count, args.K, args.V), torch.bfloat16, args.device) \
                if args.has_dh0 else None
            torch.npu.synchronize()

            for repeat_idx in range(args.repeat):
                run_label = f"branch={branch} repeat={repeat_idx + 1}/{args.repeat}"
                print(f"DHU PERF phase op_launch_start {run_label}", flush=True)
                t0 = time.perf_counter()
                outputs = fla_ascendc.npu_chunk_gated_delta_rule_bwd_dhu(
                    q,
                    k,
                    w,
                    d_o,
                    dv,
                    scale=1.0 / math.sqrt(float(args.K)),
                    chunk_size=args.chunk_size,
                    g=g,
                    gK=gk,
                    h0=h0,
                    dht=None,
                    cu_seqlens=cu_seqlens,
                    chunk_indices=chunk_indices,
                )
                torch.npu.synchronize()
                print(
                    f"DHU PERF phase_time python_sync={time.perf_counter() - t0:.6f}s {run_label}",
                    flush=True,
                )
                print(
                    "DHU PERF outputs="
                    f"{[None if output is None else tuple(output.shape) for output in outputs]} {run_label}",
                    flush=True,
                )
                del outputs
                release_aclnn_keepalive()
                gc.collect()

            del q, k, w, d_o, dv, g, gk, h0
            release_aclnn_keepalive()
            gc.collect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
