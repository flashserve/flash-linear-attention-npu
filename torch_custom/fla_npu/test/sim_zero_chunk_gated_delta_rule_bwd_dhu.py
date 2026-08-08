#!/usr/bin/env python3
import argparse
import gc
import importlib
import math
import os

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


def make_chunk_indices(total: int, chunk_size: int) -> list[int]:
    return [item for chunk_idx in range((total + chunk_size - 1) // chunk_size) for item in (0, chunk_idx)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=["g", "gK"], default="g")
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--T", type=int, default=128)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--V", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()

    with torch.no_grad():
        shape_k = (args.B, args.H, args.T, args.K)
        shape_v = (args.B, args.H, args.T, args.V)
        q = torch.zeros(shape_k, dtype=torch.bfloat16)
        k = torch.zeros(shape_k, dtype=torch.bfloat16)
        w = torch.zeros(shape_k, dtype=torch.bfloat16)
        d_o = torch.zeros(shape_v, dtype=torch.bfloat16)
        dv = torch.zeros(shape_v, dtype=torch.bfloat16)
        g = torch.zeros((args.B, args.H, args.T), dtype=torch.float32)
        gk = torch.zeros(shape_k, dtype=torch.float32)
        cu_seqlens = [0, args.T]
        chunk_indices = make_chunk_indices(args.T, args.chunk_size)

        outputs = fla_ascendc.npu_chunk_gated_delta_rule_bwd_dhu(
            q.npu(),
            k.npu(),
            w.npu(),
            d_o.npu(),
            dv.npu(),
            scale=1.0 / math.sqrt(float(args.K)),
            chunk_size=args.chunk_size,
            g=g.npu() if args.branch == "g" else None,
            gK=gk.npu() if args.branch == "gK" else None,
            h0=None,
            dht=None,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

        print(
            "SIM ZERO ChunkGatedDeltaRuleBwdDhu DONE "
            f"branch={args.branch} B={args.B} H={args.H} T={args.T} "
            f"K={args.K} V={args.V} chunk_size={args.chunk_size} "
            f"outputs={[None if out is None else tuple(out.shape) for out in outputs]}",
            flush=True,
        )

        del q, k, w, d_o, dv, g, gk, outputs
        release_aclnn_keepalive()
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
