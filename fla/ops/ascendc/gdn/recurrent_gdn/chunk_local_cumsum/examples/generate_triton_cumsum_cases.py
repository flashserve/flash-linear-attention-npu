#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401
import triton


def find_repo_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / ".git").exists():
            return path
    return Path(__file__).resolve().parents[7]


REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(REPO_ROOT))

try:
    from fla.ops.triton.triton_core.cumsum import chunk_local_cumsum  # noqa: E402
except ModuleNotFoundError:
    from triton_ops.triton_core.cumsum import chunk_local_cumsum  # noqa: E402


def compute_block_t(h, chunk_size):
    return triton.next_power_of_2((1 << 17) // (h * chunk_size))


def build_chunk_indices(cu_seqlens, block_t):
    indices = []
    for seq_id in range(len(cu_seqlens) - 1):
        seq_len = cu_seqlens[seq_id + 1] - cu_seqlens[seq_id]
        for block_id in range(math.ceil(seq_len / block_t)):
            indices.append([seq_id, block_id])
    return torch.tensor(indices, dtype=torch.long, device="npu")


def reference_impl(g, chunk_size, reverse=False, scale=None, cu_seqlens=None, chunk_indices=None, block_t=None):
    out = torch.zeros_like(g, dtype=torch.float32)
    bsz, t_total, heads = g.shape
    if cu_seqlens is None:
        for b_idx in range(bsz):
            for t_idx in range(t_total):
                chunk_start = (t_idx // chunk_size) * chunk_size
                chunk_end = min(chunk_start + chunk_size, t_total)
                if reverse:
                    out[b_idx, t_idx, :] = g[b_idx, t_idx:chunk_end, :].sum(dim=0)
                else:
                    out[b_idx, t_idx, :] = g[b_idx, chunk_start:t_idx + 1, :].sum(dim=0)
    else:
        assert bsz == 1
        assert block_t is not None
        cu_cpu = cu_seqlens.cpu().tolist()
        chunk_cpu = chunk_indices.cpu().tolist()
        for seq_id, block_id in chunk_cpu:
            bos = cu_cpu[seq_id]
            eos = cu_cpu[seq_id + 1]
            seq_len = eos - bos
            t_start = block_id * block_t
            t_end = min(t_start + block_t, seq_len)
            for local_t in range(t_start, t_end):
                chunk_start = (local_t // chunk_size) * chunk_size
                chunk_end = min(chunk_start + chunk_size, seq_len)
                if reverse:
                    out[0, bos + local_t, :] = g[0, bos + local_t:bos + chunk_end, :].sum(dim=0)
                else:
                    out[0, bos + local_t, :] = g[0, bos + chunk_start:bos + local_t + 1, :].sum(dim=0)
    if scale is not None:
        out = out * float(scale)
    return out


def save_tensor(path, tensor, dtype):
    array = tensor.detach().cpu().contiguous().numpy().astype(dtype, copy=False)
    array.tofile(path)


def generate_case(out_dir, spec):
    name = spec["name"]
    bsz = spec["B"]
    t_total = spec["T"]
    heads = spec["H"]
    chunk_size = spec["chunk_size"]
    reverse = spec.get("reverse", False)
    scale = spec.get("scale", None)
    cu_values = spec.get("cu_seqlens")

    block_t = compute_block_t(heads, chunk_size)
    if block_t < chunk_size:
        raise RuntimeError(f"{name}: BLOCK_T={block_t} is smaller than chunk_size={chunk_size}")

    torch.manual_seed(spec["seed"])
    torch.npu.manual_seed(spec["seed"])
    g = torch.randn((bsz, t_total, heads), dtype=torch.float32, device="npu") * spec.get("input_scale", 0.25)

    cu_seqlens = None
    chunk_indices = None
    chunk_indices_out = None
    if cu_values is not None:
        cu_seqlens = torch.tensor(cu_values, dtype=torch.long, device="npu")
        chunk_indices = build_chunk_indices(cu_values, block_t)
        chunk_indices_out = {str(block_t): chunk_indices}

    y = chunk_local_cumsum(
        g,
        chunk_size=chunk_size,
        reverse=reverse,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices_out,
        output_dtype=torch.float32,
    )
    torch.npu.synchronize()

    ref = reference_impl(
        g.detach().cpu(),
        chunk_size=chunk_size,
        reverse=reverse,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        block_t=block_t,
    )
    triton_out = y.detach().cpu()
    max_ref_diff = float((triton_out - ref).abs().max().item())
    if not torch.allclose(triton_out, ref, atol=3e-5, rtol=3e-5):
        raise RuntimeError(f"{name}: Triton golden mismatch against PyTorch reference, max_diff={max_ref_diff}")

    g_file = f"{name}_g.bin"
    out_file = f"{name}_triton.bin"
    cu_file = "-"
    chunk_file = "-"
    save_tensor(out_dir / g_file, g, np.float32)
    save_tensor(out_dir / out_file, y, np.float32)
    if cu_seqlens is not None:
        cu_file = f"{name}_cu_seqlens.bin"
        chunk_file = f"{name}_chunk_indices.bin"
        save_tensor(out_dir / cu_file, cu_seqlens, np.int64)
        save_tensor(out_dir / chunk_file, chunk_indices, np.int64)

    return {
        "name": name,
        "B": bsz,
        "T": t_total,
        "H": heads,
        "chunk_size": chunk_size,
        "reverse": int(reverse),
        "scale": 1.0 if scale is None else float(scale),
        "is_varlen": int(cu_seqlens is not None),
        "block_t": int(block_t),
        "g_file": g_file,
        "golden_file": out_file,
        "cu_file": cu_file,
        "chunk_indices_file": chunk_file,
        "max_ref_diff": max_ref_diff,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate Triton golden cases for ChunkLocalCumsum.")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "testdata"))
    parser.add_argument(
        "--extra-fgd-case",
        action="store_true",
        help="Append a flash_gated_delta_rule-style performance case with g shape [64, 128, 512].",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        {"name": "fixed_prefix_c64", "B": 1, "T": 64, "H": 32, "chunk_size": 64, "reverse": False, "seed": 11},
        {"name": "fixed_reverse_c64", "B": 1, "T": 128, "H": 32, "chunk_size": 64, "reverse": True, "seed": 12},
        {"name": "fixed_scale_batch2", "B": 2, "T": 128, "H": 32, "chunk_size": 64, "reverse": False, "scale": 0.25, "seed": 13},
        {"name": "fixed_reverse_scale_c32", "B": 1, "T": 96, "H": 64, "chunk_size": 32, "reverse": True, "scale": -0.5, "seed": 14},
        {"name": "fixed_tiny_c16", "B": 1, "T": 17, "H": 32, "chunk_size": 16, "reverse": False, "seed": 16},
        {"name": "fixed_reverse_tail_c16", "B": 1, "T": 73, "H": 64, "chunk_size": 16, "reverse": True, "seed": 17},
        {
            "name": "fixed_scale_neg_c16",
            "B": 2,
            "T": 95,
            "H": 32,
            "chunk_size": 16,
            "reverse": False,
            "scale": -1.25,
            "seed": 18,
        },
        {"name": "fixed_batch3_c32", "B": 3, "T": 65, "H": 64, "chunk_size": 32, "reverse": False, "seed": 19},
        {"name": "fixed_reverse_batch2_c32", "B": 2, "T": 127, "H": 64, "chunk_size": 32, "reverse": True, "seed": 20},
        {"name": "fixed_h128_c32", "B": 1, "T": 96, "H": 128, "chunk_size": 32, "reverse": False, "seed": 21},
        {"name": "fixed_reverse_h128_c32", "B": 1, "T": 80, "H": 128, "chunk_size": 32, "reverse": True, "seed": 22},
        {"name": "fixed_h96_c16", "B": 1, "T": 129, "H": 96, "chunk_size": 16, "reverse": False, "seed": 23},
        {"name": "fixed_reverse_h96_c16", "B": 2, "T": 130, "H": 96, "chunk_size": 16, "reverse": True, "seed": 24},
        {"name": "fixed_h16_c64", "B": 1, "T": 129, "H": 16, "chunk_size": 64, "reverse": False, "seed": 25},
        {"name": "fixed_reverse_h16_c64", "B": 2, "T": 191, "H": 16, "chunk_size": 64, "reverse": True, "seed": 26},
        {"name": "fixed_h8_c64", "B": 3, "T": 257, "H": 8, "chunk_size": 64, "reverse": False, "seed": 27},
        {"name": "fixed_reverse_h8_c64", "B": 1, "T": 255, "H": 8, "chunk_size": 64, "reverse": True, "seed": 28},
        {
            "name": "fixed_scale_small_input",
            "B": 1,
            "T": 128,
            "H": 32,
            "chunk_size": 64,
            "reverse": False,
            "scale": 3.0,
            "input_scale": 0.03125,
            "seed": 29,
        },
        {
            "name": "fixed_reverse_scale_large_input",
            "B": 1,
            "T": 128,
            "H": 32,
            "chunk_size": 64,
            "reverse": True,
            "scale": -0.75,
            "input_scale": 0.5,
            "seed": 30,
        },
        {"name": "fixed_tail_c32_h32", "B": 2, "T": 97, "H": 32, "chunk_size": 32, "reverse": False, "seed": 31},
        {"name": "fixed_reverse_tail_c32_h32", "B": 1, "T": 193, "H": 32, "chunk_size": 32, "reverse": True, "seed": 32},
        {"name": "fixed_single_chunk_c64_b2", "B": 2, "T": 31, "H": 32, "chunk_size": 64, "reverse": False, "seed": 33},
        {"name": "fixed_reverse_single_chunk_c64", "B": 1, "T": 63, "H": 32, "chunk_size": 64, "reverse": True, "seed": 34},
        {"name": "fixed_exact_two_blocks_c64", "B": 2, "T": 128, "H": 32, "chunk_size": 64, "reverse": False, "seed": 35},
        {
            "name": "varlen_prefix_c64",
            "B": 1,
            "T": 128,
            "H": 32,
            "chunk_size": 64,
            "reverse": False,
            "cu_seqlens": [0, 50, 128],
            "seed": 36,
        },
        {
            "name": "varlen_reverse_scale",
            "B": 1,
            "T": 128,
            "H": 32,
            "chunk_size": 64,
            "reverse": True,
            "scale": 0.25,
            "cu_seqlens": [0, 50, 128],
            "seed": 15,
        },
        {
            "name": "varlen_tail_c32_h64",
            "B": 1,
            "T": 113,
            "H": 64,
            "chunk_size": 32,
            "reverse": False,
            "cu_seqlens": [0, 1, 49, 113],
            "seed": 37,
        },
        {
            "name": "varlen_reverse_tail_c32_h64",
            "B": 1,
            "T": 113,
            "H": 64,
            "chunk_size": 32,
            "reverse": True,
            "cu_seqlens": [0, 1, 49, 113],
            "seed": 38,
        },
        {
            "name": "varlen_c16_h96",
            "B": 1,
            "T": 201,
            "H": 96,
            "chunk_size": 16,
            "reverse": False,
            "scale": -0.25,
            "cu_seqlens": [0, 17, 99, 201],
            "seed": 39,
        },
        {
            "name": "varlen_reverse_c16_h96",
            "B": 1,
            "T": 201,
            "H": 96,
            "chunk_size": 16,
            "reverse": True,
            "scale": 0.5,
            "cu_seqlens": [0, 17, 99, 201],
            "seed": 40,
        },
    ]

    if args.extra_fgd_case:
        # flash_gated_delta_rule passes g as [B, T, H]. For H=512, chunk_size=64 would
        # make BLOCK_T=4 < chunk_size, so use the largest valid chunk size.
        specs.append(
            {
                "name": "fgd_g_64_128_512_c16",
                "B": 64,
                "T": 128,
                "H": 512,
                "chunk_size": 16,
                "reverse": False,
                "input_scale": 0.05,
                "seed": 41,
            }
        )

    cases = [generate_case(out_dir, spec) for spec in specs]

    manifest_path = out_dir / "cases.txt"
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("# name B T H chunk_size reverse scale is_varlen g_file golden_file cu_file chunk_indices_file\n")
        for case in cases:
            f.write(
                "{name} {B} {T} {H} {chunk_size} {reverse} {scale:.9g} {is_varlen} "
                "{g_file} {golden_file} {cu_file} {chunk_indices_file}\n".format(**case)
            )

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"golden": "triton", "cases": cases}, f, indent=2)

    print(f"Generated {len(cases)} Triton golden cases under {out_dir}")
    for case in cases:
        print(f"  {case['name']}: block_t={case['block_t']} max_ref_diff={case['max_ref_diff']:.6g}")


if __name__ == "__main__":
    main()
