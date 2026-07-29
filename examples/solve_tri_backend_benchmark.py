#!/usr/bin/env python3
"""Compare the AscendC and Triton-Ascend solve_tri paths on GDR KKT input."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from fla_npu.ops.ascendc import solve_tri as ascendc_solve_tri
from fla_npu.ops.triton import (
    chunk_local_cumsum,
    chunk_scaled_dot_kkt_fwd,
    solve_tril_npu,
)

from flash_gated_delta_rule_100layer_stress import (
    DEFAULT_CU_SEQLENS,
    prepare_varlen_metadata,
)


def _build_kkt_input(
    *,
    tokens: int,
    heads: int,
    chunk_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[int],
    dict[str, torch.Tensor],
    dict[str, list[int]],
]:
    if tokens != DEFAULT_CU_SEQLENS[-1]:
        offsets = [0, tokens // 4, tokens // 2, 3 * tokens // 4, tokens]
    else:
        offsets = DEFAULT_CU_SEQLENS
    cu_seqlens, chunk_indices, chunk_indices_list = prepare_varlen_metadata(
        offsets,
        chunk_size,
        device,
    )
    k = F.normalize(
        torch.randn(
            1,
            heads,
            tokens,
            128,
            dtype=torch.float32,
            device=device,
        ),
        dim=-1,
    ).to(dtype)
    g = F.logsigmoid(
        torch.randn(
            1,
            tokens,
            heads,
            dtype=torch.float32,
            device=device,
        )
    ).to(dtype)
    beta = torch.sigmoid(
        torch.randn(
            1,
            tokens,
            heads,
            dtype=torch.float32,
            device=device,
        )
    ).to(dtype)
    cumulative_g = chunk_local_cumsum(
        g,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices,
        head_first=False,
    )
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=cumulative_g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices[str(chunk_size)],
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )
    return A, cu_seqlens, offsets, chunk_indices, chunk_indices_list


def _run_triton(
    A: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    chunk_indices: dict[str, torch.Tensor],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    return solve_tril_npu(
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices,
        output_dtype=output_dtype,
    )


def _run_ascendc(
    A: torch.Tensor,
    *,
    offsets: list[int],
    chunk_indices_list: dict[str, list[int]],
    chunk_size: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    return ascendc_solve_tri(
        A.to(output_dtype).squeeze(0).contiguous(),
        cu_seqlens=offsets,
        chunk_indices=chunk_indices_list[str(chunk_size)],
        layout="tnd",
    ).unsqueeze(0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("triton", "ascendc", "compare"),
        default="compare",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    device = torch.device(f"npu:{args.device}")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.npu.set_device(device)
    torch.manual_seed(20260729)
    torch.npu.manual_seed_all(20260729)
    A, cu_seqlens, offsets, chunk_indices, chunk_indices_list = (
        _build_kkt_input(
            tokens=args.tokens,
            heads=args.heads,
            chunk_size=args.chunk_size,
            dtype=dtype,
            device=device,
        )
    )

    def run(backend: str) -> torch.Tensor:
        if backend == "triton":
            return _run_triton(
                A,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                output_dtype=dtype,
            )
        return _run_ascendc(
            A,
            offsets=offsets,
            chunk_indices_list=chunk_indices_list,
            chunk_size=args.chunk_size,
            output_dtype=dtype,
        )

    if args.backend == "compare":
        triton_output = run("triton")
        ascendc_output = run("ascendc")
        torch.npu.synchronize()
        difference = triton_output.float() - ascendc_output.float()
        denominator = torch.linalg.vector_norm(ascendc_output.float()).clamp_min(
            1e-12
        )
        result = {
            "triton_finite": bool(torch.isfinite(triton_output).all().item()),
            "ascendc_finite": bool(torch.isfinite(ascendc_output).all().item()),
            "max_abs_diff": float(difference.abs().max().item()),
            "relative_l2": float(
                (torch.linalg.vector_norm(difference) / denominator).item()
            ),
        }
        print(json.dumps(result, sort_keys=True))
        return 0 if result["triton_finite"] and result["ascendc_finite"] else 1

    output = run(args.backend)
    torch.npu.synchronize()
    for _ in range(args.repeats):
        output = run(args.backend)
    torch.npu.synchronize()
    print(
        json.dumps(
            {
                "backend": args.backend,
                "repeats": args.repeats,
                "finite": bool(torch.isfinite(output).all().item()),
                "shape": list(output.shape),
                "dtype": str(output.dtype),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
