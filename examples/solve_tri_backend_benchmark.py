#!/usr/bin/env python3
"""Compare AscendC with fla-org's Triton-Ascend solve on GDR KKT input.

Set ``FLA_ORG_ROOT`` to the fla-org/flash-linear-attention source checkout.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from fla_npu.ops.ascendc import solve_tri as ascendc_solve_tri
from fla_npu.ops.triton import (
    chunk_local_cumsum,
    chunk_scaled_dot_kkt_fwd,
)

from flash_gated_delta_rule_100layer_stress import (
    DEFAULT_CU_SEQLENS,
    prepare_varlen_metadata,
)

_FLA_ORG_SOLVE_TRIL = None


def get_fla_org_solve_tril():
    """Load and verify fla-org's Triton-Ascend solve implementation."""

    global _FLA_ORG_SOLVE_TRIL
    if _FLA_ORG_SOLVE_TRIL is not None:
        return _FLA_ORG_SOLVE_TRIL

    root_value = os.environ.get("FLA_ORG_ROOT")
    if not root_value:
        raise RuntimeError(
            "FLA_ORG_ROOT must point to a fla-org/flash-linear-attention "
            "source checkout"
        )
    root = Path(root_value).expanduser().resolve()
    marker = (
        root
        / "fla"
        / "ops"
        / "utils"
        / "backends"
        / "triton_ascend"
        / "solve_tril.py"
    )
    if not marker.is_file():
        raise RuntimeError(
            "FLA_ORG_ROOT does not contain fla-org's Triton-Ascend "
            f"solve backend: {marker}"
        )

    loaded_fla = sys.modules.get("fla")
    if loaded_fla is not None:
        origin_value = getattr(loaded_fla, "__file__", "")
        origin = Path(origin_value).resolve() if origin_value else None
        if origin is None or not origin.is_relative_to(root):
            for module_name in tuple(sys.modules):
                if module_name == "fla" or module_name.startswith("fla."):
                    del sys.modules[module_name]

    root_text = str(root)
    while root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    module = importlib.import_module(
        "fla.ops.utils.backends.triton_ascend.solve_tril"
    )
    module_path = Path(module.__file__).resolve()
    if not module_path.is_relative_to(root):
        raise RuntimeError(
            "resolved solve_tri backend is not from FLA_ORG_ROOT: "
            f"{module_path}"
        )
    _FLA_ORG_SOLVE_TRIL = module.solve_tril_npu
    return _FLA_ORG_SOLVE_TRIL


def _build_input(
    *,
    tokens: int,
    heads: int,
    chunk_size: int,
    dtype: torch.dtype,
    a_dtype: torch.dtype,
    input_source: str,
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
    if input_source == "zeros":
        A = torch.zeros(
            1,
            tokens,
            heads,
            chunk_size,
            dtype=a_dtype,
            device=device,
        )
        return A, cu_seqlens, offsets, chunk_indices, chunk_indices_list

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
        output_dtype=a_dtype,
    )
    return A, cu_seqlens, offsets, chunk_indices, chunk_indices_list


def _run_fla_org(
    A: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    chunk_indices: dict[str, torch.Tensor],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    solve_tril = get_fla_org_solve_tril()
    return solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices[str(A.shape[-1])],
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
        choices=("fla-org", "ascendc", "compare"),
        default="compare",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--a-dtype",
        choices=("fp32", "bf16"),
        default="fp32",
        help=(
            "KKT output dtype passed to solve; fp32 matches the real GDR "
            "adapter"
        ),
    )
    parser.add_argument(
        "--input-source",
        choices=("kkt", "zeros"),
        default="kkt",
        help=(
            "kkt uses the real GDR producer; zeros isolates solve performance "
            "from producer/cast stream dependencies"
        ),
    )
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    if args.backend in ("fla-org", "compare"):
        # Import fla-org before the first local Triton kernel initializes the
        # runtime. Loading it after KKT has launched can leave its autotuner
        # observing a partially initialized backend.
        get_fla_org_solve_tril()

    device = torch.device(f"npu:{args.device}")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    a_dtype = (
        torch.float32 if args.a_dtype == "fp32" else torch.bfloat16
    )
    torch.npu.set_device(device)
    torch.manual_seed(20260729)
    torch.npu.manual_seed_all(20260729)
    A, cu_seqlens, offsets, chunk_indices, chunk_indices_list = (
        _build_input(
            tokens=args.tokens,
            heads=args.heads,
            chunk_size=args.chunk_size,
            dtype=dtype,
            a_dtype=a_dtype,
            input_source=args.input_source,
            device=device,
        )
    )

    def run(backend: str) -> torch.Tensor:
        if backend == "fla-org":
            return _run_fla_org(
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
        fla_org_output = run("fla-org")
        ascendc_output = run("ascendc")
        torch.npu.synchronize()
        difference = fla_org_output.float() - ascendc_output.float()
        denominator = torch.linalg.vector_norm(ascendc_output.float()).clamp_min(
            1e-12
        )
        result = {
            "a_dtype": str(A.dtype),
            "input_source": args.input_source,
            "output_dtype": str(dtype),
            "fla_org_finite": bool(
                torch.isfinite(fla_org_output).all().item()
            ),
            "ascendc_finite": bool(torch.isfinite(ascendc_output).all().item()),
            "max_abs_diff": float(difference.abs().max().item()),
            "relative_l2": float(
                (torch.linalg.vector_norm(difference) / denominator).item()
            ),
        }
        print(json.dumps(result, sort_keys=True))
        return (
            0
            if result["fla_org_finite"] and result["ascendc_finite"]
            else 1
        )

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
                "a_dtype": str(A.dtype),
                "input_source": args.input_source,
                "output_dtype": str(output.dtype),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
