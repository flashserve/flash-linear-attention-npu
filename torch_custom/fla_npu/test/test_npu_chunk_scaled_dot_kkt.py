#!/usr/bin/env python3
"""Precision test for fla_npu.ops.ascendc.chunk_scaled_dot_kkt.

The custom op covers the gk=None fixed-length path and uses head-first layout:
  k    : [B, Hk, T, K]
  g    : [B, Hv, T]
  beta : [B, Hv, T]
  out  : [B, Hk, T, BT]

For this KKT op, GVA inputs may provide g/beta with Hv heads while A remains
key-head aligned. The dumped GPU path uses the first Hk g/beta heads.

Fixed-length mode omits cu_seqlens/chunk_indices. Varlen mode passes flat
chunk_indices as [seq0, chunk0, seq1, chunk1, ...].
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import torch

from fla_npu.ops import ascendc as ascendc_ops


MAX_ABS_TOL = 5e-3
MEAN_ABS_TOL = 5e-4
ZERO_TOL = 1e-6


@dataclass(frozen=True)
class Case:
    B: int
    Hk: int
    Hv: int
    T: int
    K: int
    BT: int
    dtype: torch.dtype = torch.float16


CASES = (
    Case(1, 1, 1, 1, 8, 16),
    Case(1, 2, 2, 17, 8, 16),
    Case(2, 3, 3, 64, 16, 16),
    Case(1, 2, 2, 96, 32, 32),
    Case(2, 4, 4, 128, 64, 64),
    Case(1, 2, 2, 160, 32, 128),
    Case(1, 2, 4, 33, 16, 16),
    Case(2, 3, 6, 64, 16, 16),
    Case(1, 2, 2, 17, 8, 16, torch.bfloat16),
    Case(2, 3, 3, 64, 16, 16, torch.bfloat16),
    Case(1, 2, 2, 96, 32, 32, torch.bfloat16),
    Case(2, 4, 4, 128, 64, 64, torch.bfloat16),
    Case(1, 2, 4, 96, 32, 32, torch.bfloat16),
)


def chunk_scaled_dot_kkt_reference(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
    cu_seqlens: list[int] | None = None,
    chunk_indices: list[int] | None = None,
) -> torch.Tensor:
    """CPU fp32 reference for the gk=None fixed and varlen paths."""

    if k.dim() != 4:
        raise ValueError(f"k must be [B,Hk,T,K], got {tuple(k.shape)}")
    if g.dim() != 3 or beta.dim() != 3:
        raise ValueError(f"g and beta must be [B,Hv,T], got g={tuple(g.shape)}, beta={tuple(beta.shape)}")

    B, Hk, T, _ = k.shape
    Hv = g.shape[1]
    if g.shape[0] != B or g.shape[2] != T or beta.shape != g.shape or Hv % Hk != 0:
        raise ValueError(
            "GVA shapes must satisfy k=[B,Hk,T,K], g/beta=[B,Hv,T], and Hv % Hk == 0, "
            f"got g={tuple(g.shape)}, beta={tuple(beta.shape)}, k={tuple(k.shape)}"
        )

    out = torch.zeros((B, Hk, T, chunk_size), dtype=torch.float32)
    k_f = k.float()
    g_f = g.float()
    beta_f = beta.float()

    for b in range(B):
        for h in range(Hk):
            for start, end in iter_chunk_ranges(T, chunk_size, cu_seqlens, chunk_indices):
                valid = end - start
                k_block = k_f[b, h, start:end, :]
                score = k_block @ k_block.T
                gate = torch.exp(torch.clamp(g_f[b, h, start:end, None] - g_f[b, h, None, start:end], -50.0, 50.0))
                scaled = score * gate * beta_f[b, h, start:end, None]
                mask = torch.tril(torch.ones((valid, valid), dtype=torch.bool), diagonal=-1)
                out[b, h, start:end, :valid] = torch.where(mask, scaled, torch.zeros_like(scaled))

    return out


def iter_chunk_ranges(
    total_t: int,
    chunk_size: int,
    cu_seqlens: list[int] | None = None,
    chunk_indices: list[int] | None = None,
) -> Iterable[tuple[int, int]]:
    if cu_seqlens is None:
        for start in range(0, total_t, chunk_size):
            yield start, min(start + chunk_size, total_t)
        return

    if chunk_indices is None or len(chunk_indices) % 2 != 0:
        raise ValueError("chunk_indices must be a flat even-length list")
    for idx in range(0, len(chunk_indices), 2):
        seq_idx = int(chunk_indices[idx])
        local_chunk = int(chunk_indices[idx + 1])
        bos = int(cu_seqlens[seq_idx])
        eos = int(cu_seqlens[seq_idx + 1])
        start = bos + local_chunk * chunk_size
        end = min(start + chunk_size, eos)
        if start < end:
            yield start, end


def make_varlen_metadata(total_t: int, chunk_size: int) -> tuple[list[int], list[int]]:
    if total_t <= chunk_size:
        cu_seqlens = [0, total_t]
    else:
        split = min(total_t - 1, max(1, total_t // 2 + 1))
        cu_seqlens = [0, split, total_t]

    chunk_indices: list[int] = []
    for seq_idx, (bos, eos) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        seq_len = eos - bos
        for local_chunk in range((seq_len + chunk_size - 1) // chunk_size):
            chunk_indices.extend([seq_idx, local_chunk])
    return cu_seqlens, chunk_indices


def make_inputs(case: Case, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    k = (torch.randn(case.B, case.Hk, case.T, case.K, dtype=torch.float32) * 0.2).to(case.dtype)
    gate_delta = torch.randn(case.B, case.Hv, case.T, dtype=torch.float32) * 0.02
    g = torch.empty_like(gate_delta)
    for start in range(0, case.T, case.BT):
        end = min(start + case.BT, case.T)
        g[:, :, start:end] = torch.cumsum(gate_delta[:, :, start:end], dim=2)
    beta = torch.sigmoid(torch.randn(case.B, case.Hv, case.T, dtype=torch.float32))
    return k, g.contiguous(), beta.contiguous()


def _require_npu_op():
    try:
        import torch_npu  # noqa: F401
        from fla_npu.ops import ascendc as _ascendc_ops  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("NPU test requires torch_npu and fla_npu to be importable") from exc

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("NPU device is not available")
    if not hasattr(ascendc_ops, "chunk_scaled_dot_kkt"):
        raise RuntimeError(
            "fla_npu.ops.ascendc.chunk_scaled_dot_kkt is not exported. "
            "Rebuild and reinstall the integrated fla_npu wheel."
        )

    torch.npu.utils.set_device(0)


def _check_zero_regions(
    out: torch.Tensor,
    case: Case,
    cu_seqlens: list[int] | None = None,
    chunk_indices: list[int] | None = None,
) -> float:
    max_zero = 0.0
    for start, end in iter_chunk_ranges(case.T, case.BT, cu_seqlens, chunk_indices):
        valid = end - start
        block = out[:, :, start:end, :]
        upper = torch.triu(block[:, :, :, :valid], diagonal=0)
        max_zero = max(max_zero, upper.abs().max().item())
        if valid < case.BT:
            max_zero = max(max_zero, block[:, :, :, valid:].abs().max().item())
    return max_zero


def run_case(case: Case, seed: int, cpu_only: bool) -> bool:
    print(
        f"Case B={case.B} Hk={case.Hk} Hv={case.Hv} T={case.T} "
        f"K={case.K} BT={case.BT} dtype={case.dtype}"
    )
    k, g, beta = make_inputs(case, seed)
    golden = chunk_scaled_dot_kkt_reference(k, g, beta, case.BT)

    if cpu_only:
        max_zero = _check_zero_regions(golden, case)
        passed = golden.shape == (case.B, case.Hk, case.T, case.BT) and max_zero <= ZERO_TOL
        if case.B != 1:
            print(f"  CPU golden shape={tuple(golden.shape)} max_zero={max_zero:.3e} passed={passed}")
            return passed
        cu_seqlens, chunk_indices = make_varlen_metadata(case.T, case.BT)
        varlen_golden = chunk_scaled_dot_kkt_reference(k, g, beta, case.BT, cu_seqlens, chunk_indices)
        varlen_max_zero = _check_zero_regions(varlen_golden, case, cu_seqlens, chunk_indices)
        passed = (
            passed
            and varlen_golden.shape == (case.B, case.Hk, case.T, case.BT)
            and varlen_max_zero <= ZERO_TOL
        )
        print(
            f"  CPU golden shape={tuple(golden.shape)} max_zero={max_zero:.3e} "
            f"varlen_chunks={len(chunk_indices) // 2} varlen_max_zero={varlen_max_zero:.3e} "
            f"passed={passed}"
        )
        return passed

    out = ascendc_ops.chunk_scaled_dot_kkt(
        k.npu(),
        g.npu(),
        beta.npu(),
        chunk_size=case.BT,
    ).cpu()

    diff = (out.float() - golden).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    max_zero = _check_zero_regions(out.float(), case)
    shape_ok = tuple(out.shape) == (case.B, case.Hk, case.T, case.BT)
    dtype_ok = out.dtype == torch.float32
    passed = shape_ok and dtype_ok and max_abs <= MAX_ABS_TOL and mean_abs <= MEAN_ABS_TOL and max_zero <= ZERO_TOL
    print(
        "  "
        f"fixed shape={tuple(out.shape)} dtype={out.dtype} "
        f"max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} max_zero={max_zero:.3e} "
        f"passed={passed}"
    )

    # Varlen uses a flattened physical batch and therefore requires B=1.
    # B>1 cases still cover the fixed-length batch dimension above.
    if case.B != 1:
        return passed

    cu_seqlens, chunk_indices = make_varlen_metadata(case.T, case.BT)
    varlen_golden = chunk_scaled_dot_kkt_reference(k, g, beta, case.BT, cu_seqlens, chunk_indices)
    varlen_out = ascendc_ops.chunk_scaled_dot_kkt(
        k.npu(),
        g.npu(),
        beta.npu(),
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=case.BT,
    ).cpu()
    varlen_diff = (varlen_out.float() - varlen_golden).abs()
    varlen_max_abs = varlen_diff.max().item()
    varlen_mean_abs = varlen_diff.mean().item()
    varlen_max_zero = _check_zero_regions(varlen_out.float(), case, cu_seqlens, chunk_indices)
    varlen_passed = (
        tuple(varlen_out.shape) == (case.B, case.Hk, case.T, case.BT)
        and varlen_out.dtype == torch.float32
        and varlen_max_abs <= MAX_ABS_TOL
        and varlen_mean_abs <= MEAN_ABS_TOL
        and varlen_max_zero <= ZERO_TOL
    )
    print(
        "  "
        f"varlen cu={cu_seqlens} chunks={len(chunk_indices) // 2} "
        f"max_abs={varlen_max_abs:.6e} mean_abs={varlen_mean_abs:.6e} "
        f"max_zero={varlen_max_zero:.3e} passed={varlen_passed}"
    )
    return passed and varlen_passed


def iter_cases(limit: int | None) -> Iterable[Case]:
    return CASES if limit is None else CASES[:limit]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true", help="Only run CPU golden checks, without calling NPU op.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N cases.")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)

    if not args.cpu_only:
        _require_npu_op()

    results = [run_case(case, args.seed + idx, args.cpu_only) for idx, case in enumerate(iter_cases(args.limit))]
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
