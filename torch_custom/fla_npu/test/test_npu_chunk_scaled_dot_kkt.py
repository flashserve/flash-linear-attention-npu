#!/usr/bin/env python3
"""Precision test for torch.ops.npu.npu_chunk_scaled_dot_kkt.

The custom op covers the gk=None fixed-length path and uses head-first layout:
  k    : [B, H, T, K]
  g    : [B, H, T]
  beta : [B, H, T]
  out  : [B, H, T, BT]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import torch


MAX_ABS_TOL = 5e-3
MEAN_ABS_TOL = 5e-4
ZERO_TOL = 1e-6


@dataclass(frozen=True)
class Case:
    B: int
    H: int
    T: int
    K: int
    BT: int
    dtype: torch.dtype = torch.float16


CASES = (
    Case(1, 1, 1, 8, 16),
    Case(1, 2, 17, 8, 16),
    Case(2, 3, 64, 16, 16),
    Case(1, 2, 96, 32, 32),
    Case(2, 4, 128, 64, 64),
    Case(1, 2, 160, 32, 128),
    Case(1, 2, 17, 8, 16, torch.bfloat16),
    Case(2, 3, 64, 16, 16, torch.bfloat16),
    Case(1, 2, 96, 32, 32, torch.bfloat16),
    Case(2, 4, 128, 64, 64, torch.bfloat16),
)


def chunk_scaled_dot_kkt_reference(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """CPU fp32 reference for the gk=None path."""

    if k.dim() != 4:
        raise ValueError(f"k must be [B,H,T,K], got {tuple(k.shape)}")
    if g.shape != k.shape[:3] or beta.shape != k.shape[:3]:
        raise ValueError(
            "g and beta must be [B,H,T], "
            f"got g={tuple(g.shape)}, beta={tuple(beta.shape)}, k={tuple(k.shape)}"
        )

    B, H, T, _ = k.shape
    out = torch.zeros((B, H, T, chunk_size), dtype=torch.float32)
    k_f = k.float()
    g_f = g.float()
    beta_f = beta.float()

    for b in range(B):
        for h in range(H):
            for start in range(0, T, chunk_size):
                end = min(start + chunk_size, T)
                valid = end - start
                k_block = k_f[b, h, start:end, :]
                score = k_block @ k_block.T
                gate = torch.exp(torch.clamp(g_f[b, h, start:end, None] - g_f[b, h, None, start:end], -50.0, 50.0))
                scaled = score * gate * beta_f[b, h, start:end, None]
                mask = torch.tril(torch.ones((valid, valid), dtype=torch.bool), diagonal=-1)
                out[b, h, start:end, :valid] = torch.where(mask, scaled, torch.zeros_like(scaled))

    return out


def make_inputs(case: Case, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    k = (torch.randn(case.B, case.H, case.T, case.K, dtype=torch.float32) * 0.2).to(case.dtype)
    gate_delta = torch.randn(case.B, case.H, case.T, dtype=torch.float32) * 0.02
    g = torch.empty_like(gate_delta)
    for start in range(0, case.T, case.BT):
        end = min(start + case.BT, case.T)
        g[:, :, start:end] = torch.cumsum(gate_delta[:, :, start:end], dim=2)
    beta = torch.sigmoid(torch.randn(case.B, case.H, case.T, dtype=torch.float32))
    return k, g.contiguous(), beta.contiguous()


def _require_npu_op():
    try:
        import torch_npu  # noqa: F401
        import fla_npu  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("NPU test requires torch_npu and fla_npu to be importable") from exc

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("NPU device is not available")
    if not hasattr(torch.ops.npu, "npu_chunk_scaled_dot_kkt"):
        raise RuntimeError(
            "torch.ops.npu.npu_chunk_scaled_dot_kkt is not registered. "
            "Regenerate and rebuild torch_custom/fla_npu after updating npu_custom.yaml."
        )

    torch.npu.utils.set_device(0)


def _check_zero_regions(out: torch.Tensor, case: Case) -> float:
    max_zero = 0.0
    for start in range(0, case.T, case.BT):
        end = min(start + case.BT, case.T)
        valid = end - start
        block = out[:, :, start:end, :]
        upper = torch.triu(block[:, :, :, :valid], diagonal=0)
        max_zero = max(max_zero, upper.abs().max().item())
        if valid < case.BT:
            max_zero = max(max_zero, block[:, :, :, valid:].abs().max().item())
    return max_zero


def run_case(case: Case, seed: int, cpu_only: bool) -> bool:
    print(f"Case B={case.B} H={case.H} T={case.T} K={case.K} BT={case.BT} dtype={case.dtype}")
    k, g, beta = make_inputs(case, seed)
    golden = chunk_scaled_dot_kkt_reference(k, g, beta, case.BT)

    if cpu_only:
        max_zero = _check_zero_regions(golden, case)
        passed = golden.shape == (case.B, case.H, case.T, case.BT) and max_zero <= ZERO_TOL
        print(f"  CPU golden shape={tuple(golden.shape)} max_zero={max_zero:.3e} passed={passed}")
        return passed

    out = torch.ops.npu.npu_chunk_scaled_dot_kkt(
        k.npu(),
        g.npu(),
        beta.npu(),
        chunk_size=case.BT,
    ).cpu()

    diff = (out.float() - golden).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    max_zero = _check_zero_regions(out.float(), case)
    shape_ok = tuple(out.shape) == (case.B, case.H, case.T, case.BT)
    dtype_ok = out.dtype == torch.float32
    passed = shape_ok and dtype_ok and max_abs <= MAX_ABS_TOL and mean_abs <= MEAN_ABS_TOL and max_zero <= ZERO_TOL
    print(
        "  "
        f"shape={tuple(out.shape)} dtype={out.dtype} "
        f"max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} max_zero={max_zero:.3e} "
        f"passed={passed}"
    )
    return passed


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
