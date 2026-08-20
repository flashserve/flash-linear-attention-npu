#!/usr/bin/env python3
"""Reference and NPU checks for the Phase 3 local_cumsum + KKT pilot."""

from __future__ import annotations

import argparse
import math
import os
import unittest
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Case:
    batch: int
    heads: int
    tokens: int
    chunk_size: int
    dtype: torch.dtype
    cu_seqlens: tuple[int, ...] | None = None


def canonical_chunk_indices(cu_seqlens: tuple[int, ...], chunk_size: int) -> tuple[int, ...]:
    indices = []
    for seq, (begin, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        for local_chunk in range(math.ceil((end - begin) / chunk_size)):
            indices.extend((seq, local_chunk))
    return tuple(indices)


def local_cumsum_block_t(chunk_size: int) -> int:
    value = (1 << 17) // chunk_size
    return 1 << (max(value, 1) - 1).bit_length()


def local_cumsum_chunk_indices(cu_seqlens: tuple[int, ...], chunk_size: int) -> torch.Tensor:
    block_t = local_cumsum_block_t(chunk_size)
    rows = []
    for seq, (begin, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        for local_block in range(math.ceil((end - begin) / block_t)):
            rows.append((seq, local_block))
    return torch.tensor(rows, dtype=torch.int64)


def iter_chunks(case: Case):
    if case.cu_seqlens is None:
        for batch in range(case.batch):
            for start in range(0, case.tokens, case.chunk_size):
                yield batch, start, min(start + case.chunk_size, case.tokens)
        return
    for begin, end in zip(case.cu_seqlens, case.cu_seqlens[1:]):
        for start in range(begin, end, case.chunk_size):
            yield 0, start, min(start + case.chunk_size, end)


def chunk_cumsum_kkt_reference(k: torch.Tensor, raw_g: torch.Tensor, beta: torch.Tensor, case: Case):
    g_cumsum = torch.zeros_like(raw_g, dtype=torch.float32)
    A = torch.zeros(
        (case.batch, case.heads, case.tokens, case.chunk_size), dtype=torch.float32
    )
    k_fp32 = k.to(torch.float32)
    beta_fp32 = beta.to(torch.float32)
    for batch, start, end in iter_chunks(case):
        for head in range(case.heads):
            gate = torch.cumsum(raw_g[batch, head, start:end].to(torch.float32), dim=0)
            g_cumsum[batch, head, start:end] = gate
            score = k_fp32[batch, head, start:end] @ k_fp32[batch, head, start:end].T
            gate_scale = torch.exp(torch.clamp(gate[:, None] - gate[None, :], -50.0, 50.0))
            value = score * gate_scale * beta_fp32[batch, head, start:end, None]
            valid = end - start
            strict_lower = torch.tril(torch.ones((valid, valid), dtype=torch.bool), diagonal=-1)
            A[batch, head, start:end, :valid] = torch.where(
                strict_lower, value, torch.zeros_like(value)
            )
    return g_cumsum, A


def make_inputs(case: Case, seed: int):
    torch.manual_seed(seed)
    k = (torch.randn(case.batch, case.heads, case.tokens, 128) * 0.2).to(case.dtype)
    raw_g = torch.randn(case.batch, case.heads, case.tokens, dtype=torch.float32) * 0.02
    beta = torch.sigmoid(torch.randn(case.batch, case.heads, case.tokens, dtype=torch.float32))
    return k.contiguous(), raw_g.contiguous(), beta.contiguous()


def assert_zero_contract(test: unittest.TestCase, A: torch.Tensor, case: Case):
    for batch, start, end in iter_chunks(case):
        valid = end - start
        block = A[batch, :, start:end]
        test.assertEqual(torch.count_nonzero(torch.triu(block[..., :valid], diagonal=0)).item(), 0)
        if valid < case.chunk_size:
            test.assertEqual(torch.count_nonzero(block[..., valid:]).item(), 0)


class ChunkCumsumKktReferenceTest(unittest.TestCase):
    def test_dense_reference_contract(self):
        case = Case(2, 3, 129, 64, torch.float16)
        k, raw_g, beta = make_inputs(case, 20260725)
        g_cumsum, A = chunk_cumsum_kkt_reference(k, raw_g, beta, case)
        self.assertEqual(g_cumsum.shape, raw_g.shape)
        self.assertEqual(g_cumsum.dtype, torch.float32)
        self.assertEqual(A.shape, (2, 3, 129, 64))
        self.assertEqual(A.dtype, torch.float32)
        self.assertTrue(torch.isfinite(g_cumsum).all())
        self.assertTrue(torch.isfinite(A).all())
        assert_zero_contract(self, A, case)

    def test_varlen_cumsum_resets_at_sequence_and_chunk_boundaries(self):
        case = Case(1, 2, 130, 64, torch.bfloat16, (0, 65, 130))
        k, raw_g, beta = make_inputs(case, 20260726)
        g_cumsum, A = chunk_cumsum_kkt_reference(k, raw_g, beta, case)
        expected = torch.zeros_like(raw_g)
        for batch, start, end in iter_chunks(case):
            expected[batch, :, start:end] = torch.cumsum(raw_g[batch, :, start:end], dim=1)
        torch.testing.assert_close(g_cumsum, expected, rtol=0, atol=0)
        self.assertEqual(canonical_chunk_indices(case.cu_seqlens, case.chunk_size), (0, 0, 0, 1, 1, 0, 1, 1))
        assert_zero_contract(self, A, case)


NPU_CASES = (
    Case(1, 2, 128, 64, torch.float16),
    Case(1, 2, 129, 128, torch.float16),
    Case(1, 2, 128, 64, torch.bfloat16),
    Case(1, 2, 129, 128, torch.bfloat16),
    Case(1, 2, 130, 64, torch.float16, (0, 65, 130)),
    Case(1, 2, 258, 128, torch.float16, (0, 129, 258)),
    Case(1, 2, 130, 64, torch.bfloat16, (0, 65, 130)),
    Case(1, 2, 258, 128, torch.bfloat16, (0, 129, 258)),
)


DENSE_FP16_C64_EXACT_CASES = (
    Case(1, 1, 1, 64, torch.float16),
    Case(1, 1, 2, 64, torch.float16),
    Case(1, 2, 63, 64, torch.float16),
    Case(1, 2, 64, 64, torch.float16),
    Case(1, 3, 65, 64, torch.float16),
    Case(1, 2, 127, 64, torch.float16),
    Case(1, 4, 128, 64, torch.float16),
    Case(2, 1, 129, 64, torch.float16),
    Case(2, 3, 193, 64, torch.float16),
    Case(3, 4, 257, 64, torch.float16),
)


DENSE_BF16_C64_EXACT_CASES = tuple(
    Case(case.batch, case.heads, case.tokens, case.chunk_size, torch.bfloat16)
    for case in DENSE_FP16_C64_EXACT_CASES
)


DENSE_FP16_C128_EXACT_CASES = tuple(
    Case(case.batch, case.heads, case.tokens, 128, torch.float16)
    for case in DENSE_FP16_C64_EXACT_CASES
)


DENSE_BF16_C128_EXACT_CASES = tuple(
    Case(case.batch, case.heads, case.tokens, 128, torch.bfloat16)
    for case in DENSE_FP16_C64_EXACT_CASES
)


VARLEN_FP16_C64_EXACT_CASES = (
    Case(1, 1, 1, 64, torch.float16, (0, 1)),
    Case(1, 1, 2, 64, torch.float16, (0, 1, 2)),
    Case(1, 2, 63, 64, torch.float16, (0, 31, 63)),
    Case(1, 2, 64, 64, torch.float16, (0, 64)),
    Case(1, 3, 65, 64, torch.float16, (0, 1, 65)),
    Case(1, 2, 127, 64, torch.float16, (0, 63, 127)),
    Case(1, 4, 128, 64, torch.float16, (0, 64, 128)),
    Case(1, 1, 129, 64, torch.float16, (0, 65, 129)),
    Case(1, 3, 193, 64, torch.float16, (0, 1, 65, 193)),
    Case(1, 4, 257, 64, torch.float16, (0, 64, 129, 257)),
)


VARLEN_BF16_C64_EXACT_CASES = tuple(
    Case(1, case.heads, case.tokens, 64, torch.bfloat16, case.cu_seqlens)
    for case in VARLEN_FP16_C64_EXACT_CASES
)


VARLEN_FP16_C128_EXACT_CASES = tuple(
    Case(1, case.heads, case.tokens, 128, torch.float16, case.cu_seqlens)
    for case in VARLEN_FP16_C64_EXACT_CASES
)


VARLEN_BF16_C128_EXACT_CASES = tuple(
    Case(1, case.heads, case.tokens, 128, torch.bfloat16, case.cu_seqlens)
    for case in VARLEN_FP16_C64_EXACT_CASES
)


def run_npu_case(case: Case, seed: int):
    from fla_npu.ops import ascendc

    k, raw_g, beta = make_inputs(case, seed)
    expected_g, expected_A = chunk_cumsum_kkt_reference(k, raw_g, beta, case)
    indices = None if case.cu_seqlens is None else canonical_chunk_indices(case.cu_seqlens, case.chunk_size)
    actual_g, actual_A = ascendc.chunk_cumsum_kkt(
        k.npu(),
        raw_g.npu(),
        beta.npu(),
        cu_seqlens=case.cu_seqlens,
        chunk_indices=indices,
        chunk_size=case.chunk_size,
    )
    actual_g = actual_g.cpu()
    actual_A = actual_A.cpu()
    if not torch.isfinite(actual_g).all() or not torch.isfinite(actual_A).all():
        raise AssertionError("ChunkCumsumKkt produced NaN or Inf")
    torch.testing.assert_close(actual_g, expected_g, rtol=0, atol=0)
    torch.testing.assert_close(actual_A, expected_A, rtol=5e-3, atol=5e-3)
    assert_zero_contract(unittest.TestCase(), actual_A, case)


def _assert_exact(name: str, actual: torch.Tensor, expected: torch.Tensor):
    if torch.equal(actual, expected):
        return
    mismatch = torch.nonzero(actual != expected)
    first = tuple(int(value) for value in mismatch[0].tolist())
    diff = (actual - expected).abs()
    raise AssertionError(
        f"{name} is not bit-exact: equal={int((actual == expected).sum())}/{actual.numel()}, "
        f"first={first}, actual={actual[first].item()}, expected={expected[first].item()}, "
        f"max_abs={diff.max().item()}"
    )


def run_npu_exact_stitch_case(case: Case, seed: int):
    from fla_npu.ops import ascendc

    k, raw_g, beta = make_inputs(case, seed)
    k_npu = k.npu()
    raw_g_npu = raw_g.npu()
    beta_npu = beta.npu()
    cu_seqlens_npu = None
    local_indices_npu = None
    fused_indices = None
    if case.cu_seqlens is not None:
        cu_seqlens_npu = torch.tensor(case.cu_seqlens, dtype=torch.int64).npu()
        local_indices_npu = local_cumsum_chunk_indices(
            case.cu_seqlens, case.chunk_size
        ).npu()
        fused_indices = canonical_chunk_indices(case.cu_seqlens, case.chunk_size)
    baseline_g = ascendc.chunk_local_cumsum(
        raw_g_npu,
        case.chunk_size,
        cu_seqlens=cu_seqlens_npu,
        chunk_indices_out=local_indices_npu,
        reverse=False,
        scale=1.0,
        head_first=True,
        output_dtype="float32",
    )
    baseline_A = ascendc.chunk_scaled_dot_kkt(
        k_npu,
        baseline_g,
        beta_npu,
        cu_seqlens=case.cu_seqlens,
        chunk_indices=fused_indices,
        chunk_size=case.chunk_size,
    )
    actual_g, actual_A = ascendc.chunk_cumsum_kkt(
        k_npu,
        raw_g_npu,
        beta_npu,
        cu_seqlens=case.cu_seqlens,
        chunk_indices=fused_indices,
        chunk_size=case.chunk_size,
    )
    torch.npu.synchronize()
    baseline_g = baseline_g.cpu()
    baseline_A = baseline_A.cpu()
    actual_g = actual_g.cpu()
    actual_A = actual_A.cpu()
    if not torch.isfinite(actual_g).all() or not torch.isfinite(actual_A).all():
        raise AssertionError("ChunkCumsumKkt produced NaN or Inf")
    _assert_exact("g_cumsum", actual_g, baseline_g)
    _assert_exact("A_raw", actual_A, baseline_A)
    assert_zero_contract(unittest.TestCase(), actual_A, case)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--exact-dense-fp16-c64", action="store_true")
    parser.add_argument("--exact-dense-bf16-c64", action="store_true")
    parser.add_argument("--exact-dense-fp16-c128", action="store_true")
    parser.add_argument("--exact-dense-bf16-c128", action="store_true")
    parser.add_argument("--exact-varlen-fp16-c64", action="store_true")
    parser.add_argument("--exact-varlen-bf16-c64", action="store_true")
    parser.add_argument("--exact-varlen-fp16-c128", action="store_true")
    parser.add_argument("--exact-varlen-bf16-c128", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", type=int, default=int(os.environ.get("TEST_DEVICE_ID", "0")))
    args = parser.parse_args(argv)
    if args.cpu_only:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ChunkCumsumKktReferenceTest)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1

    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    if args.exact_dense_fp16_c64:
        cases = DENSE_FP16_C64_EXACT_CASES
    elif args.exact_dense_bf16_c64:
        cases = DENSE_BF16_C64_EXACT_CASES
    elif args.exact_dense_fp16_c128:
        cases = DENSE_FP16_C128_EXACT_CASES
    elif args.exact_dense_bf16_c128:
        cases = DENSE_BF16_C128_EXACT_CASES
    elif args.exact_varlen_fp16_c64:
        cases = VARLEN_FP16_C64_EXACT_CASES
    elif args.exact_varlen_bf16_c64:
        cases = VARLEN_BF16_C64_EXACT_CASES
    elif args.exact_varlen_fp16_c128:
        cases = VARLEN_FP16_C128_EXACT_CASES
    elif args.exact_varlen_bf16_c128:
        cases = VARLEN_BF16_C128_EXACT_CASES
    else:
        cases = NPU_CASES
    cases = cases if args.limit is None else cases[: args.limit]
    for index, case in enumerate(cases):
        if (args.exact_dense_fp16_c64 or args.exact_dense_bf16_c64 or
                args.exact_dense_fp16_c128 or args.exact_dense_bf16_c128 or
                args.exact_varlen_fp16_c64 or args.exact_varlen_bf16_c64 or
                args.exact_varlen_fp16_c128 or args.exact_varlen_bf16_c128):
            run_npu_exact_stitch_case(case, 20260800 + index)
        else:
            run_npu_case(case, 20260725 + index)
        print(f"[PASS] {index + 1}/{len(cases)}: {case}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
