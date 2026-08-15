# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# CANN Open Software License Agreement Version 2.0.
# -----------------------------------------------------------------------------------------------------------

"""A/B precision coverage for the Phase 2 fused KKT + solve_tri kernel."""

from __future__ import annotations

import math
import os

import torch
import torch_npu

from fla_npu.ops import ascendc


DEVICE_ID = int(os.environ.get("TEST_DEVICE_ID", 0))


def canonical_chunks(cu_seqlens: list[int] | None, chunk_size: int) -> list[int] | None:
    if cu_seqlens is None:
        return None
    result = []
    for sequence, (begin, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        for local_chunk in range(math.ceil((end - begin) / chunk_size)):
            result.extend((sequence, local_chunk))
    return result


def valid_token_rows(tensor: torch.Tensor, cu_seqlens: list[int] | None, chunk_size: int):
    sequences = [(batch, 0, tensor.shape[2]) for batch in range(tensor.shape[0])]
    if cu_seqlens is not None:
        sequences = [(0, begin, end) for begin, end in zip(cu_seqlens, cu_seqlens[1:])]
    for batch, begin, end in sequences:
        for chunk_begin in range(begin, end, chunk_size):
            chunk_end = min(chunk_begin + chunk_size, end)
            yield tensor[batch, :, chunk_begin:chunk_end, :]


def valid_square_values(tensor: torch.Tensor, cu_seqlens: list[int] | None,
                        chunk_size: int) -> torch.Tensor:
    values = []
    sequences = [(batch, 0, tensor.shape[2]) for batch in range(tensor.shape[0])]
    if cu_seqlens is not None:
        sequences = [(0, begin, end) for begin, end in zip(cu_seqlens, cu_seqlens[1:])]
    for batch, begin, end in sequences:
        for chunk_begin in range(begin, end, chunk_size):
            valid = min(chunk_size, end - chunk_begin)
            values.append(tensor[batch, :, chunk_begin:chunk_begin + valid, :valid].flatten())
    return torch.cat(values)


def cpu_inverse_reference(raw: torch.Tensor, cu_seqlens: list[int] | None,
                          chunk_size: int) -> torch.Tensor:
    raw_cpu = raw.float().cpu()
    reference = torch.zeros_like(raw_cpu)
    sequences = [(batch, 0, raw.shape[2]) for batch in range(raw.shape[0])]
    if cu_seqlens is not None:
        sequences = [(0, begin, end) for begin, end in zip(cu_seqlens, cu_seqlens[1:])]
    for batch, begin, end in sequences:
        for chunk_begin in range(begin, end, chunk_size):
            valid = min(chunk_size, end - chunk_begin)
            for head in range(raw.shape[1]):
                block = raw_cpu[batch, head, chunk_begin:chunk_begin + valid, :valid].double()
                inverse = torch.linalg.inv(torch.eye(valid, dtype=torch.float64) + block)
                reference[batch, head, chunk_begin:chunk_begin + valid, :valid] = inverse.float()
    return reference


def run_case(name: str, dtype: torch.dtype, batch: int, heads: int, tokens: int,
             chunk_size: int, cu_seqlens: list[int] | None) -> None:
    print(f"{name}: starting", flush=True)
    torch.manual_seed(20260724)
    if cu_seqlens is not None and batch != 1:
        raise ValueError("varlen cases require physical batch=1")
    k = (torch.randn(batch, heads, tokens, 128, dtype=dtype) * 0.05).npu()
    g = torch.cumsum(-torch.rand(batch, heads, tokens, dtype=torch.float32) * 0.1, dim=-1).npu()
    beta = torch.sigmoid(torch.randn(batch, heads, tokens, dtype=torch.float32)).npu()
    chunk_indices = canonical_chunks(cu_seqlens, chunk_size)

    raw = ascendc.chunk_scaled_dot_kkt(
        k, g, beta, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, chunk_size=chunk_size
    )
    if cu_seqlens is None:
        expected = ascendc.solve_tri(raw.to(dtype), layout="bhtd")
    else:
        token_first = raw.transpose(1, 2).contiguous().squeeze(0)
        token_first = ascendc.solve_tri(
            token_first.to(dtype),
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            layout="tnd",
        )
        expected = token_first.unsqueeze(0).transpose(1, 2).contiguous()
    torch.npu.synchronize()
    print(f"{name}: two-kernel baseline complete", flush=True)
    actual = ascendc.chunk_kkt_solve_tri(
        k, g, beta, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, chunk_size=chunk_size
    )
    torch.npu.synchronize()
    print(f"{name}: fused kernel complete", flush=True)
    if os.environ.get("GDN_PHASE2_DEBUG") == "1":
        print(f"{name}: fused first8={actual.flatten()[:8].float().cpu().tolist()}", flush=True)

    baseline_finite = bool(torch.isfinite(expected.float()).all())
    if baseline_finite:
        for chunk_index, (expected_chunk, actual_chunk) in enumerate(zip(
            valid_token_rows(expected, cu_seqlens, chunk_size),
            valid_token_rows(actual, cu_seqlens, chunk_size),
        )):
            if torch.equal(expected_chunk.cpu(), actual_chunk.cpu()):
                continue
            expected_cpu = expected_chunk.cpu()
            actual_cpu = actual_chunk.cpu()
            difference = expected_cpu.float() - actual_cpu.float()
            mismatch = torch.nonzero(expected_cpu != actual_cpu, as_tuple=False)
            first = tuple(int(value) for value in mismatch[0])
            max_abs = float(difference.abs().max())
            actual_nonzero = int(torch.count_nonzero(actual_cpu))
            actual_min = float(actual_cpu.float().min())
            actual_max = float(actual_cpu.float().max())
            raw_chunk = next(iter(valid_token_rows(raw.to(dtype), cu_seqlens, chunk_size))).cpu()
            raw_equal = torch.equal(raw_chunk, actual_cpu)
            raw_max_abs = float((raw_chunk.float() - actual_cpu.float()).abs().max())
            raise AssertionError(
                f"{name}: valid A rows are not bit exact; chunk={chunk_index}, "
                f"mismatches={mismatch.shape[0]}, first={first}, "
                f"expected={float(expected_cpu[first]):.6e}, actual={float(actual_cpu[first]):.6e}, "
                f"max_abs={max_abs:.6e}, actual_nonzero={actual_nonzero}, "
                f"actual_range=[{actual_min:.6e}, {actual_max:.6e}], "
                f"equals_raw_kkt={raw_equal}, raw_max_abs={raw_max_abs:.6e}"
            )
    if not bool(torch.isfinite(actual.float()).all()):
        raise AssertionError(f"{name}: fused output contains non-finite values")
    if not baseline_finite:
        reference = cpu_inverse_reference(raw, cu_seqlens, chunk_size)
        actual_cpu = valid_square_values(actual.float().cpu(), cu_seqlens, chunk_size)
        reference_cpu = valid_square_values(reference, cu_seqlens, chunk_size)
        difference = actual_cpu - reference_cpu
        max_abs = float(difference.abs().max())
        cosine = float(torch.nn.functional.cosine_similarity(
            actual_cpu, reference_cpu, dim=0
        ))
        if max_abs > 5e-3 or cosine < 0.999:
            raise AssertionError(
                f"{name}: standalone solve_tri baseline is non-finite and fused output "
                f"does not meet the CPU inverse reference; max_abs={max_abs:.6e}, cosine={cosine:.9f}"
            )
        print(
            f"{name}: standalone solve_tri baseline is non-finite; "
            f"CPU reference max_abs={max_abs:.6e}, cosine={cosine:.9f}",
            flush=True,
        )
    print(f"{name}: PASS")


def main() -> None:
    torch.npu.set_device(DEVICE_ID)
    torch.npu.set_compile_mode(jit_compile=False)
    cases = (
        ("dense_bf16_b1_h1_c64", torch.bfloat16, 1, 1, 64, 64, None),
        ("dense_bf16_b2_h4_c64", torch.bfloat16, 2, 4, 128, 64, None),
        ("dense_fp16_b3_h8_tail_c64", torch.float16, 3, 8, 129, 64, None),
        ("dense_bf16_b1_h1_c128", torch.bfloat16, 1, 1, 128, 128, None),
        ("varlen_bf16_h4_tail_c64", torch.bfloat16, 1, 4, 130, 64, [0, 1, 65, 130]),
        ("varlen_fp16_h4_tail_c128", torch.float16, 1, 4, 130, 128, [0, 65, 130]),
    )
    limit = int(os.environ.get("GDN_PHASE2_CASE_LIMIT", len(cases)))
    selected_cases = cases[:limit]
    for case in selected_cases:
        run_case(*case)
    print(f"ChunkKktSolveTri A/B: {len(selected_cases)}/{len(selected_cases)} PASS")


if __name__ == "__main__":
    main()
