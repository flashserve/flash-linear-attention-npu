# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""`fla_npu.ops.ascendc` 的 ChunkKdaFwdFinalize 实机回归。"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

import ct
import torch

from fla_npu.ops.ascendc import chunk_kda_fwd_finalize


torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)


@dataclass(frozen=True)
class FinalizeCase:
    name: str
    output_layout: str
    batch: int
    tokens: int
    state_v_first: bool = False
    cu_seqlens: Optional[tuple[int, ...]] = None
    rank4_packed_v_new: bool = False
    explicit_chunk_indices: bool = False


CASES = (
    FinalizeCase("dense_bsnd", "BSND", batch=2, tokens=65),
    FinalizeCase(
        "dense_bnsd_state_v_first",
        "BNSD",
        batch=1,
        tokens=64,
        state_v_first=True,
    ),
    FinalizeCase(
        "varlen_tnd_rank4_v_new",
        "TND",
        batch=1,
        tokens=130,
        cu_seqlens=(0, 65, 130),
        rank4_packed_v_new=True,
    ),
    FinalizeCase(
        "varlen_ntd_explicit_indices",
        "NTD",
        batch=1,
        tokens=70,
        state_v_first=True,
        cu_seqlens=(0, 70),
        explicit_chunk_indices=True,
    ),
)


def _chunk_indices(cu_seqlens: tuple[int, ...]) -> list[int]:
    return [
        value
        for sequence, (begin, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:]))
        for chunk in range(math.ceil((end - begin) / 64))
        for value in (sequence, chunk)
    ]


def _sequence_ranges(tokens: int, cu_seqlens: Optional[tuple[int, ...]]):
    boundaries = cu_seqlens or (0, tokens)
    return tuple(zip(boundaries, boundaries[1:]))


def _make_inputs(case: FinalizeCase):
    torch.manual_seed(20260917)
    heads = 2
    packed = case.output_layout in {"TND", "NTD"}
    ranges = _sequence_ranges(case.tokens, case.cu_seqlens)
    chunks = sum(math.ceil((end - begin) / 64) for begin, end in ranges)
    if not packed:
        chunks = math.ceil(case.tokens / 64)

    qg = (torch.randn(case.batch, heads, case.tokens, 128) * 0.1).to(torch.bfloat16)
    aqk = (torch.randn(case.batch, heads, case.tokens, 64) * 0.1).to(torch.bfloat16)
    v_new = (torch.randn(case.batch, heads, case.tokens, 128) * 0.1).to(torch.bfloat16)
    h_kv = (
        torch.randn(case.batch, heads, chunks, 128, 128) * 0.1
    ).to(torch.bfloat16)

    for batch in range(case.batch):
        batch_ranges = ranges if packed else ((0, case.tokens),)
        for begin, end in batch_ranges:
            for chunk_begin in range(begin, end, 64):
                chunk_end = min(chunk_begin + 64, end)
                valid_rows = chunk_end - chunk_begin
                causal = (
                    torch.arange(64).unsqueeze(0)
                    <= torch.arange(valid_rows).unsqueeze(1)
                )
                aqk[batch, :, chunk_begin:chunk_end] *= causal.unsqueeze(0)

    h_storage = h_kv.transpose(-1, -2).contiguous() if case.state_v_first else h_kv
    qg_input = qg[0] if packed else qg
    aqk_input = aqk[0] if packed else aqk
    if packed and not case.rank4_packed_v_new:
        v_new_input = v_new[0]
    else:
        v_new_input = v_new
    return qg_input, aqk_input, v_new_input, h_storage, h_kv


def _reference(case: FinalizeCase, qg, aqk, v_new, h_kv, compute_dtype):
    packed = case.output_layout in {"TND", "NTD"}
    qg = qg.unsqueeze(0) if packed else qg
    aqk = aqk.unsqueeze(0) if packed else aqk
    if v_new.ndim == 3:
        v_new = v_new.unsqueeze(0)
    output = torch.empty(
        (case.batch, qg.shape[1], case.tokens, 128), dtype=compute_dtype
    )
    for batch in range(case.batch):
        global_chunk = 0
        ranges = _sequence_ranges(case.tokens, case.cu_seqlens if packed else None)
        for begin, end in ranges:
            for chunk_begin in range(begin, end, 64):
                chunk_end = min(chunk_begin + 64, end)
                valid_rows = chunk_end - chunk_begin
                state_term = torch.bmm(
                    qg[batch, :, chunk_begin:chunk_end].to(compute_dtype),
                    h_kv[batch, :, global_chunk].to(compute_dtype),
                )
                local_term = torch.bmm(
                    aqk[batch, :, chunk_begin:chunk_end, :valid_rows].to(compute_dtype),
                    v_new[batch, :, chunk_begin:chunk_end].to(compute_dtype),
                )
                output[batch, :, chunk_begin:chunk_end] = state_term + local_term
                global_chunk += 1

    if case.output_layout == "BSND":
        return output.permute(0, 2, 1, 3).contiguous()
    if case.output_layout == "BNSD":
        return output
    if case.output_layout == "TND":
        return output[0].transpose(0, 1).contiguous()
    return output[0]


@torch.inference_mode()
def _run_case(case: FinalizeCase, device: torch.device) -> None:
    qg, aqk, v_new, h_storage, h_kv = _make_inputs(case)
    reference_fp64 = _reference(case, qg, aqk, v_new, h_kv, torch.float64)
    reference_npu = _reference(case, qg, aqk, v_new, h_kv, torch.float32)
    reference_npu = reference_npu.to(torch.bfloat16).float()
    chunk_indices = (
        _chunk_indices(case.cu_seqlens)
        if case.cu_seqlens is not None and case.explicit_chunk_indices
        else None
    )
    actual = chunk_kda_fwd_finalize(
        qg.to(device),
        aqk.to(device),
        v_new.to(device),
        h_storage.to(device),
        output_layout=case.output_layout,
        state_v_first=case.state_v_first,
        cu_seqlens=case.cu_seqlens,
        chunk_indices=chunk_indices,
    ).cpu()
    expected_shape = tuple(reference_npu.shape)
    if actual.dtype != torch.bfloat16 or tuple(actual.shape) != expected_shape:
        raise AssertionError(
            f"{case.name}: expected BF16 {expected_shape}, got {actual.dtype} {tuple(actual.shape)}"
        )
    if not actual.is_contiguous():
        raise AssertionError(f"{case.name}: output must be contiguous")
    if not torch.isfinite(actual).all():
        raise AssertionError(f"{case.name}: output contains non-finite values")
    result = ct.dual(
        actual.float(), reference_fp64.float(), reference_npu, level="L1"
    )
    if not bool(result.get("success")):
        raise AssertionError(f"{case.name}: dual benchmark failed: {result}")
    max_abs = (actual.float() - reference_npu).abs().max().item()
    print(f"{case.name}: PASS, max_abs={max_abs:.6f}")


def main() -> None:
    device_id = int(os.environ.get("TEST_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    device = torch.device(f"npu:{device_id}")
    for case in CASES:
        _run_case(case, device)


if __name__ == "__main__":
    main()
