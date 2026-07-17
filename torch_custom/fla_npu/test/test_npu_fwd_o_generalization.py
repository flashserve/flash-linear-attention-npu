# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""ChunkFwdO workspace 流水回归，覆盖 fixed/varlen、MHA/GVA、chunk 64/128 和尾块。"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

import ct
import torch

from fla_npu.ops import ascendc as ascendc_ops


torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)


@dataclass(frozen=True)
class FwdOCase:
    name: str
    batch: int
    k_heads: int
    v_heads: int
    tokens: int
    v_dim: int
    chunk_size: int
    cu_seqlens: Optional[tuple[int, ...]] = None
    k_dim: int = 128


CASES = (
    FwdOCase("fixed_mha_chunk64_tail", 2, 4, 4, 129, 128, 64),
    FwdOCase("fixed_gva_chunk128_tail_v256", 1, 2, 4, 257, 256, 128),
    FwdOCase("varlen_gva_chunk64_tail", 1, 2, 4, 321, 128, 64, (0, 65, 193, 321)),
)


def _chunk_indices(cu_seqlens: tuple[int, ...], chunk_size: int) -> list[int]:
    indices = []
    for sequence, (begin, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        for chunk in range(math.ceil((end - begin) / chunk_size)):
            indices.extend((sequence, chunk))
    return indices


def _reference(
    case: FwdOCase,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    npu_aligned: bool,
) -> torch.Tensor:
    compute_dtype = torch.float32 if npu_aligned else torch.float64
    output = torch.zeros((case.batch, case.v_heads, case.tokens, case.v_dim), dtype=compute_dtype)
    head_ratio = case.v_heads // case.k_heads
    sequences = case.cu_seqlens or tuple(range(case.batch + 1))
    state_offset = 0

    for sequence, (begin, end) in enumerate(zip(sequences, sequences[1:])):
        batch_index = 0 if case.cu_seqlens is not None else sequence
        sequence_begin = begin if case.cu_seqlens is not None else 0
        sequence_end = end if case.cu_seqlens is not None else case.tokens
        chunk_count = math.ceil((sequence_end - sequence_begin) / case.chunk_size)
        for head in range(case.v_heads):
            key_head = head // head_ratio
            for chunk in range(chunk_count):
                chunk_begin = sequence_begin + chunk * case.chunk_size
                chunk_end = min(chunk_begin + case.chunk_size, sequence_end)
                q_chunk = q[batch_index, key_head, chunk_begin:chunk_end].to(compute_dtype)
                k_chunk = k[batch_index, key_head, chunk_begin:chunk_end].to(compute_dtype)
                v_chunk = v[batch_index, head, chunk_begin:chunk_end].to(compute_dtype)
                g_chunk = g[batch_index, head, chunk_begin:chunk_end].to(compute_dtype)
                state_index = state_offset + chunk if case.cu_seqlens is not None else chunk
                state = h[batch_index, head, state_index].to(compute_dtype)

                attention = q_chunk @ k_chunk.transpose(0, 1)
                attention *= torch.exp(g_chunk[:, None] - g_chunk[None, :])
                attention = torch.tril(attention)
                if npu_aligned:
                    attention = attention.to(q.dtype).float()
                state_output = (q_chunk @ state) * torch.exp(g_chunk)[:, None]
                output[batch_index, head, chunk_begin:chunk_end] = (
                    state_output + attention @ v_chunk
                ) * scale
        state_offset += chunk_count
    return output.to(q.dtype).float() if npu_aligned else output


def _run_case(case: FwdOCase) -> None:
    torch.manual_seed(20260717)
    dtype = torch.bfloat16
    scale = 1.0 / math.sqrt(case.k_dim)
    chunk_count = (
        sum(math.ceil((end - begin) / case.chunk_size)
            for begin, end in zip(case.cu_seqlens, case.cu_seqlens[1:]))
        if case.cu_seqlens is not None
        else math.ceil(case.tokens / case.chunk_size)
    )
    q = torch.randn(case.batch, case.k_heads, case.tokens, case.k_dim, dtype=dtype) * 0.1
    k = torch.randn_like(q) * 0.1
    v = torch.randn(case.batch, case.v_heads, case.tokens, case.v_dim, dtype=dtype) * 0.1
    h = torch.randn(case.batch, case.v_heads, chunk_count, case.k_dim, case.v_dim, dtype=dtype) * 0.1
    # g=0 隔离 QK/AttnV 和 QH 两条 workspace 流水，避免设备 Exp 近似影响 CPU 双标杆。
    g = torch.zeros(case.batch, case.v_heads, case.tokens, dtype=torch.float32)

    reference_fp64 = _reference(case, q, k, v, h, g, scale, npu_aligned=False)
    reference_npu = _reference(case, q, k, v, h, g, scale, npu_aligned=True)
    chunk_indices = _chunk_indices(case.cu_seqlens, case.chunk_size) if case.cu_seqlens is not None else None
    output = ascendc_ops.npu_chunk_fwd_o(
        q.npu(),
        k.npu(),
        v.npu(),
        h.npu(),
        scale,
        g=g.npu(),
        g_gamma=None,
        cu_seqlens=list(case.cu_seqlens) if case.cu_seqlens is not None else None,
        chunk_indices=chunk_indices,
        chunk_size=case.chunk_size,
        transpose_state_layout=False,
    )
    result = ct.dual(output.cpu().float(), reference_fp64.float(), reference_npu.float(), level="L1")
    if not bool(result.get("success")):
        raise AssertionError(f"{case.name} dual benchmark failed: {result}")
    print(f"{case.name}: PASS")


def main() -> None:
    torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))
    selected = os.environ.get("FWD_O_CASE", "").strip()
    cases = [case for case in CASES if not selected or case.name == selected]
    if not cases:
        raise SystemExit(f"unknown FWD_O_CASE={selected}")
    for case in cases:
        _run_case(case)


if __name__ == "__main__":
    main()
