#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest
import torch
import torch_npu
import ascend_ops


CASE_FILE = (
    Path(__file__).resolve().parents[4]
    / "tests/op_cases/chunk_gated_delta_rule_fwd_h.json"
)
with CASE_FILE.open("r", encoding="utf-8") as case_file:
    STAGE0_SPEC = json.load(case_file)["stage0_debug"]

CHUNK_SIZE = int(STAGE0_SPEC["chunk_size"])
K_DIM = int(STAGE0_SPEC["k_dim"])
V_DIM = int(STAGE0_SPEC["v_dim"])
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
MAX_ABS_TOL = float(STAGE0_SPEC["max_abs_tolerance"])
RELATIVE_L2_TOL = float(STAGE0_SPEC["relative_l2_tolerance"])
ALL_CASES = {
    case["id"]: case
    for case in STAGE0_SPEC["positive_cases"] + STAGE0_SPEC["production_only_cases"]
}


def _chunk_count(length: int) -> int:
    return (length + CHUNK_SIZE - 1) // CHUNK_SIZE


def _stage0_storage_golden(
    w: torch.Tensor,
    h_entry: torch.Tensor,
    cu_seqlens: Optional[List[int]],
) -> torch.Tensor:
    """Compute P = W @ H from the same low-precision storage boundary as the Cube kernel."""
    batch, num_heads, total_tokens, _ = w.shape
    output = torch.empty((batch, num_heads, total_tokens, V_DIM), dtype=torch.float32)
    if cu_seqlens is None:
        sequences = [(batch_idx, 0, total_tokens, 0) for batch_idx in range(batch)]
    else:
        sequences = []
        chunk_offset = 0
        for begin, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            sequences.append((0, begin, end, chunk_offset))
            chunk_offset += _chunk_count(end - begin)

    w_fp32 = w.float()
    h_storage_fp32 = h_entry.to(w.dtype).float()
    for batch_idx, begin, end, chunk_offset in sequences:
        for local_chunk in range(_chunk_count(end - begin)):
            token_begin = begin + local_chunk * CHUNK_SIZE
            token_end = min(token_begin + CHUNK_SIZE, end)
            h_chunk = chunk_offset + local_chunk
            for head_idx in range(num_heads):
                output[batch_idx, head_idx, token_begin:token_end] = (
                    w_fp32[batch_idx, head_idx, token_begin:token_end]
                    @ h_storage_fp32[batch_idx, head_idx, h_chunk]
                )
    return output


def _coordinate_inputs(
    dtype: torch.dtype,
    batch: int,
    num_heads: int,
    total_tokens: int,
    total_chunks: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    w = torch.zeros((batch, num_heads, total_tokens, K_DIM), dtype=dtype)
    h_entry = torch.zeros((batch, num_heads, total_chunks, K_DIM, V_DIM), dtype=dtype)
    for batch_idx in range(batch):
        for head_idx in range(num_heads):
            row_ids = torch.arange(total_tokens)
            k_ids = (row_ids + 11 * head_idx + 17 * batch_idx) % K_DIM
            signs = torch.where((row_ids & 1) == 0, 1.0, -1.0).to(dtype)
            w[batch_idx, head_idx, row_ids, k_ids] = signs
            for chunk_idx in range(total_chunks):
                base = 0.25 * batch_idx + 0.125 * head_idx + 0.0625 * chunk_idx
                row = torch.arange(K_DIM, dtype=torch.float32).unsqueeze(1)
                col = torch.arange(V_DIM, dtype=torch.float32).unsqueeze(0)
                h_entry[batch_idx, head_idx, chunk_idx] = (
                    base + row / 256.0 - col / 512.0
                ).to(dtype)
    return w, h_entry


def _random_inputs(
    dtype: torch.dtype,
    batch: int,
    num_heads: int,
    total_tokens: int,
    total_chunks: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260824)
    w = torch.randn(
        (batch, num_heads, total_tokens, K_DIM), generator=generator, dtype=torch.float32
    ).mul_(0.05).to(dtype)
    h_entry = torch.randn(
        (batch, num_heads, total_chunks, K_DIM, V_DIM), generator=generator, dtype=torch.float32
    ).mul_(0.05).to(dtype)
    return w, h_entry


def _run_stage0_debug(
    w: torch.Tensor,
    h_entry: torch.Tensor,
    cu_seqlens: Optional[List[int]],
) -> torch.Tensor:
    return torch.ops.ascend_ops.chunk_gated_delta_rule_fwd_h_stage0_debug(
        w.npu(),
        h_entry.npu(),
        chunk_size=CHUNK_SIZE,
        cu_seqlens=cu_seqlens,
    ).cpu()


def _assert_stage0_close(actual: torch.Tensor, expected: torch.Tensor, exact: bool) -> None:
    assert actual.dtype == torch.float32
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    if exact:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return

    diff = (actual - expected).abs().double()
    max_abs = float(diff.max().item())
    relative_l2 = float(diff.norm().div(expected.double().norm().clamp_min(1e-30)).item())
    assert max_abs <= MAX_ABS_TOL, f"max_abs={max_abs} exceeds {MAX_ABS_TOL}"
    assert relative_l2 <= RELATIVE_L2_TOL, (
        f"relative_l2={relative_l2} exceeds {RELATIVE_L2_TOL}"
    )


def _case_cu_seqlens(case: Dict) -> Optional[List[int]]:
    if case["layout"] == "dense":
        return None
    return [int(value) for value in case["cu_seqlens"]]


def _canonical_chunk_indices(cu_seqlens: Optional[List[int]]) -> Optional[List[int]]:
    if cu_seqlens is None:
        return None
    indices = []
    for sequence_idx, (begin, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        for local_chunk in range(_chunk_count(end - begin)):
            indices.extend([sequence_idx, local_chunk])
    return indices


def _initial_states(num_sequences: int, num_heads: int) -> torch.Tensor:
    states = torch.empty((num_sequences, num_heads, K_DIM, V_DIM), dtype=torch.float32)
    identity = torch.eye(K_DIM, V_DIM, dtype=torch.float32)
    for sequence_idx in range(num_sequences):
        scale = 1.0 + 0.5 * sequence_idx
        states[sequence_idx] = identity * scale
    return states


def _expected_production_outputs(
    w: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: Optional[List[int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, num_heads, total_tokens, _ = w.shape
    total_chunks = (
        _chunk_count(total_tokens)
        if cu_seqlens is None
        else sum(
            _chunk_count(end - begin)
            for begin, end in zip(cu_seqlens[:-1], cu_seqlens[1:])
        )
    )
    expected_h = torch.empty((batch, num_heads, total_chunks, K_DIM, V_DIM), dtype=w.dtype)
    expected_v_new = torch.empty((batch, num_heads, total_tokens, V_DIM), dtype=w.dtype)
    if cu_seqlens is None:
        sequences = [(batch_idx, 0, total_tokens, batch_idx, 0) for batch_idx in range(batch)]
    else:
        sequences = []
        chunk_offset = 0
        for sequence_idx, (begin, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
            sequences.append((0, begin, end, sequence_idx, chunk_offset))
            chunk_offset += _chunk_count(end - begin)

    w_fp32 = w.float()
    for batch_idx, begin, end, sequence_idx, chunk_offset in sequences:
        state = initial_state[sequence_idx]
        chunk_count = _chunk_count(end - begin)
        expected_h[batch_idx, :, chunk_offset:chunk_offset + chunk_count] = (
            state.to(w.dtype).unsqueeze(1).expand(num_heads, chunk_count, K_DIM, V_DIM)
        )
        expected_v_new[batch_idx, :, begin:end] = -torch.matmul(
            w_fp32[batch_idx, :, begin:end], state
        ).to(w.dtype)
    return expected_h, expected_v_new


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype_name", STAGE0_SPEC["dtypes"])
@pytest.mark.parametrize("input_kind", STAGE0_SPEC["input_kinds"])
@pytest.mark.parametrize(
    "case",
    STAGE0_SPEC["positive_cases"],
    ids=[case["id"] for case in STAGE0_SPEC["positive_cases"]],
)
def test_stage0_precision(dtype_name, input_kind, case):
    dtype = DTYPES[dtype_name]
    batch = int(case["batch"])
    num_heads = int(case["num_heads"])
    cu_seqlens = _case_cu_seqlens(case)
    total_tokens = int(case.get("total_tokens", cu_seqlens[-1] if cu_seqlens else 0))
    total_chunks = (
        _chunk_count(total_tokens)
        if cu_seqlens is None
        else sum(
            _chunk_count(end - begin)
            for begin, end in zip(cu_seqlens[:-1], cu_seqlens[1:])
        )
    )
    make_inputs = _coordinate_inputs if input_kind == "coordinate" else _random_inputs
    w, h_entry = make_inputs(dtype, batch, num_heads, total_tokens, total_chunks)

    expected = _stage0_storage_golden(w, h_entry, cu_seqlens=cu_seqlens)
    actual = _run_stage0_debug(w, h_entry, cu_seqlens=cu_seqlens)

    _assert_stage0_close(actual, expected, exact=input_kind == "coordinate")


def test_stage0_debug_interface_exists():
    assert hasattr(torch.ops.ascend_ops, "chunk_gated_delta_rule_fwd_h_stage0_debug")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype_name", STAGE0_SPEC["dtypes"])
@pytest.mark.parametrize("case_id", STAGE0_SPEC["production_cases"])
@pytest.mark.parametrize("output_final_state", [False, True])
def test_stage0_production_path(dtype_name, case_id, output_final_state):
    case = ALL_CASES[case_id]
    dtype = DTYPES[dtype_name]
    batch = int(case["batch"])
    num_heads = int(case["num_heads"])
    cu_seqlens = _case_cu_seqlens(case)
    total_tokens = int(case.get("total_tokens", cu_seqlens[-1] if cu_seqlens else 0))
    num_sequences = batch if cu_seqlens is None else len(cu_seqlens) - 1
    total_chunks = (
        _chunk_count(total_tokens)
        if cu_seqlens is None
        else sum(
            _chunk_count(end - begin)
            for begin, end in zip(cu_seqlens[:-1], cu_seqlens[1:])
        )
    )
    w, _ = _coordinate_inputs(dtype, batch, num_heads, total_tokens, total_chunks)
    k = torch.zeros((batch, num_heads, total_tokens, K_DIM), dtype=dtype)
    u = torch.zeros((batch, num_heads, total_tokens, V_DIM), dtype=dtype)
    g = torch.zeros((batch, num_heads, total_tokens), dtype=torch.float32)
    initial_state = _initial_states(num_sequences, num_heads)
    chunk_indices = _canonical_chunk_indices(cu_seqlens)

    h, v_new, final_state = torch.ops.ascend_ops.chunk_gated_delta_rule_fwd_h(
        k.npu(),
        w.npu(),
        u.npu(),
        g.npu(),
        initial_state=initial_state.npu(),
        output_final_state=output_final_state,
        chunk_size=CHUNK_SIZE,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    expected_h, expected_v_new = _expected_production_outputs(w, initial_state, cu_seqlens)
    torch.testing.assert_close(h.cpu(), expected_h, rtol=0, atol=0)
    torch.testing.assert_close(v_new.cpu(), expected_v_new, rtol=0, atol=0)
    if output_final_state:
        torch.testing.assert_close(final_state.cpu(), initial_state, rtol=0, atol=0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype_name", STAGE0_SPEC["dtypes"])
def test_stage0_short_tail_fallback_and_debug_contract(dtype_name):
    case = STAGE0_SPEC["short_tail_case"]
    dtype = DTYPES[dtype_name]
    batch = int(case["batch"])
    num_heads = int(case["num_heads"])
    total_tokens = int(case["total_tokens"])
    w, h_entry = _coordinate_inputs(dtype, batch, num_heads, total_tokens, 1)
    with pytest.raises(RuntimeError, match=case["expected_debug_error"]):
        _run_stage0_debug(w, h_entry, cu_seqlens=None)

    k = torch.zeros((batch, num_heads, total_tokens, K_DIM), dtype=dtype)
    u = torch.zeros((batch, num_heads, total_tokens, V_DIM), dtype=dtype)
    g = torch.zeros((batch, num_heads, total_tokens), dtype=torch.float32)
    initial_state = (
        torch.eye(K_DIM, V_DIM, dtype=torch.float32)
        .reshape(1, 1, K_DIM, V_DIM)
        .expand(batch, num_heads, K_DIM, V_DIM)
        .contiguous()
    )
    h, v_new, final_state = torch.ops.ascend_ops.chunk_gated_delta_rule_fwd_h(
        k.npu(),
        w.npu(),
        u.npu(),
        g.npu(),
        initial_state=initial_state.npu(),
        output_final_state=True,
        chunk_size=CHUNK_SIZE,
    )
    expected_h = initial_state.to(dtype).unsqueeze(2)
    torch.testing.assert_close(h.cpu(), expected_h, rtol=0, atol=0)
    torch.testing.assert_close(v_new.cpu(), -w, rtol=0, atol=0)
    torch.testing.assert_close(final_state.cpu(), initial_state, rtol=0, atol=0)
