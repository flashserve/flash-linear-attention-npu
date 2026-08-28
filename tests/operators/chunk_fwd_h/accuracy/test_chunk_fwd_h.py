# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the BSD 3-Clause License (the "License").

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch_npu

from fla_npu.ops.ascendc import chunk_fwd_h


CHUNK_SIZE = 64
K_DIM = 128
V_DIM = 128
CASE_FILE = Path(__file__).resolve().parents[3] / "op_cases" / "chunk_fwd_h.json"
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _canonical_chunk_indices(seqlens: Sequence[int]) -> Tuple[int, ...]:
    result: List[int] = []
    for sequence, length in enumerate(seqlens):
        for chunk in range((length + CHUNK_SIZE - 1) // CHUNK_SIZE):
            result.extend((sequence, chunk))
    return tuple(result)


def _cu_seqlens(seqlens: Sequence[int]) -> Tuple[int, ...]:
    result = [0]
    for length in seqlens:
        result.append(result[-1] + length)
    return tuple(result)


def _gate_exp(value: torch.Tensor, use_exp2: bool) -> torch.Tensor:
    if use_exp2:
        return torch.exp(value * math.log(2.0))
    return torch.exp(value)


def _to_logical_state(state: torch.Tensor, state_v_first: bool) -> torch.Tensor:
    return state.transpose(-1, -2).contiguous() if state_v_first else state.contiguous()


def _to_physical_state(state: torch.Tensor, state_v_first: bool) -> torch.Tensor:
    return state.transpose(-1, -2).contiguous() if state_v_first else state.contiguous()


def _reference(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor],
    gk: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    use_exp2: bool,
    state_v_first: bool,
    seqlens: Optional[Sequence[int]],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    batch, _, total_tokens, _ = k.shape
    v_heads = u.shape[1]
    if seqlens is None:
        sequence_spans = [(batch_id, 0, total_tokens) for batch_id in range(batch)]
    else:
        cu = _cu_seqlens(seqlens)
        sequence_spans = [(0, cu[idx], cu[idx + 1]) for idx in range(len(seqlens))]

    total_chunks = sum((end - begin + CHUNK_SIZE - 1) // CHUNK_SIZE
                       for _, begin, end in sequence_spans)
    h_chunk_count = (
        total_chunks if seqlens is not None else (total_tokens + CHUNK_SIZE - 1) // CHUNK_SIZE
    )
    h_logical = torch.empty((batch, v_heads, h_chunk_count, K_DIM, V_DIM), dtype=torch.bfloat16)
    v_new = torch.empty_like(u)
    state_dtype = initial_state.dtype if initial_state is not None else torch.float32
    final_logical = torch.empty(
        (len(sequence_spans), v_heads, K_DIM, V_DIM), dtype=state_dtype
    ) if output_final_state else None
    initial_logical = (
        _to_logical_state(initial_state, state_v_first) if initial_state is not None else None
    )

    global_chunk = 0
    for sequence, (physical_batch, begin, end) in enumerate(sequence_spans):
        for hv in range(v_heads):
            state = (
                initial_logical[sequence, hv].clone()
                if initial_logical is not None
                else torch.zeros((K_DIM, V_DIM), dtype=state_dtype)
            )
            sequence_chunk = 0
            for token_begin in range(begin, end, CHUNK_SIZE):
                token_end = min(token_begin + CHUNK_SIZE, end)
                chunk_slot = global_chunk + sequence_chunk if seqlens is not None else sequence_chunk
                h_current = state.to(torch.bfloat16)
                h_logical[physical_batch, hv, chunk_slot] = h_current

                w_chunk = w[physical_batch, hv, token_begin:token_end]
                p_acc = w_chunk.float() @ h_current.float()
                p = p_acc.to(state_dtype)
                v_new_fp32 = u[physical_batch, hv, token_begin:token_end].float() - p.float()
                v_new_chunk = v_new_fp32.to(torch.bfloat16)
                v_new[physical_batch, hv, token_begin:token_end] = v_new_chunk

                is_last = token_end == end
                if not is_last or output_final_state:
                    if g is not None:
                        gate = g[physical_batch, hv, token_begin:token_end].float()
                        right = (
                            v_new_fp32 * _gate_exp(gate[-1] - gate, use_exp2).unsqueeze(-1)
                        ).to(torch.bfloat16)
                        group_size = v_heads // k.shape[1]
                        key_head = hv // group_size
                        left = k[physical_batch, key_head, token_begin:token_end]
                        decay = _gate_exp(gate[-1], use_exp2)
                        next_state = decay * state.float() + left.float().transpose(0, 1) @ right.float()
                    else:
                        gate = gk[physical_batch, hv, token_end - 1].float()
                        left = k[physical_batch, hv, token_begin:token_end]
                        next_state = (
                            _gate_exp(gate, use_exp2).unsqueeze(-1) * state.float()
                            + left.float().transpose(0, 1) @ v_new_chunk.float()
                        )
                    state = next_state.to(state_dtype)
                sequence_chunk += 1

            if final_logical is not None:
                final_logical[sequence, hv] = state
        if seqlens is not None:
            global_chunk += (end - begin + CHUNK_SIZE - 1) // CHUNK_SIZE

    return (
        _to_physical_state(h_logical, state_v_first),
        v_new,
        _to_physical_state(final_logical, state_v_first) if final_logical is not None else None,
    )


def _make_inputs(case: Dict) -> Tuple[Dict, Optional[Sequence[int]]]:
    torch.manual_seed(case["seed"])
    batch = case["batch"]
    seqlens = case.get("seqlens")
    total_tokens = sum(seqlens) if seqlens is not None else case["seqlen"]
    k_heads = case["k_heads"]
    v_heads = case["v_heads"]
    gate_dtype = DTYPES[case["gate_dtype"]]

    k = (torch.randn(batch, k_heads, total_tokens, K_DIM) * 0.05).to(torch.bfloat16)
    w = (torch.randn(batch, v_heads, total_tokens, K_DIM) * 0.05).to(torch.bfloat16)
    u = (torch.randn(batch, v_heads, total_tokens, V_DIM) * 0.05).to(torch.bfloat16)
    g = None
    gk = None
    if case["mode"] == "g":
        gate_step = -torch.rand(batch, v_heads, total_tokens) * 0.02
        g = torch.cumsum(gate_step, dim=2).to(gate_dtype)
    else:
        gate_step = -torch.rand(batch, v_heads, total_tokens, K_DIM) * 0.02
        gk = torch.cumsum(gate_step, dim=2).to(gate_dtype)

    initial_state = None
    if case["state_dtype"] is not None:
        sequences = len(seqlens) if seqlens is not None else batch
        logical_state = torch.randn(sequences, v_heads, K_DIM, V_DIM) * 0.05
        initial_state = _to_physical_state(
            logical_state.to(DTYPES[case["state_dtype"]]), case["state_v_first"]
        )
    return {
        "k": k,
        "w": w,
        "u": u,
        "g": g,
        "gk": gk,
        "initial_state": initial_state,
    }, seqlens


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    rtol, atol = (0.03, 0.03) if expected.dtype == torch.float32 else (0.02, 0.02)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=lambda msg: f"{name}: {msg}")


def run_case(case: Dict) -> None:
    inputs, seqlens = _make_inputs(case)
    expected = _reference(
        **inputs,
        output_final_state=case["output_final_state"],
        use_exp2=case["use_exp2"],
        state_v_first=case["state_v_first"],
        seqlens=seqlens,
    )
    cu = _cu_seqlens(seqlens) if seqlens is not None else None
    indices = (
        _canonical_chunk_indices(seqlens)
        if seqlens is not None and case["provide_chunk_indices"]
        else None
    )
    actual = chunk_fwd_h(
        inputs["k"].npu(),
        inputs["w"].npu(),
        inputs["u"].npu(),
        g=inputs["g"].npu() if inputs["g"] is not None else None,
        gk=inputs["gk"].npu() if inputs["gk"] is not None else None,
        initial_state=(
            inputs["initial_state"].npu() if inputs["initial_state"] is not None else None
        ),
        output_final_state=case["output_final_state"],
        chunk_size=CHUNK_SIZE,
        save_new_value=True,
        cu_seqlens=cu,
        chunk_indices=indices,
        use_exp2=case["use_exp2"],
        state_v_first=case["state_v_first"],
    )
    actual_cpu = tuple(value.cpu() if value is not None else None for value in actual)
    _assert_close(f"{case['id']}: h", actual_cpu[0], expected[0])
    _assert_close(f"{case['id']}: v_new", actual_cpu[1], expected[1])
    if case["output_final_state"]:
        if actual_cpu[2] is None or expected[2] is None:
            raise AssertionError(f"{case['id']}: final_state must be present")
        _assert_close(f"{case['id']}: final_state", actual_cpu[2], expected[2])
    elif actual_cpu[2] is not None:
        raise AssertionError(f"{case['id']}: final_state must be None")
    print(f"[PASS] {case['id']}")


def main() -> None:
    torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", "0")))
    data = json.loads(CASE_FILE.read_text(encoding="utf-8"))
    selected = {
        case_id.strip()
        for case_id in os.environ.get("CHUNK_FWD_H_CASE_IDS", "").split(",")
        if case_id.strip()
    }
    cases = [case for case in data["cases"] if not selected or case["id"] in selected]
    if selected and len(cases) != len(selected):
        found = {case["id"] for case in cases}
        raise ValueError(f"unknown case ids: {sorted(selected - found)}")
    for case in cases:
        run_case(case)


if __name__ == "__main__":
    main()
