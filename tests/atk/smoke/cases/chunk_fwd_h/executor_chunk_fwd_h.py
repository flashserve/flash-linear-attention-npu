"""CPU reference from FLAN 05b530df (tests/atk/chunk_fwd_h/executor_chunk_fwd_h.py)."""

from dataclasses import dataclass
import math
from typing import Optional

import torch


CHUNK_SIZE = 64


K_DIM = 128


V_DIM = 128


def _to_logical_state(state: torch.Tensor, state_v_first: bool) -> torch.Tensor:
    return state.transpose(-1, -2).contiguous() if state_v_first else state.contiguous()


def _to_physical_state(state: torch.Tensor, state_v_first: bool) -> torch.Tensor:
    return state.transpose(-1, -2).contiguous() if state_v_first else state.contiguous()


def _gate_exp(value: torch.Tensor, use_exp2: bool) -> torch.Tensor:
    return torch.exp(value * math.log(2.0)) if use_exp2 else torch.exp(value)


@dataclass
class PreparedInputs:
    k: torch.Tensor
    w: torch.Tensor
    u: torch.Tensor
    g: Optional[torch.Tensor]
    gk: Optional[torch.Tensor]
    initial_state: Optional[torch.Tensor]
    cu_seqlens: Optional[tuple[int, ...]]
    chunk_indices: Optional[tuple[int, ...]]
    seqlens: Optional[list[int]]


def _sequence_spans(inputs: PreparedInputs) -> list[tuple[int, int, int]]:
    batch, _, total_tokens, _ = inputs.k.shape
    if inputs.seqlens is None:
        return [(batch_id, 0, total_tokens) for batch_id in range(batch)]
    cu = inputs.cu_seqlens
    assert cu is not None
    return [(0, cu[index], cu[index + 1]) for index in range(len(cu) - 1)]


def _reference(
    inputs: PreparedInputs,
    *,
    output_final_state: bool,
    use_exp2: bool,
    state_v_first: bool,
    high_precision: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    k, w, u, g, gk = inputs.k, inputs.w, inputs.u, inputs.g, inputs.gk
    batch, _, total_tokens, _ = k.shape
    v_heads = u.shape[1]
    sequence_spans = _sequence_spans(inputs)
    total_chunks = sum(
        (end - begin + CHUNK_SIZE - 1) // CHUNK_SIZE
        for _, begin, end in sequence_spans
    )
    dense_chunks = (total_tokens + CHUNK_SIZE - 1) // CHUNK_SIZE
    h_chunk_count = total_chunks if inputs.seqlens is not None else dense_chunks

    elem_dtype = torch.float64 if high_precision else torch.bfloat16
    calc_dtype = torch.float64 if high_precision else torch.float32
    h_logical = torch.empty(
        (batch, h_chunk_count, v_heads, K_DIM, V_DIM),
        dtype=elem_dtype,
        device=k.device,
    )
    v_new = torch.empty(tuple(u.shape), dtype=elem_dtype, device=u.device)
    state_dtype = (
        inputs.initial_state.dtype
        if inputs.initial_state is not None
        else calc_dtype
    )
    final_logical = (
        torch.empty(
            (len(sequence_spans), v_heads, K_DIM, V_DIM),
            dtype=state_dtype,
            device=k.device,
        )
        if output_final_state
        else None
    )
    initial_logical = (
        _to_logical_state(inputs.initial_state, state_v_first)
        if inputs.initial_state is not None
        else None
    )

    global_chunk = 0
    for sequence, (physical_batch, sequence_begin, sequence_end) in enumerate(
        sequence_spans
    ):
        for value_head in range(v_heads):
            state = (
                initial_logical[sequence, value_head].clone()
                if initial_logical is not None
                else torch.zeros(
                    (K_DIM, V_DIM), dtype=state_dtype, device=k.device
                )
            )
            sequence_chunk = 0
            for token_begin in range(sequence_begin, sequence_end, CHUNK_SIZE):
                token_end = min(token_begin + CHUNK_SIZE, sequence_end)
                chunk_slot = (
                    global_chunk + sequence_chunk
                    if inputs.seqlens is not None
                    else sequence_chunk
                )
                h_current = state.to(elem_dtype)
                h_logical[physical_batch, chunk_slot, value_head] = h_current

                w_chunk = w[physical_batch, value_head, token_begin:token_end]
                p_acc = w_chunk.to(calc_dtype) @ h_current.to(calc_dtype)
                p = p_acc.to(state_dtype)
                v_new_fp32 = (
                    u[physical_batch, value_head, token_begin:token_end].to(calc_dtype)
                    - p.to(calc_dtype)
                )
                v_new_chunk = v_new_fp32.to(elem_dtype)
                v_new[physical_batch, value_head, token_begin:token_end] = v_new_chunk

                is_last = token_end == sequence_end
                if not is_last or output_final_state:
                    if g is not None:
                        gate = g[
                            physical_batch, value_head, token_begin:token_end
                        ].to(calc_dtype)
                        right = (
                            v_new_fp32
                            * _gate_exp(gate[-1] - gate, use_exp2).unsqueeze(-1)
                        ).to(elem_dtype)
                        group_size = v_heads // k.shape[1]
                        key_head = value_head // group_size
                        left = k[
                            physical_batch, key_head, token_begin:token_end
                        ]
                        next_state = (
                            _gate_exp(gate[-1], use_exp2) * state.to(calc_dtype)
                            + left.to(calc_dtype).transpose(0, 1) @ right.to(calc_dtype)
                        )
                    else:
                        assert gk is not None
                        gate = gk[
                            physical_batch, value_head, token_end - 1
                        ].to(calc_dtype)
                        left = k[
                            physical_batch, value_head, token_begin:token_end
                        ]
                        next_state = (
                            _gate_exp(gate, use_exp2).unsqueeze(-1) * state.to(calc_dtype)
                            + left.to(calc_dtype).transpose(0, 1) @ v_new_chunk.to(calc_dtype)
                        )
                    state = next_state.to(state_dtype)
                sequence_chunk += 1

            if final_logical is not None:
                final_logical[sequence, value_head] = state
        if inputs.seqlens is not None:
            global_chunk += (
                sequence_end - sequence_begin + CHUNK_SIZE - 1
            ) // CHUNK_SIZE

    return (
        _to_physical_state(h_logical, state_v_first),
        v_new,
        _to_physical_state(final_logical, state_v_first)
        if final_logical is not None
        else None,
    )
