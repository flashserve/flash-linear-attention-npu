"""Pure PyTorch reference for recurrent_gated_delta_rule."""

from __future__ import annotations

import torch


def recurrent_gated_delta_rule_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    actual_seq_lengths: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None = None,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the recurrent update using the operator's FP32 semantics."""
    actual_seq_lengths_list = actual_seq_lengths.detach().cpu().tolist()
    state_indices = ssm_state_indices.detach().cpu().tolist()
    accepted_tokens: list[int] | None = None
    if num_accepted_tokens is not None:
        accepted_tokens = num_accepted_tokens.detach().cpu().tolist()

    calc_dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    final_state = state.to(calc_dtype).clone()
    total_tokens, key_heads, _ = query.shape
    value_heads, value_dim = value.shape[1:]
    head_group = value_heads // key_heads
    out = torch.zeros(
        (total_tokens, value_heads, value_dim),
        dtype=calc_dtype,
        device=query.device,
    )

    seq_start = int(actual_seq_lengths_list[0])
    for batch_index, seq_len_value in enumerate(actual_seq_lengths_list[1:]):
        seq_len = int(seq_len_value)
        if seq_len <= 0:
            continue
        seq_end = seq_start + seq_len
        state_token_index = seq_start
        if accepted_tokens is not None:
            state_token_index += int(accepted_tokens[batch_index]) - 1
        initial_state_slot = int(state_indices[state_token_index])

        for value_head in range(value_heads):
            key_head = value_head // head_group
            recurrent_state = final_state[initial_state_slot, value_head].clone()
            for token_index in range(seq_start, seq_end):
                if g is not None:
                    recurrent_state *= torch.exp(
                        g[token_index, value_head].to(calc_dtype)
                    )
                if gk is not None:
                    recurrent_state *= torch.exp(
                        gk[token_index, value_head].to(calc_dtype)
                    ).unsqueeze(0)

                key_vector = key[token_index, key_head].to(calc_dtype)
                delta = value[token_index, value_head].to(calc_dtype)
                delta -= torch.matmul(recurrent_state, key_vector)
                delta *= beta[token_index, value_head].to(calc_dtype)
                recurrent_state += torch.outer(delta, key_vector)
                scaled_query = query[token_index, key_head].to(calc_dtype) * float(
                    scale
                )
                out[token_index, value_head] = torch.matmul(
                    recurrent_state, scaled_query
                )
                state_slot = int(state_indices[token_index])
                final_state[state_slot, value_head] = recurrent_state

        seq_start = seq_end

    return out.to(value.dtype), final_state.to(state.dtype)
