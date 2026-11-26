"""CPU portion of FLA_ATK/recurrent_gated_delta_rule/executor_recurrent_gated_delta_rule.py
at 72d372032214b8866cd6073431b0d478e7744ba5. Only the model FP64 path is used.
"""
from typing import Any

import torch


REQUIRED_TENSORS = (
    "query",
    "key",
    "value",
    "state",
    "beta",
    "actual_seq_lengths",
    "ssm_state_indices",
)
OPTIONAL_TENSORS = ("num_accepted_tokens", "g", "gk")


def _optional_tensor(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if torch.is_tensor(value) and value.numel() == 0:
        return None
    if not torch.is_tensor(value):
        raise TypeError(f"optional operator input must be a tensor or None, got {type(value)!r}")
    return value


def _convert_tensor(value: Any, *, high_precision: bool) -> torch.Tensor | None:
    tensor = _optional_tensor(value)
    if tensor is not None and high_precision and tensor.is_floating_point():
        tensor = tensor.to(torch.float64)
    return tensor


def _validate_inputs(inputs: dict[str, Any]) -> None:
    query = inputs["query"]
    key = inputs["key"]
    value = inputs["value"]
    state = inputs["state"]
    beta = inputs["beta"]
    actual_seq_lengths = inputs["actual_seq_lengths"]
    state_indices = inputs["ssm_state_indices"]
    accepted_tokens = inputs["num_accepted_tokens"]
    g = inputs["g"]
    gk = inputs["gk"]

    if query.ndim != 3 or key.shape != query.shape:
        raise ValueError("query and key must have the same rank-3 shape")
    total_tokens, key_heads, key_dim = query.shape
    if value.ndim != 3 or value.shape[0] != total_tokens:
        raise ValueError("value must be rank 3 and share query's token dimension")
    value_heads, value_dim = value.shape[1:]
    if beta.shape != (total_tokens, value_heads):
        raise ValueError("beta shape must be (T, value_heads)")
    if state.ndim != 4 or tuple(state.shape[1:]) != (
        value_heads,
        value_dim,
        key_dim,
    ):
        raise ValueError("state shape must be (block_num, value_heads, value_dim, key_dim)")
    if key_dim != 128 or value_dim != 128:
        raise ValueError("key_dim and value_dim must both equal 128")
    if key_heads <= 0 or value_heads % key_heads:
        raise ValueError("value_heads must be divisible by key_heads")
    if actual_seq_lengths.ndim != 1 or actual_seq_lengths.numel() < 2:
        raise ValueError("actual_seq_lengths must be a non-empty rank-1 tensor")
    sequence_lengths = actual_seq_lengths.detach().cpu().tolist()
    if any(int(length) < 0 for length in sequence_lengths):
        raise ValueError("actual_seq_lengths must be non-negative")
    if sum(int(length) for length in sequence_lengths) != total_tokens:
        raise ValueError("actual_seq_lengths must sum to the query token count")
    if state_indices.shape != (total_tokens,):
        raise ValueError("ssm_state_indices shape must be (T,)")
    if state_indices.numel():
        minimum = int(state_indices.min().item())
        maximum = int(state_indices.max().item())
        if minimum < 0 or maximum >= state.shape[0]:
            raise ValueError("ssm_state_indices contains an out-of-range state slot")
    if accepted_tokens is not None:
        batch_size = actual_seq_lengths.numel() - 1
        if accepted_tokens.shape != (batch_size,):
            raise ValueError("num_accepted_tokens shape must be (B,)")
        accepted = accepted_tokens.detach().cpu().tolist()
        valid_lengths = sequence_lengths[1:]
        if any(
            int(count) < 1 or int(count) > int(length)
            for count, length in zip(accepted, valid_lengths)
        ):
            raise ValueError("num_accepted_tokens is outside its sequence length")
    if g is None and gk is None:
        raise ValueError("at least one of g or gk must be provided")
    if g is not None and g.shape != (total_tokens, value_heads):
        raise ValueError("g shape must be (T, value_heads)")
    if gk is not None and gk.shape != (total_tokens, value_heads, key_dim):
        raise ValueError("gk shape must be (T, value_heads, key_dim)")


def _prepare_inputs(values: dict[str, Any], *, high_precision: bool) -> dict[str, Any]:
    missing = [name for name in REQUIRED_TENSORS if name not in values]
    if missing:
        raise ValueError(f"missing direct operator inputs: {missing}")
    inputs = {
        name: _convert_tensor(values[name], high_precision=high_precision)
        for name in REQUIRED_TENSORS
    }
    inputs.update(
        {
            name: _convert_tensor(values.get(name), high_precision=high_precision)
            for name in OPTIONAL_TENSORS
        }
    )
    inputs["scale"] = float(values.get("scale", 1.0))
    _validate_inputs(inputs)
    return inputs


def _run_fp64_composed(inputs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        actual_seq_lengths = inputs["actual_seq_lengths"].detach().cpu().tolist()
        state_indices = inputs["ssm_state_indices"].detach().cpu().tolist()
        accepted_tensor = inputs["num_accepted_tokens"]
        accepted_tokens = (
            None if accepted_tensor is None else accepted_tensor.detach().cpu().tolist()
        )

        query = inputs["query"]
        key = inputs["key"]
        value = inputs["value"]
        beta = inputs["beta"]
        g = inputs["g"]
        gk = inputs["gk"]
        final_state = inputs["state"].clone()
        total_tokens, key_heads, _ = query.shape
        value_heads, value_dim = value.shape[1:]
        head_group = value_heads // key_heads
        out = torch.zeros(
            (total_tokens, value_heads, value_dim),
            dtype=query.dtype,
            device=query.device,
        )

        seq_start = int(actual_seq_lengths[0])
        for batch_index, seq_len_value in enumerate(actual_seq_lengths[1:]):
            seq_len = int(seq_len_value)
            if seq_len <= 0:
                continue
            seq_end = seq_start + seq_len
            state_token = seq_start
            if accepted_tokens is not None:
                state_token += int(accepted_tokens[batch_index]) - 1
            initial_slot = int(state_indices[state_token])

            for value_head in range(value_heads):
                key_head = value_head // head_group
                recurrent_state = final_state[initial_slot, value_head].clone()
                for token_index in range(seq_start, seq_end):
                    if g is not None:
                        recurrent_state *= torch.exp(g[token_index, value_head])
                    if gk is not None:
                        recurrent_state *= torch.exp(
                            gk[token_index, value_head]
                        ).unsqueeze(0)
                    key_vector = key[token_index, key_head]
                    delta = value[token_index, value_head]
                    delta = delta - torch.matmul(recurrent_state, key_vector)
                    delta = delta * beta[token_index, value_head]
                    recurrent_state += torch.outer(delta, key_vector)
                    scaled_query = query[token_index, key_head] * inputs["scale"]
                    out[token_index, value_head] = torch.matmul(
                        recurrent_state, scaled_query
                    )
                    final_state[int(state_indices[token_index]), value_head] = (
                        recurrent_state
                    )
            seq_start = seq_end
        return out, final_state
    finally:
        torch.set_num_threads(previous_threads)
