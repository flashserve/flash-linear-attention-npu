"""Readable CPU golden for the target ``recurrent_gated_delta_rule`` contract.

The implementation intentionally follows the mathematical recurrence one
token at a time. It is not optimized: clarity and independence from the NPU
kernel are more important than CPU speed.

Both packed ``ssm_state_indices[T]`` and request-row
``ssm_state_indices[B, W]`` metadata are supported.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


_METADATA_DTYPE = torch.int32
QK_L2_NORM_EPS = 1.0e-6


def l2_normalize_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    eps: float = QK_L2_NORM_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """L2-normalize Q/K along ``Dk`` with the GDN epsilon convention.

    Results are cast back to the input dtype. The recurrent golden passes FP32
    tensors here so fused normalization stays in FP32, as required by the
    target kernel contract.
    """
    if not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError(f"eps must be finite and positive, got {eps}.")

    def _normalize(tensor: torch.Tensor) -> torch.Tensor:
        calc_dtype = (
            torch.float64 if tensor.dtype == torch.float64 else torch.float32
        )
        tensor_float = tensor.to(calc_dtype)
        inverse_norm = torch.rsqrt(
            tensor_float.square().sum(dim=-1, keepdim=True) + float(eps)
        )
        return (tensor_float * inverse_norm).to(tensor.dtype)

    return _normalize(query), _normalize(key)


def _require_cpu(name: str, tensor: torch.Tensor | None) -> None:
    if tensor is not None and tensor.device.type != "cpu":
        raise ValueError(
            f"recurrent_gated_delta_rule CPU golden requires {name} on CPU, "
            f"got device={tensor.device}."
        )


def _require_shape(
    name: str,
    tensor: torch.Tensor,
    expected: tuple[int, ...],
) -> None:
    actual = tuple(tensor.shape)
    if actual != expected:
        raise ValueError(f"{name} shape must be {expected}, got {actual}.")


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    actual_seq_lengths: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None,
    g: torch.Tensor | None,
    gk: torch.Tensor | None,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    use_gate_in_kernel: bool,
) -> tuple[list[int], list[int] | list[list[int]], list[int] | None]:
    """Validate structural metadata needed by the formula and return lists.

    This is deliberately not a replacement for the public API's complete
    host-side validation (for example, production-only dimension limits).
    """
    named_tensors = {
        "query": query,
        "key": key,
        "value": value,
        "state": state,
        "beta": beta,
        "actual_seq_lengths": actual_seq_lengths,
        "ssm_state_indices": ssm_state_indices,
        "num_accepted_tokens": num_accepted_tokens,
        "g": g,
        "gk": gk,
        "A_log": A_log,
        "dt_bias": dt_bias,
    }
    for name, tensor in named_tensors.items():
        _require_cpu(name, tensor)

    if query.ndim != 3:
        raise ValueError(f"query must be rank 3, got shape={tuple(query.shape)}.")
    total_tokens, key_heads, key_dim = query.shape
    _require_shape("key", key, (total_tokens, key_heads, key_dim))

    if value.ndim != 3:
        raise ValueError(f"value must be rank 3, got shape={tuple(value.shape)}.")
    value_tokens, value_heads, value_dim = value.shape
    if value_tokens != total_tokens:
        raise ValueError(
            f"query/value token counts must match, got {total_tokens} and "
            f"{value_tokens}."
        )
    if (
        key_heads <= 0
        or value_heads <= 0
        or value_heads % key_heads != 0
        or value_heads // key_heads > 16
    ):
        raise ValueError(
            "value head count must be 1-16 times the key head "
            f"count, got Nk={key_heads}, Nv={value_heads}."
        )

    if state.ndim != 4:
        raise ValueError(f"state must be rank 4, got shape={tuple(state.shape)}.")
    state_blocks = state.shape[0]
    _require_shape(
        "state",
        state,
        (state_blocks, value_heads, value_dim, key_dim),
    )
    if state_blocks <= 0:
        raise ValueError("state must contain at least one State pool slot.")
    _require_shape("beta", beta, (total_tokens, value_heads))
    if g is not None:
        _require_shape("g", g, (total_tokens, value_heads))
    if gk is not None:
        raise ValueError("gk is reserved; only scalar gate g is supported.")
    if A_log is not None:
        _require_shape("A_log", A_log, (value_heads,))
    if dt_bias is not None:
        _require_shape("dt_bias", dt_bias, (value_heads,))
    if use_gate_in_kernel:
        if g is None:
            raise ValueError(
                "g is required when use_gate_in_kernel=True because g is "
                "the raw scalar gate input."
            )
        if A_log is None:
            raise ValueError("A_log is required when use_gate_in_kernel=True.")
    if not math.isfinite(float(scale)):
        raise ValueError(f"scale must be finite, got {scale}.")

    if actual_seq_lengths.ndim != 1 or actual_seq_lengths.numel() < 2:
        raise ValueError(
            "actual_seq_lengths must be [prefix, length_0, ..., length_B-1]."
        )
    if actual_seq_lengths.dtype != _METADATA_DTYPE:
        raise TypeError(
            "actual_seq_lengths must use INT32, got "
            f"{actual_seq_lengths.dtype}."
        )
    lengths = [int(item) for item in actual_seq_lengths.tolist()]
    prefix_tokens = lengths[0]
    seq_lengths = lengths[1:]
    if prefix_tokens < 0 or any(length < 0 for length in seq_lengths):
        raise ValueError(f"sequence lengths must be non-negative, got {lengths}.")
    if sum(lengths) != total_tokens:
        raise ValueError(
            "actual_seq_lengths must sum to T, got "
            f"sum={sum(lengths)} and T={total_tokens}."
        )

    if ssm_state_indices.dtype != _METADATA_DTYPE:
        raise TypeError(
            "ssm_state_indices must use INT32, got "
            f"{ssm_state_indices.dtype}."
        )
    batch_size = len(seq_lengths)
    if ssm_state_indices.ndim == 1:
        _require_shape("ssm_state_indices", ssm_state_indices, (total_tokens,))
        state_widths = seq_lengths
    elif ssm_state_indices.ndim == 2:
        if ssm_state_indices.shape[0] != batch_size:
            raise ValueError(
                "2-D ssm_state_indices row count must equal B, got "
                f"shape={tuple(ssm_state_indices.shape)} and B={batch_size}."
            )
        state_width = int(ssm_state_indices.shape[1])
        if state_width <= 0:
            raise ValueError("2-D ssm_state_indices must have a positive W.")
        if any(length > state_width for length in seq_lengths):
            raise ValueError(
                f"every sequence length must be <= W={state_width}, got "
                f"{seq_lengths}."
            )
        state_widths = [state_width] * batch_size
    else:
        raise ValueError(
            "ssm_state_indices must be rank 1 [T] or rank 2 [B, W], got "
            f"shape={tuple(ssm_state_indices.shape)}."
        )

    flat_indices = ssm_state_indices.reshape(-1).to(torch.int64)
    if torch.any(flat_indices < 0) or torch.any(flat_indices >= state_blocks):
        raise ValueError(
            "ssm_state_indices values must be in [0, BlockNum), got "
            f"BlockNum={state_blocks}."
        )
    if ssm_state_indices.ndim == 1:
        if total_tokens > state_blocks:
            raise ValueError(
                f"1-D ssm_state_indices requires T <= BlockNum, got T={total_tokens} "
                f"and BlockNum={state_blocks}."
            )
    elif ssm_state_indices.numel() > state_blocks:
        raise ValueError(
            "2-D ssm_state_indices requires B*W <= BlockNum, got "
            f"B*W={ssm_state_indices.numel()} and BlockNum={state_blocks}."
        )
    if torch.unique(flat_indices).numel() != flat_indices.numel():
        raise ValueError("ssm_state_indices values must be globally unique.")

    accepted: list[int] | None = None
    if num_accepted_tokens is not None:
        if num_accepted_tokens.dtype != _METADATA_DTYPE:
            raise TypeError(
                "num_accepted_tokens must use INT32, got "
                f"{num_accepted_tokens.dtype}."
            )
        _require_shape(
            "num_accepted_tokens",
            num_accepted_tokens,
            (batch_size,),
        )
        accepted = [int(item) for item in num_accepted_tokens.tolist()]
        for batch_index, seq_len in enumerate(seq_lengths):
            # Empty rows are skipped before the kernel reads accepted[b].
            if seq_len == 0:
                continue
            if not 1 <= accepted[batch_index] <= state_widths[batch_index]:
                raise ValueError(
                    f"num_accepted_tokens[{batch_index}] must be in "
                    f"[1, {state_widths[batch_index]}], got "
                    f"{accepted[batch_index]}."
                )

    state_indices = ssm_state_indices.tolist()
    return lengths, state_indices, accepted


def _state_slot(
    state_indices: list[int] | list[list[int]],
    *,
    table_mode: bool,
    batch_index: int,
    sequence_start: int,
    relative_token: int,
) -> int:
    """Map a request-local checkpoint/token position to a State pool slot."""
    if table_mode:
        return int(state_indices[batch_index][relative_token])
    return int(state_indices[sequence_start + relative_token])


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
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the RecurrentGDN recurrence on CPU.

    For every value head and token, the implementation computes::

        q_eff, k_eff = q, k
        if use_qk_l2norm_in_kernel:
            q_eff = q / sqrt(sum(q * q) + 1e-6)
            k_eff = k / sqrt(sum(k * k) + 1e-6)

        g_eff = g
        if use_gate_in_kernel:
            g_eff = -exp(A_log) * softplus(g + dt_bias)

        beta_eff = sigmoid(beta) if use_beta_sigmoid_in_kernel else beta
        if use_beta_sigmoid_in_kernel and allow_neg_eigval:
            beta_eff = 2 * beta_eff
        decayed_state = exp(g_eff) * state if g is present else state
        delta = beta_eff * (value - decayed_state @ k_eff)
        state = decayed_state + outer(delta, k_eff)
        out = state @ (scale * q_eff)

    ``state`` is treated as a State pool. The returned ``final_state`` is a
    clone containing all per-token writebacks; the input ``state`` is not
    mutated. Arithmetic follows the operator's FP32 recurrence semantics and
    is cast to the public output/state dtypes only when returned.
    """
    lengths, state_indices, accepted_tokens = _validate_inputs(
        query,
        key,
        value,
        state,
        beta,
        scale,
        actual_seq_lengths,
        ssm_state_indices,
        num_accepted_tokens,
        g,
        gk,
        A_log,
        dt_bias,
        use_gate_in_kernel,
    )
    calc_dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    if use_qk_l2norm_in_kernel:
        # Match the fused recurrent kernel: Q/K are loaded into the recurrence
        # compute dtype and normalized there, without a low-precision cast in
        # between normalization and the recurrent update.
        effective_query, effective_key = l2_normalize_qk(
            query.to(calc_dtype),
            key.to(calc_dtype),
        )
    else:
        effective_query, effective_key = query, key

    effective_g = g
    if use_gate_in_kernel:
        # Validation above guarantees both tensors are present. A_log is
        # head-wise and broadcasts across the packed token dimension.
        assert effective_g is not None
        assert A_log is not None
        gate_input = effective_g.to(calc_dtype)
        if dt_bias is not None:
            gate_input = gate_input + dt_bias.to(calc_dtype).reshape(1, -1)
        gate_scale = torch.exp(A_log.to(calc_dtype)).reshape(1, -1)
        effective_g = -gate_scale * F.softplus(gate_input)

    effective_beta = beta.to(calc_dtype)
    if use_beta_sigmoid_in_kernel:
        effective_beta = torch.sigmoid(effective_beta)
        if allow_neg_eigval:
            effective_beta = effective_beta * 2.0

    final_state = state.to(calc_dtype).clone()
    total_tokens, key_heads, _ = query.shape
    value_heads, value_dim = value.shape[1:]
    value_heads_per_key_head = value_heads // key_heads
    out = torch.zeros(
        (total_tokens, value_heads, value_dim),
        dtype=calc_dtype,
        device="cpu",
    )

    table_mode = ssm_state_indices.ndim == 2
    sequence_start = lengths[0]
    for batch_index, sequence_length in enumerate(lengths[1:]):
        sequence_end = sequence_start + sequence_length
        if sequence_length == 0:
            sequence_start = sequence_end
            continue

        accepted_count = (
            1 if accepted_tokens is None else accepted_tokens[batch_index]
        )
        initial_state_slot = _state_slot(
            state_indices,
            table_mode=table_mode,
            batch_index=batch_index,
            sequence_start=sequence_start,
            relative_token=accepted_count - 1,
        )

        for value_head in range(value_heads):
            key_head = value_head // value_heads_per_key_head
            recurrent_state = final_state[initial_state_slot, value_head].clone()

            for token_index in range(sequence_start, sequence_end):
                relative_token = token_index - sequence_start

                if effective_g is not None:
                    recurrent_state *= torch.exp(
                        effective_g[token_index, value_head].to(calc_dtype)
                    )
                key_vector = effective_key[token_index, key_head].to(calc_dtype)
                prediction = torch.matmul(recurrent_state, key_vector)
                delta = value[token_index, value_head].to(calc_dtype) - prediction
                delta *= effective_beta[token_index, value_head]
                recurrent_state += torch.outer(delta, key_vector)

                scaled_query = (
                    effective_query[token_index, key_head].to(calc_dtype)
                    * float(scale)
                )
                out[token_index, value_head] = torch.matmul(
                    recurrent_state,
                    scaled_query,
                )

                writeback_slot = _state_slot(
                    state_indices,
                    table_mode=table_mode,
                    batch_index=batch_index,
                    sequence_start=sequence_start,
                    relative_token=relative_token,
                )
                final_state[writeback_slot, value_head] = recurrent_state

        sequence_start = sequence_end

    return out.to(value.dtype), final_state.to(state.dtype)


# A beginner-friendly name for direct imports; the ATK executor keeps using the
# historical ``*_reference`` symbol above.
recurrent_gated_delta_rule_cpu_golden = recurrent_gated_delta_rule_reference
