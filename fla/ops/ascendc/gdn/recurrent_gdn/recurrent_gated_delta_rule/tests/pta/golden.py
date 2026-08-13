"""Small-operator reference for ``recurrent_gated_delta_rule``.

The reference works on either CPU or GPU.  Callers can select float64 for a
high-precision CUDA golden while the default keeps the historical float32 to
bfloat16 behavior used by ``test_accuracy.py``.
"""

from typing import Optional, Tuple

import torch


def recurrent_gated_delta_rule_golden(
    query: torch.Tensor,              # [T, NK, Dk]
    key: torch.Tensor,                # [T, NK, Dk]
    value: torch.Tensor,              # [T, NV, Dv]
    state: torch.Tensor,              # [BlockNum, NV, Dv, Dk]
    beta: torch.Tensor,               # [T, NV]
    scale: float,
    actual_seq_lengths: torch.Tensor, # [B + 1]
    ssm_state_indices: torch.Tensor,  # [T]
    num_accepted_tokens: Optional[torch.Tensor] = None,  # [B]
    g: Optional[torch.Tensor] = None,   # [T, NV]
    gk: Optional[torch.Tensor] = None,  # [T, NV, Dk]
    *,
    compute_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the recurrence with PyTorch operators on ``query.device``.

    ``actual_seq_lengths[0]`` is the invalid prefix length.  Each remaining
    entry is a sequence length.  A sequence loads its initial state through
    ``ssm_state_indices`` at either its first token or the token selected by
    ``num_accepted_tokens``.  Every computed token writes its state back to
    the physical block selected by ``ssm_state_indices``.
    """

    total_tokens, key_heads, key_dim = query.shape
    _, value_heads, value_dim = value.shape
    if value_heads % key_heads:
        raise ValueError("value heads must be divisible by key heads")
    if compute_dtype not in (torch.float32, torch.float64):
        raise ValueError("compute_dtype must be torch.float32 or torch.float64")

    device = query.device
    q = query.to(device=device, dtype=compute_dtype) * scale
    k = key.to(device=device, dtype=compute_dtype)
    v = value.to(device=device, dtype=compute_dtype)
    beta_f = beta.to(device=device, dtype=compute_dtype)
    final_state = state.to(device=device, dtype=compute_dtype).clone()
    alpha = (
        torch.exp(g.to(device=device, dtype=compute_dtype))
        if g is not None
        else None
    )
    alpha_k = (
        torch.exp(gk.to(device=device, dtype=compute_dtype))
        if gk is not None
        else None
    )

    output = torch.zeros(
        (total_tokens, value_heads, value_dim),
        dtype=compute_dtype,
        device=device,
    )
    head_map = torch.arange(value_heads, device=device) // (
        value_heads // key_heads
    )
    q = q[:, head_map]
    k = k[:, head_map]

    lengths = actual_seq_lengths.detach().cpu().to(torch.int64).tolist()
    state_indices = ssm_state_indices.detach().cpu().to(torch.int64).tolist()
    accepted_tokens = (
        num_accepted_tokens.detach().cpu().to(torch.int64).tolist()
        if num_accepted_tokens is not None
        else None
    )

    sequence_start = int(lengths[0])
    for batch, sequence_length in enumerate(lengths[1:]):
        sequence_end = sequence_start + int(sequence_length)
        if sequence_start == sequence_end:
            continue

        initial_token = sequence_start
        if accepted_tokens is not None:
            initial_token += int(accepted_tokens[batch]) - 1
        recurrent_state = final_state[state_indices[initial_token]].clone()

        for token in range(sequence_start, sequence_end):
            if alpha is not None:
                recurrent_state = recurrent_state * alpha[token, :, None, None]
            if alpha_k is not None:
                recurrent_state = recurrent_state * alpha_k[token, :, None, :]

            state_key = torch.bmm(
                recurrent_state, k[token, :, :, None]
            ).squeeze(-1)
            delta = beta_f[token, :, None] * (v[token] - state_key)
            recurrent_state = recurrent_state + delta[:, :, None] * k[token, :, None, :]
            output[token] = torch.bmm(
                recurrent_state, q[token, :, :, None]
            ).squeeze(-1)
            final_state[state_indices[token]] = recurrent_state

        sequence_start = sequence_end

    return output.to(output_dtype), final_state.to(output_dtype)
