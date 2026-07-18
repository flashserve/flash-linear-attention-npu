"""
CPU reference (golden) implementation for recurrent_gated_delta_rule.

Mathematical formula:
  S_t := alpha_t * S_{t-1} + beta_t * (v_t - alpha_t * S_{t-1} @ k_t) @ k_t^T
  o_t := S_t @ q_t * scale

Where alpha_t = exp(g_t).
"""
import torch
from typing import Optional, Tuple


def recurrent_gated_delta_rule_golden(
    query: torch.Tensor,             # [T, NK, dk]  bf16
    key: torch.Tensor,               # [T, NK, dk]  bf16
    value: torch.Tensor,             # [T, NV, dv]  bf16
    state: torch.Tensor,             # [T, NV, dv, dk]  bf16
    beta: torch.Tensor,              # [T, NV]  bf16
    scale: float,
    actual_seq_lengths: torch.Tensor, # [B]  int32
    ssm_state_indices: torch.Tensor,  # [T]  int32
    num_accepted_tokens: Optional[torch.Tensor] = None,  # [B] int32
    g: Optional[torch.Tensor] = None,   # [T, NV] fp32
    gk: Optional[torch.Tensor] = None,  # [T, NK, dk] fp32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        attn_out:    [T, NV, dv]     bf16
        final_state: [T, NV, dv, dk] bf16
    """
    T, NK, dk = query.shape
    _, NV, dv = value.shape
    B = actual_seq_lengths.shape[0]

    # Cast all to float32 for precision
    q_f = query.float() * scale
    k_f = key.float()
    v_f = value.float()
    beta_f = beta.float()
    state_f = state.float()

    alpha = torch.exp(g.float()) if g is not None else None
    alpha_k = torch.exp(gk.float()) if gk is not None else None

    attn_out = torch.zeros(T, NV, dv, dtype=torch.float32)
    final_state = state_f.clone()

    seq_ptr = int(actual_seq_lengths[0].item())
    for b in range(1, B):
        seq_len = int(actual_seq_lengths[b].item())
        seq0 = seq_ptr
        seq1 = seq0 + seq_len
        seq_ptr = seq1

        # Determine which state position to load as initial state
        state_token_idx = seq0
        if num_accepted_tokens is not None:
            accepted = int(num_accepted_tokens[b-1].item())
            state_token_idx = seq0 + accepted - 1

        state_offset = int(ssm_state_indices[state_token_idx].item())

        for h_v in range(NV):
            h_k = h_v // (NV // NK)

            # Load initial state for this head: [dv, dk]
            S = final_state[state_offset, h_v].clone()

            for seq_i in range(seq0, seq1):
                q_h = q_f[seq_i, h_k]       # [dk]
                k_h = k_f[seq_i, h_k]       # [dk]
                v_h = v_f[seq_i, h_v]       # [dv]
                b_h = beta_f[seq_i, h_v]    # scalar

                # Apply gama decay
                if alpha is not None:
                    S = S * alpha[seq_i, h_v]

                # Apply gamaK decay (element-wise along dk dimension)
                if alpha_k is not None:
                    S = S * alpha_k[seq_i, h_v].unsqueeze(0)

                # delta = v - S @ k
                Sk = torch.mv(S, k_h)       # [dv]
                delta = v_h - Sk
                delta = delta * b_h

                # Rank-1 update: S += delta * k^T
                S = S + torch.outer(delta, k_h)

                # Attention output: o = S @ q_scaled
                attn = torch.mv(S, q_h)     # [dv]

                # Store
                attn_out[seq_i, h_v] = attn
                final_state[int(ssm_state_indices[seq_i].item()), h_v] = S

    return attn_out.to(torch.bfloat16), final_state.to(torch.bfloat16)
