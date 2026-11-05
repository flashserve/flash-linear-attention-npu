# Copyright (c) 2026 Tianjin University, Ltd.
"""Phase A scaffold. CPU zeros describe the ABI only, never numerical accuracy."""

import torch


def chunk_kda_bwd_finalize_cpu(q, k, v, gk, raw_g, beta, a_log, dt_bias, akk,
                             v_new, h, dh, dv_scan, d_aqk, dq_raw, *, scale,
                             lower_bound=-5.0, chunk_size=64, safe_gate=True,
                             use_gate_in_kernel=True, use_exp2=True,
                             state_v_first=False, q_rstd=None, k_rstd=None,
                             cu_seqlens=None, chunk_indices=None,
                             use_beta_sigmoid_in_kernel=False,
                             allow_neg_eigval=False, output_layout=None):
    """ABI placeholder. d_beta and d_g ALWAYS use token-major layout."""
    rank3 = q.ndim == 3
    output_layout = output_layout or ("NTD" if rank3 else "BNSD")
    token_shape = list(q.shape)
    first = 0 if rank3 else 1
    token_shape[first], token_shape[first + 1] = token_shape[first + 1], token_shape[first]
    qkv_shape = token_shape if output_layout in ("TND", "BSND") else list(q.shape)
    return (torch.zeros(qkv_shape, dtype=torch.bfloat16),
            torch.zeros(qkv_shape, dtype=torch.bfloat16),
            torch.zeros(qkv_shape, dtype=torch.bfloat16),
            torch.zeros(token_shape[:-1], dtype=beta.dtype),
            torch.zeros(token_shape, dtype=torch.float32),
            torch.zeros(a_log.shape, dtype=torch.float32),
            torch.zeros(dt_bias.shape, dtype=torch.float32))
