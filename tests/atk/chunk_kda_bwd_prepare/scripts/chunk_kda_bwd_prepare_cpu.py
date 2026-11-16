# Copyright (c) 2026 Tianjin University, Ltd.
"""Phase A scaffold. CPU zeros describe the ABI only, never numerical accuracy."""

import torch


def chunk_kda_bwd_prepare_cpu(aqk, v_new, d_o, h, *, scale, layout,
                            chunk_size=64, state_v_first=False,
                            cu_seqlens=None, chunk_indices=None):
    """ABI placeholder: head-major (FP32 d_aqk, BF16 dv, FP32 dq_raw)."""
    if layout == "BNSD":
        b, heads, t, _ = aqk.shape
        leading = (b, heads, t)
    elif layout == "BSND":
        b, t, heads, _ = aqk.shape
        leading = (b, heads, t)
    elif layout == "NTD":
        heads, t, _ = aqk.shape
        leading = (heads, t)
    else:
        t, heads, _ = aqk.shape
        leading = (heads, t)
    return (torch.zeros((*leading, 64), dtype=torch.float32),
            torch.zeros((*leading, 128), dtype=torch.bfloat16),
            torch.zeros((*leading, 128), dtype=torch.float32))
