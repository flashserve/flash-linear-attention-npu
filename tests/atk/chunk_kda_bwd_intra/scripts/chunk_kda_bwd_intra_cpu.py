# Copyright (c) 2026 Tianjin University, Ltd.
"""Phase A scaffold. CPU zeros describe the ABI only, never numerical accuracy."""

import torch


def chunk_kda_bwd_intra_cpu(q, k, gk, beta, dAqk, dAkk, dq, dk, db, dg, *,
                          cu_seqlens=None, chunk_indices=None, chunk_size=64,
                          safe_gate=True, layout="BSND"):
    """ABI placeholder: all four outputs FP32, each matching its gradient input."""
    return tuple(torch.zeros(item.shape, dtype=torch.float32) for item in (dq, dk, db, dg))
