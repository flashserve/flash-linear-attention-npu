#!/usr/bin/env python3
# Copyright (c) 2026 Tianjin University, Ltd.
"""Golden for ChunkKdaFwdIntraSubChunk (BNSD, MHA + GVA).

q/k: [B,H,T,K]; g/beta/Aqk/Akkd: [B,HV,T,*] with HV >= H and HV % H == 0.
Head map: i_h = i_hv // (HV // H)  (same as GPU Triton).
"""

from __future__ import annotations

import math
from typing import Optional

import torch


BC = 16


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> list[int]:
    indices: list[int] = []
    for seq in range(int(cu_seqlens.numel()) - 1):
        length = int(cu_seqlens[seq + 1] - cu_seqlens[seq])
        n_chunks = (length + chunk_size - 1) // chunk_size
        for local in range(n_chunks):
            indices.extend([seq, local])
    return indices


def _forward_sub_inv(L: torch.Tensor) -> torch.Tensor:
    """(I-L)^{-1} via Triton-style forward substitution. L is [..., valid, valid]."""
    valid = L.shape[-1]
    M = -L.clone()
    for i in range(2, valid):
        a = M[..., i, :].clone()
        a[..., i:] = 0
        a = a + (a.unsqueeze(-2) @ M).squeeze(-2)
        a[..., i:] = 0
        M[..., i, :] = a
    eye = torch.eye(valid, dtype=M.dtype, device=M.device)
    return M + eye


def chunk_kda_fwd_intra_sub_chunk_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    chunk_size: int,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    *,
    dtype: torch.dtype = torch.float64,
    score_dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference matching Triton safe-gate diagonal path. Inputs are BNSD.

    score_dtype: if set (e.g. bfloat16), cast qg/kpos/kneg to that type before the
    two score GEMMs — matches Ascend Cube path (Vector fp32 → Cast T → MMAD).
    Forward-sub stays in ``dtype`` (fp32/fp64).
    """
    assert q.ndim == 4 and q.shape == k.shape
    B, H, T, K = q.shape
    assert g.ndim == 4 and g.shape[0] == B and g.shape[2] == T and g.shape[3] == K
    HV = g.shape[1]
    assert HV >= H and HV % H == 0
    assert beta.shape == (B, HV, T)
    assert chunk_size in (32, 64, 128)
    BT = chunk_size
    NC = BT // BC
    group = HV // H

    q_r = q.to(dtype)
    k_r = k.to(dtype)
    g_r = g.to(dtype)
    beta_r = beta.to(dtype)

    aqk = torch.zeros(B, HV, T, BT, dtype=dtype, device=q.device)
    akkd = torch.zeros(B, HV, T, BC, dtype=dtype, device=q.device)

    def process_block(i_b: int, hv_slice: slice, i_h_map: torch.Tensor, bos: int, local_t: int,
                      local_chunk: int, i_i: int):
        i_ti = local_chunk * BT + i_i * BC
        if i_ti >= local_t:
            return
        valid = min(BC, local_t - i_ti)
        mid = i_ti + min(BC // 2, local_t - i_ti - 1)
        rows = list(range(bos + i_ti, bos + i_ti + valid))

        # g/beta indexed by HV heads in hv_slice; q/k by mapped H heads
        g_block = g_r[i_b, hv_slice, rows, :]  # [nh, valid, K]
        g_mid = g_r[i_b, hv_slice, bos + mid, :]  # [nh, K]
        gm = g_block - g_mid[:, None, :]
        gq = torch.exp2(gm)
        gk = torch.exp2(-gm)

        q_sel = q_r[i_b, i_h_map]  # [nh, T, K]
        k_sel = k_r[i_b, i_h_map]
        q_block = q_sel[:, rows, :] * gq
        k_pos = k_sel[:, rows, :] * gq
        k_neg = k_sel[:, rows, :] * gk
        if score_dtype is not None:
            # Cube: Vector computes in fp32, then stores T for MMAD.
            q_block = q_block.to(score_dtype).to(dtype)
            k_pos = k_pos.to(score_dtype).to(dtype)
            k_neg = k_neg.to(score_dtype).to(dtype)
        beta_row = beta_r[i_b, hv_slice, rows]

        aqk_blk = torch.matmul(q_block, k_neg.transpose(-1, -2)) * scale
        akk_blk = torch.matmul(k_pos, k_neg.transpose(-1, -2)) * beta_row[:, :, None]

        tril = torch.tril(torch.ones(valid, valid, dtype=torch.bool, device=q.device))
        strict = torch.tril(torch.ones(valid, valid, dtype=torch.bool, device=q.device), diagonal=-1)
        aqk_blk = torch.where(tril, aqk_blk, torch.zeros_like(aqk_blk))
        L = torch.where(strict, akk_blk, torch.zeros_like(akk_blk))
        M = _forward_sub_inv(L)

        aqk[i_b, hv_slice, rows, i_i * BC : i_i * BC + valid] = aqk_blk[:, :, :valid]
        akkd[i_b, hv_slice, rows, :valid] = M[:, :, :valid]
        if valid < BC:
            akkd[i_b, hv_slice, rows, valid:] = 0

    if cu_seqlens is None:
        nt = (T + BT - 1) // BT
        hv_idx = torch.arange(HV, device=q.device)
        i_h_map = hv_idx // group
        for i_b in range(B):
            for i_t in range(nt):
                for i_i in range(NC):
                    process_block(i_b, slice(0, HV), i_h_map, 0, T, i_t, i_i)
        return aqk, akkd

    assert B == 1
    if chunk_indices is None:
        flat = prepare_chunk_indices(cu_seqlens, BT)
        chunk_indices = torch.tensor(flat, dtype=torch.long, device=q.device)
    assert chunk_indices.numel() % 2 == 0
    hv_idx = torch.arange(HV, device=q.device)
    i_h_map = hv_idx // group
    for nt_i in range(chunk_indices.numel() // 2):
        seq = int(chunk_indices[nt_i * 2])
        local = int(chunk_indices[nt_i * 2 + 1])
        bos = int(cu_seqlens[seq])
        eos = int(cu_seqlens[seq + 1])
        local_t = eos - bos
        for i_i in range(NC):
            process_block(0, slice(0, HV), i_h_map, bos, local_t, local, i_i)

    return aqk, akkd


def _self_check():
    torch.manual_seed(0)
    B, H, HV, T, K, BT = 1, 2, 4, 64, 32, 64
    q = torch.randn(B, H, T, K)
    k = torch.randn(B, H, T, K)
    g = -torch.linspace(0, 40, T).view(1, 1, T, 1).expand(B, HV, T, K).contiguous()
    beta = torch.rand(B, HV, T)
    scale = 1.0 / math.sqrt(K)
    aqk, akkd = chunk_kda_fwd_intra_sub_chunk_ref(q, k, g, beta, scale, BT)
    assert aqk.shape == (B, HV, T, BT)
    assert akkd.shape == (B, HV, T, BC)
    assert torch.isfinite(aqk).all() and torch.isfinite(akkd).all()
    for t0 in range(0, T, BC):
        diag = torch.stack([akkd[0, 0, t0 + i, i] for i in range(BC)])
        assert torch.allclose(diag, torch.ones(BC, dtype=torch.float64), atol=1e-5, rtol=1e-5)
    # MHA path still works
    aqk_m, _ = chunk_kda_fwd_intra_sub_chunk_ref(q, k, g[:, :H], beta[:, :H], scale, BT)
    assert aqk_m.shape[1] == H
    print("golden self-check passed (MHA+GVA)", aqk.abs().mean().item(), akkd.abs().mean().item())


if __name__ == "__main__":
    _self_check()
