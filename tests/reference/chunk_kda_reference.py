# Copyright (c) 2026 Tianjin University, Ltd.
#
# Pure PyTorch reference for chunk KDA forward. This file is intentionally
# independent from torch_npu and Triton so ATK CPU golden code can reuse it.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ChunkKdaForwardResult:
    o: torch.Tensor
    final_state: Optional[torch.Tensor]
    Aqk: torch.Tensor
    Akk: torch.Tensor
    w: torch.Tensor
    u: torch.Tensor
    qg: torch.Tensor
    kg: torch.Tensor
    v_new: torch.Tensor
    h: torch.Tensor


def _chunk_spans(
    batch: int,
    total_t: int,
    chunk_size: int,
    cu_seqlens: Optional[torch.Tensor],
) -> list[tuple[int, int, int]]:
    if cu_seqlens is None:
        spans: list[tuple[int, int, int]] = []
        for b in range(batch):
            for start in range(0, total_t, chunk_size):
                spans.append((b, start, min(start + chunk_size, total_t)))
        return spans

    if batch != 1:
        raise ValueError("varlen reference expects flattened batch B=1.")

    cu = cu_seqlens.detach().cpu().tolist()
    spans = []
    for n in range(len(cu) - 1):
        seq_start, seq_end = int(cu[n]), int(cu[n + 1])
        for start in range(seq_start, seq_end, chunk_size):
            spans.append((0, start, min(start + chunk_size, seq_end)))
    return spans


def _lower_inverse(mat: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(mat.shape[-1], device=mat.device, dtype=torch.float32)
    lhs = eye + torch.tril(mat.to(torch.float32), diagonal=-1)
    return torch.linalg.solve_triangular(lhs, eye, upper=False)


def chunk_kda_forward_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    chunk_size: int = 64,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> ChunkKdaForwardResult:
    """Reference KDA forward for token-first tensors.

    Args:
        q: [B, T, H, K].
        k: [B, T, H, K].
        v: [B, T, HV, V].
        gk: [B, T, HV, K], cumulative key-wise gate in log2 space.
        beta: [B, T, HV].
        scale: attention scale.
        chunk_size: 64 or 128 in the intended NPU implementation.
        initial_state: optional [N, HV, K, V] state.
        output_final_state: whether to return final state.
        cu_seqlens: optional flattened varlen boundaries.
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q/k/v must be 4-D token-first tensors.")
    if gk.dim() != 4 or beta.dim() != 3:
        raise ValueError("gk must be [B, T, HV, K] and beta must be [B, T, HV].")
    if q.shape != k.shape:
        raise ValueError("q and k must have identical shape.")
    bsz, total_t, hq, kdim = q.shape
    b_v, t_v, hv, vdim = v.shape
    if (b_v, t_v) != (bsz, total_t):
        raise ValueError("v shape prefix must match q/k.")
    if gk.shape != (bsz, total_t, hv, kdim):
        raise ValueError("gk shape must be [B, T, HV, K].")
    if beta.shape != (bsz, total_t, hv):
        raise ValueError("beta shape must be [B, T, HV].")
    if hv % hq != 0:
        raise ValueError("HV must be divisible by H.")

    spans = _chunk_spans(bsz, total_t, chunk_size, cu_seqlens)
    nt = len(spans) if cu_seqlens is not None else (total_t + chunk_size - 1) // chunk_size
    group = hv // hq
    num_seq = bsz if cu_seqlens is None else int(cu_seqlens.numel() - 1)

    out_dtype = v.dtype
    device = q.device
    o = torch.zeros((bsz, total_t, hv, vdim), device=device, dtype=out_dtype)
    Aqk = torch.zeros((bsz, total_t, hv, chunk_size), device=device, dtype=q.dtype)
    Akk = torch.zeros((bsz, total_t, hv, chunk_size), device=device, dtype=q.dtype)
    w = torch.zeros((bsz, total_t, hv, kdim), device=device, dtype=q.dtype)
    u = torch.zeros((bsz, total_t, hv, vdim), device=device, dtype=v.dtype)
    qg = torch.zeros((bsz, total_t, hv, kdim), device=device, dtype=q.dtype)
    kg = torch.zeros((bsz, total_t, hv, kdim), device=device, dtype=k.dtype)
    v_new = torch.zeros_like(v)
    h_out = torch.zeros((bsz, nt, hv, kdim, vdim), device=device, dtype=q.dtype)

    if initial_state is None:
        state = torch.zeros((num_seq, hv, kdim, vdim), device=device, dtype=torch.float32)
    else:
        state = initial_state.to(torch.float32).clone()

    seq_state_index = 0
    prev_b = None
    chunk_counter: dict[int, int] = {}
    if cu_seqlens is not None:
        cu_cpu = cu_seqlens.detach().cpu().tolist()
    else:
        cu_cpu = None

    for span_idx, (b, start, end) in enumerate(spans):
        if cu_cpu is not None:
            while seq_state_index + 1 < len(cu_cpu) - 1 and start >= cu_cpu[seq_state_index + 1]:
                seq_state_index += 1
            state_b = seq_state_index
            chunk_idx = span_idx
        else:
            if prev_b != b:
                chunk_counter[b] = 0
                prev_b = b
            state_b = b
            chunk_idx = chunk_counter[b]
            chunk_counter[b] += 1

        cur_t = end - start
        local_cols = slice(0, cur_t)
        for ihv in range(hv):
            ih = ihv // group
            q_blk = q[b, start:end, ih].to(torch.float32)
            k_blk = k[b, start:end, ih].to(torch.float32)
            v_blk = v[b, start:end, ihv].to(torch.float32)
            g_blk = gk[b, start:end, ihv].to(torch.float32)
            beta_blk = beta[b, start:end, ihv].to(torch.float32)

            rel = g_blk[:, None, :] - g_blk[None, :, :]
            qk = torch.einsum("ik,jk,ijk->ij", q_blk, k_blk, torch.exp2(rel)) * float(scale)
            kk = torch.einsum("ik,jk,ijk->ij", k_blk, k_blk, torch.exp2(rel))
            tril_qk = torch.tril(qk, diagonal=0)
            tril_kk = torch.tril(kk * beta_blk[:, None], diagonal=-1)
            inv_akk = _lower_inverse(tril_kk)

            k_beta_g = k_blk * beta_blk[:, None] * torch.exp2(g_blk)
            v_beta = v_blk * beta_blk[:, None]
            w_blk = inv_akk @ k_beta_g
            u_blk = inv_akk @ v_beta

            last_g = g_blk[cur_t - 1]
            qg_blk = q_blk * torch.exp2(g_blk)
            kg_blk = k_blk * torch.exp2(last_g[None, :] - g_blk)

            h_prev = state[state_b, ihv]
            h_out[b, chunk_idx, ihv] = h_prev.to(h_out.dtype)
            v_new_blk = u_blk - w_blk @ h_prev
            state[state_b, ihv] = torch.exp2(last_g)[:, None] * h_prev + kg_blk.T @ v_new_blk

            o_inter = qg_blk @ h_out[b, chunk_idx, ihv].to(torch.float32) * float(scale)
            o_local = tril_qk @ v_new_blk
            o[b, start:end, ihv] = (o_inter + o_local).to(out_dtype)

            Aqk[b, start:end, ihv, local_cols] = tril_qk.to(Aqk.dtype)
            Akk[b, start:end, ihv, local_cols] = inv_akk.to(Akk.dtype)
            w[b, start:end, ihv] = w_blk.to(w.dtype)
            u[b, start:end, ihv] = u_blk.to(u.dtype)
            qg[b, start:end, ihv] = qg_blk.to(qg.dtype)
            kg[b, start:end, ihv] = kg_blk.to(kg.dtype)
            v_new[b, start:end, ihv] = v_new_blk.to(v_new.dtype)

    final_state = state if output_final_state else None
    return ChunkKdaForwardResult(
        o=o,
        final_state=final_state,
        Aqk=Aqk,
        Akk=Akk,
        w=w,
        u=u,
        qg=qg,
        kg=kg,
        v_new=v_new,
        h=h_out,
    )


def run_smoke() -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 32, 2, 16, dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn(1, 32, 2, 32, dtype=torch.float32)
    gk = torch.randn(1, 32, 2, 16, dtype=torch.float32).cumsum(dim=1) * 0.01
    beta = torch.sigmoid(torch.randn(1, 32, 2, dtype=torch.float32))
    res = chunk_kda_forward_reference(q, k, v, gk, beta, scale=16 ** -0.5, chunk_size=16, output_final_state=True)
    for name, value in vars(res).items():
        if value is not None and not torch.isfinite(value).all():
            raise RuntimeError(f"{name} contains NaN or Inf")


if __name__ == "__main__":
    run_smoke()
