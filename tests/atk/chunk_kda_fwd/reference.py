"""chunk_kda_fwd 的 CPU 标杆（CANNBot 02 唯一 reference.py）。

公式与 `docs/design.md` 的 Stage 语义一致，因果分数的稳定化写法与
`tests/atk/chunk_kda_fwd/executor_chunk_kda_fwd.py` 的 `_stable_causal_scores`
逐行保持一致（按 16 行查询块做 gate 平移，避免 2^(g_i-g_j) 溢出）。

支持的接口子集：`use_gate_in_kernel=False`、`stateVFirst=False`、
`use_qk_l2norm_in_kernel=False`、`use_beta_sigmoid_in_kernel=False`。
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import torch

_LAYOUTS = {"BSND", "BNSD", "TND", "NTD"}


def _to_bsnd(x: torch.Tensor, layout: str, *, beta: bool = False) -> torch.Tensor:
    if layout == "BSND":
        return x
    if layout == "BNSD":
        return x.transpose(1, 2)
    if layout == "TND":
        return x.unsqueeze(0)
    if layout == "NTD":
        return x.unsqueeze(0).transpose(1, 2)
    raise ValueError(f"unsupported layout: {layout}")


def _spans(batch: int, total_t: int, chunk: int, cu: Optional[list]):
    if cu is None:
        return [
            (b, b, c, s, min(s + chunk, total_t))
            for b in range(batch)
            for c, s in enumerate(range(0, total_t, chunk))
        ]
    spans = []
    chunk_id = 0
    for seq_id, (a, b) in enumerate(zip(cu, cu[1:])):
        for s in range(a, b, chunk):
            spans.append((0, seq_id, chunk_id, s, min(s + chunk, b)))
            chunk_id += 1
    return spans


def _stable_causal_scores(q: torch.Tensor, k: torch.Tensor, g: torch.Tensor, scale: float):
    head_num, length, _ = q.shape
    qk = torch.zeros((head_num, length, length), dtype=q.dtype, device=q.device)
    kk = torch.zeros_like(qk)
    column_ids = torch.arange(length, device=q.device)

    for row_start in range(0, length, 16):
        row_end = min(row_start + 16, length)
        shift = g[:, row_end - 1 : row_end]
        left_factor = torch.exp2(g[:, row_start:row_end] - shift)
        right_factor = torch.exp2(shift - g[:, :row_end])
        right = k[:, :row_end] * right_factor
        q_left = q[:, row_start:row_end] * left_factor
        k_left = k[:, row_start:row_end] * left_factor
        q_scores = torch.bmm(q_left, right.transpose(1, 2)).mul(float(scale))
        k_scores = torch.bmm(k_left, right.transpose(1, 2))
        causal = column_ids[:row_end].unsqueeze(0) <= column_ids[row_start:row_end].unsqueeze(1)
        qk[:, row_start:row_end, :row_end] = q_scores.masked_fill(~causal, 0.0)
        kk[:, row_start:row_end, :row_end] = k_scores.masked_fill(~causal, 0.0)
    return qk, kk


def chunk_kda_fwd_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    chunk_size: int = 64,
    *,
    layout: str = "BSND",
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[Iterable[int]] = None,
    dtype: Optional[torch.dtype] = None,
):
    """返回 ``(attn_out, final_state)``；未请求 final_state 时第二项为 None。"""
    if layout not in _LAYOUTS:
        raise ValueError(f"unsupported layout: {layout}")
    cd = dtype or torch.float64
    q_b = _to_bsnd(q, layout).to(cd)
    k_b = _to_bsnd(k, layout).to(cd)
    v_b = _to_bsnd(v, layout).to(cd)
    beta_b = _to_bsnd(beta, layout, beta=True).to(cd)
    g_b = _to_bsnd(g, layout).to(cd)

    batch, total_t, h_num, k_dim = q_b.shape
    hv_num, v_dim = v_b.shape[2], v_b.shape[3]
    group = hv_num // h_num
    cu = None if cu_seqlens is None else [int(x) for x in cu_seqlens]
    seq_num = batch if cu is None else len(cu) - 1

    gk = torch.empty_like(g_b)
    for _, _, _, start, end in _spans(batch, total_t, chunk_size, cu):
        gk[:, start:end] = torch.cumsum(g_b[:, start:end], dim=1) / math.log(2.0)

    out = torch.zeros((batch, total_t, hv_num, v_dim), dtype=cd)
    state = torch.zeros((seq_num, hv_num, k_dim, v_dim), dtype=cd)
    if initial_state is not None:
        state = initial_state.to(cd).clone()

    spans = _spans(batch, total_t, chunk_size, cu)
    eye_cache = {}
    for batch_id, seq_id, _chunk_id, start, end in spans:
        length = end - start
        q_c = q_b[batch_id, start:end].permute(1, 0, 2).repeat_interleave(group, 0)
        k_c = k_b[batch_id, start:end].permute(1, 0, 2).repeat_interleave(group, 0)
        v_c = v_b[batch_id, start:end].permute(1, 0, 2)
        be = beta_b[batch_id, start:end].transpose(0, 1)
        g_c = gk[batch_id, start:end].permute(1, 0, 2)

        exp_g = torch.exp2(g_c)
        qg = q_c * exp_g
        qk, kk = _stable_causal_scores(q_c, k_c, g_c, float(scale))

        strict = torch.ones((length, length), dtype=torch.bool).tril(-1)
        lhs = kk.mul(be.unsqueeze(-1)).masked_fill(~strict, 0.0)
        eye = eye_cache.get(length)
        if eye is None:
            eye = torch.eye(length, dtype=cd)
            eye_cache[length] = eye
        inv = torch.linalg.solve_triangular(lhs + eye, eye.expand(hv_num, -1, -1), upper=False)

        w = torch.bmm(inv, k_c * be.unsqueeze(-1) * exp_g)
        u = torch.bmm(inv, v_c * be.unsqueeze(-1))

        last_g = g_c[:, -1]
        kg = k_c * torch.exp2(last_g.unsqueeze(1) - g_c)

        prev = state[seq_id].clone()
        v_new = u - torch.bmm(w, prev)
        state[seq_id] = torch.exp2(last_g).unsqueeze(-1) * prev + torch.bmm(
            kg.transpose(1, 2), v_new
        )
        o = torch.bmm(qg, prev) * float(scale) + torch.bmm(qk, v_new)
        out[batch_id, start:end] = o.permute(1, 0, 2)

    final_state = state.to(torch.float32) if output_final_state else None
    if layout in {"TND", "NTD"}:
        out = out.squeeze(0)
    return out.to(q.dtype), final_state
