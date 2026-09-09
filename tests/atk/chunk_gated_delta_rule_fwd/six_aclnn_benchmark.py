#!/usr/bin/env python3
"""chunk_gated_delta_rule_fwd 的真实六 ACLNN 小算子 benchmark。"""

from __future__ import annotations

import math
from typing import Iterable, Optional


SIX_ACLNN_OPS = (
    "chunk_local_cumsum",
    "chunk_scaled_dot_kkt",
    "solve_tri",
    "recompute_w_u_fwd",
    "chunk_gated_delta_rule_fwd_h",
    "chunk_fwd_o",
)

VARLEN_CUMSUM_TRANSPORT = "public_chunk_local_cumsum"


def _chunk_indices(
    cu_seqlens: Optional[Iterable[int]], chunk_size: int
) -> Optional[list[int]]:
    if cu_seqlens is None:
        return None
    values = [int(value) for value in cu_seqlens]
    result: list[int] = []
    for sequence, (begin, end) in enumerate(zip(values, values[1:])):
        for local_chunk in range(math.ceil((end - begin) / chunk_size)):
            result.extend((sequence, local_chunk))
    return result


def expand_qk_to_value_heads(q, k, v):
    """把公开 GVA 的 Hk q/k 显式扩展到六小算子需要的 Hv。"""

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q/k/v 必须都是四维 BHTD 张量")
    if q.shape[:3] != k.shape[:3]:
        raise ValueError(f"q/k shape 不匹配：{tuple(q.shape)}/{tuple(k.shape)}")
    key_heads = q.shape[1]
    value_heads = v.shape[1]
    if key_heads <= 0 or value_heads % key_heads != 0:
        raise ValueError(f"GVA 要求 Hk 能整除 Hv：Hk={key_heads}, Hv={value_heads}")
    if key_heads == value_heads:
        return q, k
    ratio = value_heads // key_heads
    return (
        q.repeat_interleave(ratio, dim=1).contiguous(),
        k.repeat_interleave(ratio, dim=1).contiguous(),
    )


def run_six_aclnn_core(
    ascendc,
    q,
    k,
    v,
    g_token_first,
    beta_token_first,
    *,
    initial_state,
    output_final_state: bool,
    chunk_size: int,
    cu_seqlens: Optional[Iterable[int]],
    scale: float,
):
    """按生产六小算子顺序执行，并返回与 Phase6 相同的四个逻辑输出。"""

    q, k = expand_qk_to_value_heads(q, k, v)
    cu_list = None if cu_seqlens is None else [int(value) for value in cu_seqlens]
    chunk_list = _chunk_indices(cu_list, chunk_size)
    g_head_first = ascendc.chunk_local_cumsum(
        g_token_first.transpose(1, 2).contiguous(),
        chunk_size=chunk_size,
        cu_seqlens=cu_list,
        chunk_indices_out=chunk_list,
        head_first=True,
        output_dtype="float32",
    )
    beta_head_first = beta_token_first.transpose(1, 2).contiguous().float()
    a_raw = ascendc.chunk_scaled_dot_kkt(
        k=k,
        g=g_head_first,
        beta=beta_head_first,
        cu_seqlens=cu_list,
        chunk_indices=chunk_list,
        chunk_size=chunk_size,
    )
    if cu_list is None:
        a = ascendc.solve_tri(a_raw.to(q.dtype), layout="bhtd")
    else:
        a_token_first = a_raw.transpose(1, 2).contiguous().squeeze(0)
        a_token_first = ascendc.solve_tri(
            a_token_first.to(q.dtype),
            cu_seqlens=cu_list,
            chunk_indices=chunk_list,
            layout="tnd",
        )
        a = a_token_first.unsqueeze(0).transpose(1, 2).contiguous()

    w, u = ascendc.recompute_w_u_fwd(
        k,
        v,
        beta_head_first,
        a,
        chunk_size,
        g=g_head_first,
        gk=None,
        cu_seqlens=cu_list,
        chunk_indices=chunk_list,
    )
    h, v_new, final_state = ascendc.chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g=g_head_first,
        gk=None,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_list,
        chunk_indices=chunk_list,
        state_v_first=False,
    )
    o = ascendc.chunk_fwd_o(
        q,
        k,
        v_new,
        h,
        scale,
        g=g_head_first,
        g_gamma=None,
        cu_seqlens=cu_list,
        chunk_indices=chunk_list,
        chunk_size=chunk_size,
        transpose_state_layout=False,
    )
    if not output_final_state:
        final_state = None
    return o, final_state, g_head_first.transpose(1, 2).contiguous(), a
