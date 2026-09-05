# fused_recurrent_rwkv8 (WKV7) 前向的纯 PyTorch 金标（z/b 参数化口径）。
# 本文件不依赖 torch_npu / Triton，可直接在 CPU 上跑。
#
# 上游语义锚点：
#   semantics:      BlinkDL/RWKV-LM @ 9521024
#                   RWKV-v8/cuda/wkv7_cuda.cu forward_kernel (lines 10-52)
#                   - line 21:      decay = __expf(-__expf(w))（w 为 log 域参数）
#                   - lines 27-41:  逐 token 递推（sa 读出、衰减+移除+写入、y 读出）
#                   调用方式: RWKV-v8/251105_reverse_run.py:249
#                   x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a) → z = -kk, b = kk*a
#   precision_peer: fla-org/flash-linear-attention @ a4a2624b
#                   fla/ops/generalized_delta_rule/dplr
#                   （fla state 与 RWKV 朝向互为转置；gk = -exp(w)）
#
# 递推公式（per head，state (B,H,V,K)，RWKV 朝向：行 = v/q 侧，列 = k/z 侧）：
#   sa    = state @ z_t                       # 移除读出（z 即 fla/早期的 a）
#   state = state * decay_t[None, :]          # 逐通道衰减（沿列/k 侧）
#         + sa[:, None] * b_t[None, :]        # 移除（rank-1）
#         + v_t[:, None] * k_t[None, :]       # delta-rule 写入
#   y_t   = state @ (q_t * scale)             # 读出（scale 作用在 q 上）
# K（q/w/k/z/b 侧）与 V（v/o/sa 侧）是两个独立维度（io 布局 BHTC：k (B,H,T,K)、
# v (B,H,T,V)，H 在 T 前使每核 (b,h) 数据段连续）；K=V=N 是其特例。
#
# ⚠️ 本文件与 tests/atk/fused_recurrent_rwkv8/executor_fused_recurrent_rwkv8.py
# 中内嵌的 CPU 标杆是同一份逻辑的两份拷贝（executor 按 ATK 规范必须自包含）。
# 修改金标算法时两处必须同步。
from __future__ import annotations

from typing import NamedTuple, Optional

import torch

__all__ = ["FusedRecurrentRwkv8Result", "fused_recurrent_rwkv8_golden", "wkv7_decay"]


class FusedRecurrentRwkv8Result(NamedTuple):
    """算子输出。用 NamedTuple 而非 dataclass：同时支持属性访问与解包。"""

    o: torch.Tensor
    s: Optional[torch.Tensor]            # chunk 快照，output_chunk_state=False 时为 None
    sa: Optional[torch.Tensor]           # 每 token 移除读出，output_sa=False 时为 None


def wkv7_decay(w: torch.Tensor) -> torch.Tensor:
    """log 域衰减参数 -> 衰减值：decay = exp(-exp(w))，与 wkv7_cuda.cu:21 一致。

    注意与 fla 惯例的区别：fla DPLR 的 gk 是 decay 的对数（decay = exp(gk)），
    本算子的 w 是"log 的 log"（RWKV-v8 官方口径），换算关系 gk = -exp(w)。
    """
    return torch.exp(-torch.exp(w))


def _check_inputs(q, w, k, v, z, b):
    assert q.shape == w.shape == k.shape == z.shape == b.shape, \
        f'k-side shape mismatch: {q.shape=} {w.shape=} {k.shape=} {z.shape=} {b.shape=}'
    assert q.ndim == 4, f'expect q (B, H, T, K), got {q.shape}'
    assert v.ndim == 4 and v.shape[:3] == q.shape[:3], \
        f'expect v (B, H, T, V) with same B/H/T as q, got {v.shape} vs {q.shape}'


def fused_recurrent_rwkv8_golden(
    q: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    scale: float = 1.0,
    initial_state: Optional[torch.Tensor] = None,
    reverse: bool = False,
    output_chunk_state: bool = False,
    output_sa: bool = False,
    chunk_len: int = 16,
) -> FusedRecurrentRwkv8Result:
    """逐 token 递推参考实现（einsum 版；对外接口实现、精度真值来源）。

    Args:
        q, w, k, z, b: (B, H, T, K)；v: (B, H, T, V)。K 与 V 独立；io 布局 BHTC
            （H 在 T 前，与 fla DPLR 的 BTHC 口径不同，注意对接时转置）。
            w 为 log 域衰减参数（decay = exp(-exp(w))），z 为移除读出向量（= -kk），
            b 为移除强度向量（= kk * a_inctx）
        scale: q 的缩放系数，默认 1.0（与官方 CUDA 一致；fla 接口默认 1/sqrt(K)，
            属于调用方选择，不改变算子数学）
        initial_state: (B, H, K, V) fp32，计算最优朝向（与 s 快照、fla 一致，即内核账本
            Sᵀ 原样）；入口转置为内部 (V,K) 朝向计算
        reverse: T 维倒序递推（对齐 fla fused_recurrent 的 reverse）。True 时
            initial_state 种子在 t=T-1 侧，递推方向 t=T-1→0
        output_chunk_state: 是否输出 chunk 快照 s（训练预埋，对齐官方 CUDA
            kernel 的 s_ 输出）
        output_sa: 是否输出每 token 的移除读出 sa（训练预埋，对齐官方 sa_）
        chunk_len: s 快照间隔，默认 16（对齐官方 wkv7_cuda.cu backward 的 chunk
            重建粒度；非 16 值与官方 backward 不兼容）
    Returns:
        FusedRecurrentRwkv8Result(o, s, sa)：
        o 为 (B, H, T, V)，dtype 与输入一致（内部 fp32 累加）；
        s 为 (B, H, T//chunk_len, K, V) fp32，**官方 CUDA 转置布局**（= 内核 Sᵀ 原样：
        快照 [j][i] = S[i][j]，j=k/z 侧、i=v/q 侧）；
        每满 chunk_len 个 token 拍一次（按 token 下标 (t+1)%chunk_len==0 判定，
        槽位 t/chunk_len），T 非 chunk_len 倍数时尾部不满的一段无快照
        （floor 语义，与官方一致）；
        sa 为 (B, H, T, V) fp32，sa_t = state_{t-1} @ z_t（state 更新前）。
    """
    _check_inputs(q, w, k, v, z, b)
    orig_dtype = q.dtype
    q, w, k, v, z, b = (x.float() for x in (q, w, k, v, z, b))
    B, H, T, K = q.shape
    V = v.shape[-1]

    if initial_state is None:
        state = q.new_zeros(B, H, V, K)
    else:
        state = initial_state.float().transpose(-1, -2).clone()  # 接口 (K,V) → 内部 (V,K)
    decay = wkv7_decay(w)  # (B, H, T, K)

    o = torch.empty(B, H, T, V, dtype=torch.float32)
    sa_out = torch.empty(B, H, T, V, dtype=torch.float32) if output_sa else None
    s_snaps = []
    for i in range(T):
        t = T - 1 - i if reverse else i
        # sa_i = sum_j state_ij * z_j
        sa = torch.einsum('bhij,bhj->bhi', state, z[:, :, t])
        if output_sa:
            sa_out[:, :, t] = sa
        state = (state * decay[:, :, t].unsqueeze(-2)          # S_ij * w_j
                 + sa.unsqueeze(-1) * b[:, :, t].unsqueeze(-2)  # sa_i * b_j
                 + v[:, :, t].unsqueeze(-1) * k[:, :, t].unsqueeze(-2))  # v_i * k_j
        # o_i = sum_j state_ij * (q_j * scale)
        o[:, :, t] = torch.einsum('bhij,bhj->bhi', state, q[:, :, t] * scale)
        if output_chunk_state and (t + 1) % chunk_len == 0:
            # 官方 CUDA s_ 布局 = RWKV 朝向的转置（快照 [j][i] = S[i][j]，(K,V)）
            s_snaps.append((t // chunk_len, state.transpose(-1, -2)))

    o = o.to(orig_dtype)
    if output_chunk_state:
        s = q.new_zeros(B, H, T // chunk_len, K, V)
        for slot, snap in s_snaps:
            s[:, :, slot] = snap
    else:
        s = None
    return FusedRecurrentRwkv8Result(o, s, sa_out)
