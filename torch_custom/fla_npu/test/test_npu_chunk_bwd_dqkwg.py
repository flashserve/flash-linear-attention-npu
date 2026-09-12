# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from typing import Tuple
from fla_npu.ops import ascendc as ascendc_ops

import os

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

from typing import Optional
import pickle
import math
import sys

import time

# ---------------------------------------------------------------------------
# 耗时统计工具（仅诊断用，不影响计算结果）
# 通过 chunk_bwd_dqkwg_cpu(..., verbose_timing=True) 或环境变量
# FLA_NPU_PROFILE=1 启用，会按 stage 汇总耗时并打印明细。
# ---------------------------------------------------------------------------
class _StageTimer:
    def __init__(self):
        self.totals = {}
        self.counts = {}
        self._starts = {}

    def start(self, stage: str):
        self._starts[stage] = time.perf_counter()

    def stop(self, stage: str):
        t0 = self._starts.pop(stage, None)
        if t0 is None:
            return
        dt = time.perf_counter() - t0
        self.totals[stage] = self.totals.get(stage, 0.0) + dt
        self.counts[stage] = self.counts.get(stage, 0) + 1

    def summary(self, total_time: float, chunks: int) -> str:
        if not self.totals:
            return ""
        rows = sorted(self.totals.items(), key=lambda x: -x[1])
        sum_stages = sum(self.totals.values())
        overhead = max(total_time - sum_stages, 0.0)
        lines = []
        lines.append(f"[chunk_bwd_dqkwg_cpu] chunks={chunks} total={total_time*1000:.3f}ms "
                     f"sum_stages={sum_stages*1000:.3f}ms overhead={overhead*1000:.3f}ms "
                     f"({overhead/total_time*100:.1f}%)")
        lines.append(f"[chunk_bwd_dqkwg_cpu]   {'stage':<24}{'total_ms':>12}{'calls':>10}{'avg_us':>12}{'pct':>8}")
        for stage, tot in rows:
            cnt = self.counts[stage]
            avg_us = (tot / cnt) * 1e6 if cnt else 0.0
            pct = (tot / total_time * 100) if total_time > 0 else 0.0
            lines.append(f"[chunk_bwd_dqkwg_cpu]   {stage:<24}{tot*1000:>12.3f}{cnt:>10d}{avg_us:>12.2f}{pct:>7.1f}%")
        return "\n".join(lines)


class _NullTimer:
    def start(self, stage: str): pass
    def stop(self, stage: str): pass
    def summary(self, total_time: float, chunks: int) -> str: return ""

def chunk_bwd_dqkwg_cpu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    dv: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor,
    chunk_size: int = 64,
    benchmark = False,
    verbose_timing: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CPU Equivalent of chunk_bwd_kernel_dqkwg.

    优化说明：将原按 HV 头展开的 Python 循环改为对 HV 维做批量 (bmm) 计算。
    CPU 上 bmm 对每个 batch 调用与单次 mm 相同的 gemm，因此每个头的矩阵乘
    与原循环逐位等价；所有 dtype 往返 (.to(datatype).to(calc_type)) 和运算顺序
    均原样保留，维持与 kernel 一致的精度模拟。

    verbose_timing=True 或环境变量 FLA_NPU_PROFILE=1 时，按 stage 打印耗时明细，
    便于定位瓶颈（仅诊断用，关闭时零开销）。
    """
    if os.environ.get("FLA_NPU_PROFILE", "") == "1":
        verbose_timing = True
    timer = _StageTimer() if verbose_timing else _NullTimer()
    t_total_start = time.perf_counter()
    chunk_count = 0

    # 大规模 shape（float64 benchmark 下尤其严重）时，一次性 batched 处理所有
    # chunk 会生成超大中间张量导致内存耗尽/卡死。限制每次 batched 调用的最大
    # chunk 数，超出时自动拆分为多次子批量调用（chunk 间无数据依赖，结果等价）。
    _MAX_DENSE_BATCH_CHUNKS = int(os.environ.get("FLA_NPU_CPU_MAX_BATCH_CHUNKS", "32"))

    calc_type = torch.float64 if benchmark else torch.float32
    B, T, HK, K = q.shape
    HV = v.shape[2]
    V = v.shape[-1]
    if HK <= 0 or HV <= 0 or HV % HK != 0:
        raise ValueError(f"GVA requires HV divisible by HK, got HV={HV}, HK={HK}")
    n_ratio = HV // HK  # HV = n_ratio * HK
    datatype = q.dtype
    gtype = g.dtype
    if benchmark:
        datatype = torch.float64
        gtype = torch.float64

    # ---- Pad T to multiple of chunk_size (dense mode only) ----
    # 消除 ragged 尾部，使所有 chunk 都是满长度，统一走 batched 路径。
    # q/k/v/do/dv 补零；g 补最后一个有效值（保持单调递减语义）。
    # padding 位置的贡献天然为零（do=0 → ds 行为零；v=0 → ds 列为零）。
    # 唯一需要特殊处理的是 dg_last_accum 的位置：
    #   满 chunk → 加到 C-1；padded tail → 加到 L_orig-1（而非 C-1）。
    C = chunk_size
    T_orig = T
    ragged_len = (T % C) if cu_seqlens is None else 0
    pad_T = (C - ragged_len) % C if ragged_len > 0 else 0
    if pad_T > 0:
        q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, pad_T))
        k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad_T))
        v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad_T))
        do = torch.nn.functional.pad(do, (0, 0, 0, 0, 0, pad_T))
        dv = torch.nn.functional.pad(dv, (0, 0, 0, 0, 0, pad_T))
        g = torch.cat([g, g[:, -1:, :].expand(-1, pad_T, -1)], dim=1)
        T = T + pad_T

    # ---- Pre-transpose + pre-cast ----
    timer.start("pre_transpose")
    q_ct = q.permute(0, 2, 1, 3).to(calc_type).contiguous()   # [B, HK, T, K]
    k_ct = k.permute(0, 2, 1, 3).to(calc_type).contiguous()   # [B, HK, T, K]
    v_ct = v.permute(0, 2, 1, 3).to(calc_type).contiguous()   # [B, HV, T, V]
    do_ct = do.permute(0, 2, 1, 3).to(calc_type).contiguous() # [B, HV, T, V]
    dv_ct = dv.permute(0, 2, 1, 3).to(calc_type).contiguous() # [B, HV, T, V]
    g_ct = g.permute(0, 2, 1).contiguous()                    # [B, HV, T], gtype
    timer.stop("pre_transpose")

    # Keep per-value-head contributions first, then reduce them into key heads.
    timer.start("alloc")
    dq_hv = torch.empty((B, T, HV, K), dtype=datatype)
    dk_hv = torch.empty((B, T, HV, K), dtype=datatype)
    dg = torch.zeros_like(g)
    dw = torch.empty((B, T, HV, K), dtype=datatype)
    timer.stop("alloc")

    # 缓存因果 mask（按 actual_chunk_len），避免每个 chunk 重复构造
    # 同时缓存 calc_type 版本（用于融合掩码乘法），避免 where(scalar) 提升类型
    mask_cache = {}

    def get_causal_mask(L: int, device):
        m = mask_cache.get(L)
        if m is None or m.device != device:
            idx = torch.arange(L, device=device)
            m = idx[:, None] >= idx[None, :]
            mask_cache[L] = m
        return m

    def get_causal_mask_f(L: int, device):
        """返回 [1, L, L] 的 calc_type 浮点 mask，供 ds *= mask_f 原地掩码用。"""
        key = (L, str(calc_type))
        mf = mask_cache.get(key)
        if mf is None or mf.device != device:
            mf = get_causal_mask(L, device).to(calc_type)[None]
            mask_cache[key] = mf
        return mf

    # 模拟 kernel 中间结果的 dtype 往返；datatype == calc_type 时短路（避免无谓分配）
    def cast_round(t):
        if datatype == calc_type:
            return t
        return t.to(datatype).to(calc_type)

    # 将 [HK, L, K] 按 n_ratio 复制到 [HV, L, K]；n_ratio==1 时短路
    def expand_heads(t):
        if n_ratio == 1:
            return t
        return t.repeat_interleave(n_ratio, dim=0)

    def stacked_bmm(lhs_list, rhs_list):
        """将多个 (lhs, rhs) 对 stack 成一次 bmm，结果按原 batch 大小拆分。
           当列表长度为 1 时退化为普通 bmm，不引入额外 cat/split 开销。"""
        if len(lhs_list) == 0:
            return []
        if len(lhs_list) == 1:
            return [cast_round(torch.bmm(lhs_list[0], rhs_list[0]))]
        L = torch.cat(lhs_list, dim=0)
        R = torch.cat(rhs_list, dim=0)
        result = cast_round(torch.bmm(L, R))
        sizes = [t.shape[0] for t in lhs_list]
        return list(result.split(sizes, dim=0))

    def process_sequence(b_idx, t_start, t_end, seq_idx_in_batch, chunk_start_idx):
        nonlocal chunk_count
        seq_len = t_end - t_start
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        for i_t in range(num_chunks):
            chunk_start_token_idx = t_start + i_t * chunk_size
            chunk_end_token_idx = min(t_start + (i_t + 1) * chunk_size, t_end)
            L = chunk_end_token_idx - chunk_start_token_idx
            if L <= 0:
                continue
            s = chunk_start_token_idx
            e = chunk_end_token_idx
            chunk_count += 1

            # ---- 取当前 chunk 全部头的数据，统一以 head 作为 batch 维 ----
            # q/k: [L, HK, K] -> [HK, L, K] -> 复制 n_ratio 份 -> [HV, L, K]
            #   head h_idx 对应 hk_idx = h_idx // n_ratio（与原循环一致）
            timer.start("data_prep")
            q_h = expand_heads(q_ct[b_idx, :, s:e, :])
            k_h = expand_heads(k_ct[b_idx, :, s:e, :])
            v_h = v_ct[b_idx, :, s:e, :]
            do_h = do_ct[b_idx, :, s:e, :]
            # h/dh: [HV, K, V]（保留原始 datatype，供 dg_last_accum 使用）
            h_prev = h[b_idx, i_t + chunk_start_idx, :, :, :]       # [HV, K, V]
            dh_curr = dh[b_idx, i_t + chunk_start_idx, :, :, :]     # [HV, K, V]
            h_prev_t = h_prev.transpose(-1, -2).to(calc_type)       # [HV, V, K]
            dh_curr_t = dh_curr.transpose(-1, -2).to(calc_type)     # [HV, V, K]
            timer.stop("data_prep")

            # -----------------------------------------------------------
            # 1. State Contributions (Inter-chunk)
            # -----------------------------------------------------------
            # b_dq += dot(b_do, b_h); b_dk += dot(b_v, b_dh)
            timer.start("bmm_state")
            dq_from_state = cast_round(torch.bmm(do_h, h_prev_t))   # [HV, L, K]
            dk_from_state = cast_round(torch.bmm(v_h, dh_curr_t))   # [HV, L, K]
            timer.stop("bmm_state")

            # b_dw += dot(b_dv, b_h) (kernel 存 -b_dw)
            timer.start("bmm_dw")
            dv_h = dv_ct[b_idx, :, s:e, :]
            dw_c = cast_round(torch.bmm(dv_h, h_prev_t))       # [HV, L, K]
            dw[b_idx, s:e, :, :] = (-dw_c).permute(1, 0, 2)
            timer.stop("bmm_dw")

            timer.start("mask")
            mask_f = get_causal_mask_f(L, q.device)
            timer.stop("mask")

            # -----------------------------------------------------------
            # 2. Gating / Decay Logic Preparation
            # -----------------------------------------------------------
            timer.start("decay_scale")
            g_h = g_ct[b_idx, :, s:e]                            # [HV, L] (gtype)
            g_last = g_ct[b_idx, :, min(s + chunk_size, t_end) - 1]  # [HV]

            exp_gc = torch.exp(g_h)                              # [HV, L]
            exp_neg_gc_glast = torch.exp(-g_h + g_last[:, None]) # [HV, L]

            dq_from_state = dq_from_state * (exp_gc[:, :, None] * scale)
            dk_from_state = dk_from_state * exp_neg_gc_glast[:, :, None]
            timer.stop("decay_scale")

            # -----------------------------------------------------------
            # 3. Intra-chunk Attention
            # -----------------------------------------------------------
            timer.start("bmm_intra_ds")
            ds = cast_round(torch.bmm(do_h, v_h.transpose(1, 2)))  # [HV, L, L]
            timer.stop("bmm_intra_ds")
            timer.start("decay_apply")
            decay_mat = torch.exp(torch.min(g_h[:, :, None] - g_h[:, None, :], torch.tensor(0)))  # [HV, L, L]
            # 融合：避免多次临时分配，原地链式乘法
            ds.mul_(decay_mat).mul_(mask_f).mul_(scale)
            timer.stop("decay_apply")

            timer.start("bmm_qk")
            qk_t = cast_round(torch.bmm(q_h, k_h.transpose(1, 2)))   # [HV, L, L]

            ds2 = ds * qk_t
            dg_c = ds2.sum(dim=-1)
            dg_c = dg_c - ds2.sum(dim=-2)
            if datatype == gtype:
                dg_c = dg_c.to(gtype)                               # 等价 .to(datatype).to(gtype)
            else:
                dg_c = dg_c.to(datatype).to(gtype)                  # [HV, L]
            timer.stop("bmm_qk")

            # b_dg += sum(b_dq * b_q) ; b_dg -= sum(b_k * b_dk)
            # 保留显式 product+sum（与原参考一致，避免 einsum 改变归约顺序）
            timer.start("dg_state")
            dg_c = cast_round(dg_c)
            dg_c += (dq_from_state * q_h).sum(dim=-1)             # [HV, L]
            dg_c = cast_round(dg_c)
            # k_h * dk_from_state 在 dg_c 和 dg_last_accum 中各用一次，提取公共子表达式
            k_dk_prod = k_h * dk_from_state                       # [HV, L, K]
            dg_c = dg_c - k_dk_prod.sum(dim=-1)                   # [HV, L]

            # b_dg_last += sum(h * dh) * exp(g_last) + sum(dk * k)
            # 注意 h_prev/dh_curr 保留原始 datatype（与原实现一致）
            dg_last_accum = (h_prev * dh_curr).sum(dim=(-1, -2)) * torch.exp(g_last)  # [HV]
            dg_last_accum = dg_last_accum + k_dk_prod.sum(dim=(-1, -2))             # [HV]
            timer.stop("dg_state")

            # b_ds2 = b_ds * (q @ k.T)
            timer.start("dg_intra_accum")
            # 仅块最后一个有效 token 累加 dg_last
            dg_c[:, L - 1] = dg_c[:, L - 1] + dg_last_accum
            dg[b_idx, s:e, :] = dg_c.to(gtype).permute(1, 0)                   # [L, HV]
            timer.stop("dg_intra_accum")

            # -----------------------------------------------------------
            # 4. Final Accumulation for dq, dk
            # -----------------------------------------------------------
            timer.start("bmm_dqdk_intra")
            ds = cast_round(ds)
            dq_intra = cast_round(torch.bmm(ds, k_h))               # [HV, L, K]
            dk_intra = cast_round(torch.bmm(ds.transpose(1, 2), q_h))# [HV, L, K]
            timer.stop("bmm_dqdk_intra")

            timer.start("accumulate")
            dq_total = dq_from_state + dq_intra
            dk_total = dk_from_state + dk_intra
            timer.stop("accumulate")

            timer.start("write_back")
            if datatype == calc_type:
                dq_hv[b_idx, s:e, :, :] = dq_total.permute(1, 0, 2)
                dk_hv[b_idx, s:e, :, :] = dk_total.permute(1, 0, 2)
            else:
                dq_hv[b_idx, s:e, :, :] = dq_total.to(datatype).permute(1, 0, 2)
                dk_hv[b_idx, s:e, :, :] = dk_total.to(datatype).permute(1, 0, 2)
            timer.stop("write_back")

    # ------------------------------------------------------------------
    # 批量 dense 路径：把同一 b 下所有 chunk（含 padded tail）合并成
    # 一次大 bmm，消除 Python chunk 循环与算子派发开销。
    # ragged tail 在函数入口已 pad 到 chunk_size；padding 位置贡献为零
    #（do=0/v=0 → ds=0/dq=0/dk=0），唯一需修正的是 dg_last_accum 的位置：
    # padded tail 的 dg_last_accum 加到 ragged_len-1 而非 C-1。
    # chunk 之间无数据依赖（h/dh 是只读输入，dq_hv 等只写不跨 chunk 读），
    # 故改变 chunk 处理顺序不改变结果；单个 chunk 内运算顺序保持不变。
    # ------------------------------------------------------------------
    def process_dense_batched(b_idx: int, n_full: int, t_offset: int, h_offset: int,
                              ragged_len: int = 0):
        nonlocal chunk_count
        N = n_full
        if N <= 0:
            return
        C = chunk_size
        H = HV
        s0 = t_offset
        chunk_count += N
        timer.start("batched_total")

        # ---- 数据准备：从 pre-transposed tensors 切片，无需 permute+cast ----
        timer.start("batched_data_prep")
        # q/k: [B, HK, T, K] -> [HK, N*C, K] -> [HK, N, C, K] -> [N, HK, C, K]
        q_b = q_ct[b_idx, :, s0:s0 + N * C, :].reshape(HK, N, C, K).permute(1, 0, 2, 3)
        if n_ratio > 1:
            q_b = q_b.repeat_interleave(n_ratio, dim=1)           # [N, HV, C, K]
        q_b = q_b.reshape(N * H, C, K)
        k_b = k_ct[b_idx, :, s0:s0 + N * C, :].reshape(HK, N, C, K).permute(1, 0, 2, 3)
        if n_ratio > 1:
            k_b = k_b.repeat_interleave(n_ratio, dim=1)
        k_b = k_b.reshape(N * H, C, K)
        # v/do: [B, HV, T, V] -> [HV, N*C, V] -> [HV, N, C, V] -> [N, HV, C, V] -> [N*HV, C, V]
        v_b = v_ct[b_idx, :, s0:s0 + N * C, :].reshape(H, N, C, V).permute(1, 0, 2, 3).reshape(N * H, C, V)
        do_b = do_ct[b_idx, :, s0:s0 + N * C, :].reshape(H, N, C, V).permute(1, 0, 2, 3).reshape(N * H, C, V)
        # h/dh: [N, HV, K, V]（原始 datatype 保留，供 dg_last_accum）
        h_b = h[b_idx, h_offset:h_offset + N, :, :, :].reshape(N * H, K, V)
        dh_b = dh[b_idx, h_offset:h_offset + N, :, :, :].reshape(N * H, K, V)
        h_b_t = h_b.transpose(-1, -2).to(calc_type)               # [N*H, V, K]
        dh_b_t = dh_b.transpose(-1, -2).to(calc_type)            # [N*H, V, K]
        timer.stop("batched_data_prep")

        # ---- Phase 1: 合并所有无依赖的 bmm（按 V/K 形状自动分组）----
        timer.start("batched_bmm_phase1")
        # V 组 A：右侧最后维为 K (h_b_t, dh_b_t 均为 [N*H, V, K])
        # lhs: do_b, v_b, dv_b 均为 [N*H, C, V]
        dv_b = dv_ct[b_idx, :, s0:s0 + N * C, :].reshape(H, N, C, V).permute(1, 0, 2, 3).reshape(N * H, C, V)
        v_lhs_A = [do_b, v_b, dv_b]
        v_rhs_A = [h_b_t, dh_b_t, h_b_t]
        dq_state, dk_state, dw_c = stacked_bmm(v_lhs_A, v_rhs_A)
        # V 组 B：右侧最后维为 C (v_b^T 为 [N*H, V, C])，单独处理
        ds = stacked_bmm([do_b], [v_b.transpose(1, 2).contiguous()])[0]
        # K 组：lhs 最后一维为 K (q_b [N*H, C, K])
        qk_t = stacked_bmm([q_b], [k_b.transpose(1, 2).contiguous()])[0]
        timer.stop("batched_bmm_phase1")

        timer.start("batched_write_dw")
        # [N*H, C, K] -> [N, H, C, K] -> [N, C, H, K] -> [N*C, H, K]
        dw_neg = (-dw_c).reshape(N, H, C, K).permute(0, 2, 1, 3).reshape(N * C, H, K)
        dw[b_idx, s0:s0 + N * C, :, :] = dw_neg
        timer.stop("batched_write_dw")

        timer.start("batched_mask")
        mask_f = get_causal_mask_f(C, q.device)                  # [1, C, C]
        timer.stop("batched_mask")

        # ---- 2. Decay scale ----
        timer.start("batched_decay_scale")
        # g: pre-transposed [B, HV, T] -> [HV, N*C] -> [HV, N, C] -> [N, HV, C] -> [N*HV, C]
        g_b = g_ct[b_idx, :, s0:s0 + N * C].reshape(H, N, C).permute(1, 0, 2).reshape(N * H, C)  # gtype
        # 每个 chunk 的最后有效 token = chunk 内第 C-1 个位置（满 chunk 下成立）
        g_last = g_b.reshape(N, H, C)[:, :, C - 1].reshape(N * H)                          # [N*H]
        exp_gc = torch.exp(g_b)                                                             # [N*H, C]
        exp_neg_gc_glast = torch.exp(-g_b + g_last[:, None])                               # [N*H, C]
        dq_state = dq_state * (exp_gc[:, :, None] * scale)
        dk_state = dk_state * exp_neg_gc_glast[:, :, None]
        timer.stop("batched_decay_scale")

        # ---- 3. Intra-chunk attention ----
        timer.start("batched_decay_apply")
        # 用 clamp 替代 min(tensor, scalar)，避免每 chunk 创建标量张量
        # in-place exp_ 节省一次 [N*H, C, C] 分配
        g_diff = g_b[:, :, None] - g_b[:, None, :]
        decay_mat = torch.clamp(g_diff, max=0).exp_()                                      # [N*H, C, C]
        ds.mul_(decay_mat).mul_(mask_f).mul_(scale)
        timer.stop("batched_decay_apply")

        timer.start("batched_bmm_qk")
        ds2 = ds * qk_t
        dg_c = ds2.sum(dim=-1)
        dg_c = dg_c - ds2.sum(dim=-2)
        timer.stop("batched_bmm_qk")

        # ---- dg_state（显式 product+sum，与原参考一致）----
        timer.start("batched_dg_state")
        dg_c = cast_round(dg_c)
        dg_c += (dq_state * q_b).sum(dim=-1)                                              # [N*H, C]
        dg_c = cast_round(dg_c)
        # k_b * dk_state 在 dg_c 和 dg_last_accum 中各用一次，提取公共子表达式
        k_dk_prod = k_b * dk_state                                                       # [N*H, C, K]
        dg_c = dg_c - k_dk_prod.sum(dim=-1)
        # h_prev*dh_curr 保留原始 datatype（与原实现一致）
        dg_last_accum = (h_b * dh_b).sum(dim=(-1, -2)) * torch.exp(g_last)                # [N*H]
        dg_last_accum = dg_last_accum + k_dk_prod.sum(dim=(-1, -2))                      # [N*H]
        timer.stop("batched_dg_state")

        # 每个 chunk 累加 dg_last：
        #   满 chunk → 加到 C-1；padded tail → 加到 ragged_len-1（原始最后有效 token）
        timer.start("batched_dg_intra_accum")
        dg_c = dg_c.reshape(N, H, C)
        dg_last_reshaped = dg_last_accum.reshape(N, H)
        if ragged_len > 0 and N > 1:
            dg_c[:N-1, :, C - 1] += dg_last_reshaped[:N-1]
            dg_c[N-1, :, ragged_len - 1] += dg_last_reshaped[N-1]
        elif ragged_len > 0:
            dg_c[0, :, ragged_len - 1] += dg_last_reshaped[0]
        else:
            dg_c[:, :, C - 1] += dg_last_reshaped
        # [N, H, C] -> [N, C, H] -> [N*C, H]
        dg[b_idx, s0:s0 + N * C, :] = dg_c.to(gtype).permute(0, 2, 1).reshape(N * C, H)
        timer.stop("batched_dg_intra_accum")

        # ---- 4. dq/dk intra ----
        timer.start("batched_bmm_dqdk_intra")
        ds = cast_round(ds)
        dq_intra = cast_round(torch.bmm(ds, k_b))                                          # [N*H, C, K]
        dk_intra = cast_round(torch.bmm(ds.transpose(1, 2), q_b))                          # [N*H, C, K]
        timer.stop("batched_bmm_dqdk_intra")

        timer.start("batched_accumulate")
        dq_total = dq_state + dq_intra
        dk_total = dk_state + dk_intra
        timer.stop("batched_accumulate")

        # ---- 写回 dq_hv/dk_hv ----
        timer.start("batched_write_back")
        # [N*H, C, K] -> [N, H, C, K] -> [N, C, H, K] -> [N*C, H, K]
        if datatype == calc_type:
            dq_hv[b_idx, s0:s0 + N * C, :, :] = dq_total.reshape(N, H, C, K).permute(0, 2, 1, 3).reshape(N * C, H, K)
            dk_hv[b_idx, s0:s0 + N * C, :, :] = dk_total.reshape(N, H, C, K).permute(0, 2, 1, 3).reshape(N * C, H, K)
        else:
            dq_hv[b_idx, s0:s0 + N * C, :, :] = dq_total.to(datatype).reshape(N, H, C, K).permute(0, 2, 1, 3).reshape(N * C, H, K)
            dk_hv[b_idx, s0:s0 + N * C, :, :] = dk_total.to(datatype).reshape(N, H, C, K).permute(0, 2, 1, 3).reshape(N * C, H, K)
        timer.stop("batched_write_back")

        timer.stop("batched_total")

    # ------------------------------------------------------------------
    # 跨序列批量处理 ragged tails（varlen 模式下多个序列的尾部 chunk）。
    # 每个 tail 有不同的有效长度 ragged_i，pad 到 C 后批量计算。
    # padding 位置贡献为零（do=0/v=0），唯一需修正的是 dg_last_accum 位置：
    # 每个 tail 的 dg_last_accum 加到 ragged_i - 1 而非 C - 1。
    # ------------------------------------------------------------------
    def process_tails_batched(b_idx: int, tails: list):
        nonlocal chunk_count
        N = len(tails)
        C = chunk_size
        H = HV
        chunk_count += N
        timer.start("tails_total")

        # ---- 收集并 pad 每个 tail ----
        timer.start("tails_data_prep")
        q_list, k_list, v_list, do_list, dv_list, g_list, h_list, dh_list = [], [], [], [], [], [], [], []
        for ts, tr, h_idx in tails:
            te = ts + tr
            pad_len = C - tr
            q_list.append(torch.nn.functional.pad(q_ct[b_idx, :, ts:te, :], (0, 0, 0, pad_len)))
            k_list.append(torch.nn.functional.pad(k_ct[b_idx, :, ts:te, :], (0, 0, 0, pad_len)))
            v_list.append(torch.nn.functional.pad(v_ct[b_idx, :, ts:te, :], (0, 0, 0, pad_len)))
            do_list.append(torch.nn.functional.pad(do_ct[b_idx, :, ts:te, :], (0, 0, 0, pad_len)))
            dv_list.append(torch.nn.functional.pad(dv_ct[b_idx, :, ts:te, :], (0, 0, 0, pad_len)))
            g_t = g_ct[b_idx, :, ts:te]
            g_list.append(torch.cat([g_t, g_t[:, -1:].expand(-1, pad_len)], dim=1))
            h_list.append(h[b_idx, h_idx, :, :, :])
            dh_list.append(dh[b_idx, h_idx, :, :, :])

        # stack: [N, HK/HV, C, D] -> reshape [N*H, C, D]
        q_b = torch.stack(q_list, dim=0)  # [N, HK, C, K]
        if n_ratio > 1:
            q_b = q_b.repeat_interleave(n_ratio, dim=1)
        q_b = q_b.reshape(N * H, C, K)
        k_b = torch.stack(k_list, dim=0)
        if n_ratio > 1:
            k_b = k_b.repeat_interleave(n_ratio, dim=1)
        k_b = k_b.reshape(N * H, C, K)
        v_b = torch.stack(v_list, dim=0).reshape(N * H, C, V)
        do_b = torch.stack(do_list, dim=0).reshape(N * H, C, V)
        dv_b = torch.stack(dv_list, dim=0).reshape(N * H, C, V)
        g_b = torch.stack(g_list, dim=0).reshape(N * H, C)  # gtype
        h_b = torch.stack(h_list, dim=0).reshape(N * H, K, V)
        dh_b = torch.stack(dh_list, dim=0).reshape(N * H, K, V)
        h_b_t = h_b.transpose(-1, -2).to(calc_type)
        dh_b_t = dh_b.transpose(-1, -2).to(calc_type)
        timer.stop("tails_data_prep")

        # ---- Phase 1: 合并所有无依赖的 bmm（按 V/K 形状自动分组）----
        timer.start("tails_bmm_phase1")
        # V 组 A：右侧最后维为 K (h_b_t, dh_b_t 均为 [N*H, V, K])
        v_lhs_A = [do_b, v_b, dv_b]
        v_rhs_A = [h_b_t, dh_b_t, h_b_t]
        dq_state, dk_state, dw_c = stacked_bmm(v_lhs_A, v_rhs_A)
        # V 组 B：右侧最后维为 C (v_b^T 为 [N*H, V, C])，单独处理
        ds = stacked_bmm([do_b], [v_b.transpose(1, 2).contiguous()])[0]
        # K 组：lhs 最后一维为 K
        qk_t = stacked_bmm([q_b], [k_b.transpose(1, 2).contiguous()])[0]
        timer.stop("tails_bmm_phase1")

        timer.start("tails_mask")
        mask_f = get_causal_mask_f(C, q.device)
        timer.stop("tails_mask")

        # ---- 2. Decay scale ----
        timer.start("tails_decay_scale")
        g_last = g_b.reshape(N, H, C)[:, :, C - 1].reshape(N * H)
        exp_gc = torch.exp(g_b)
        exp_neg_gc_glast = torch.exp(-g_b + g_last[:, None])
        dq_state = dq_state * (exp_gc[:, :, None] * scale)
        dk_state = dk_state * exp_neg_gc_glast[:, :, None]
        timer.stop("tails_decay_scale")

        # ---- 3. Intra-chunk attention ----
        timer.start("tails_decay_apply")
        g_diff = g_b[:, :, None] - g_b[:, None, :]
        decay_mat = torch.clamp(g_diff, max=0).exp_()
        ds.mul_(decay_mat).mul_(mask_f).mul_(scale)
        timer.stop("tails_decay_apply")

        timer.start("tails_bmm_qk")
        ds2 = ds * qk_t
        dg_c = ds2.sum(dim=-1)
        dg_c = dg_c - ds2.sum(dim=-2)
        timer.stop("tails_bmm_qk")

        # ---- dg_state ----
        timer.start("tails_dg_state")
        dg_c = cast_round(dg_c)
        dg_c += (dq_state * q_b).sum(dim=-1)
        dg_c = cast_round(dg_c)
        k_dk_prod = k_b * dk_state
        dg_c = dg_c - k_dk_prod.sum(dim=-1)
        dg_last_accum = (h_b * dh_b).sum(dim=(-1, -2)) * torch.exp(g_last)
        dg_last_accum = dg_last_accum + k_dk_prod.sum(dim=(-1, -2))
        timer.stop("tails_dg_state")

        # ---- dg_last_accum 位置修正 + 写回 ----
        timer.start("tails_dg_write_back")
        dg_c = dg_c.reshape(N, H, C)
        dg_last_reshaped = dg_last_accum.reshape(N, H)
        for i, (ts, tr, _) in enumerate(tails):
            # 加到 ragged_i - 1（原始最后有效 token），而非 C - 1
            dg_c[i, :, tr - 1] += dg_last_reshaped[i]
            # 写回到原始位置 [ts, ts + tr)
            dg[b_idx, ts:ts + tr, :] = dg_c[i, :, :tr].to(gtype).permute(1, 0)
        timer.stop("tails_dg_write_back")

        # ---- 4. dq/dk intra ----
        timer.start("tails_bmm_dqdk_intra")
        ds = cast_round(ds)
        dq_intra = cast_round(torch.bmm(ds, k_b))
        dk_intra = cast_round(torch.bmm(ds.transpose(1, 2), q_b))
        timer.stop("tails_bmm_dqdk_intra")

        timer.start("tails_accumulate")
        dq_total = dq_state + dq_intra
        dk_total = dk_state + dk_intra
        timer.stop("tails_accumulate")

        # ---- 写回 dq_hv/dk_hv（只写有效长度部分）----
        timer.start("tails_write_back")
        dq_total = dq_total.reshape(N, H, C, K)
        dk_total = dk_total.reshape(N, H, C, K)
        if datatype == calc_type:
            for i, (ts, tr, _) in enumerate(tails):
                dq_hv[b_idx, ts:ts + tr, :, :] = dq_total[i, :, :tr, :].permute(1, 0, 2)
                dk_hv[b_idx, ts:ts + tr, :, :] = dk_total[i, :, :tr, :].permute(1, 0, 2)
        else:
            for i, (ts, tr, _) in enumerate(tails):
                dq_hv[b_idx, ts:ts + tr, :, :] = dq_total[i, :, :tr, :].to(datatype).permute(1, 0, 2)
                dk_hv[b_idx, ts:ts + tr, :, :] = dk_total[i, :, :tr, :].to(datatype).permute(1, 0, 2)
        # 写回 dw（只写有效长度部分）
        dw_c = dw_c.reshape(N, H, C, K)
        for i, (ts, tr, _) in enumerate(tails):
            dw[b_idx, ts:ts + tr, :, :] = (-dw_c[i, :, :tr, :]).permute(1, 0, 2)
        timer.stop("tails_write_back")

        timer.stop("tails_total")

    # ------------------------------------------------------------------
    # 合并多个序列的满 chunk 为一次大 batched 调用。
    # 不同序列的满 chunk token 范围不连续，需要 cat 拼接后统一处理，
    # 完成后拆分写回到各自的原始位置。
    # ------------------------------------------------------------------
    def process_varlen_merged(b_idx: int, full_ranges: list):
        nonlocal chunk_count
        C = chunk_size
        H = HV
        total_N = sum(nf for _, nf, _ in full_ranges)
        chunk_count += total_N
        timer.start("merged_total")

        # ---- 数据准备：cat 拼接所有序列的满 chunk ----
        timer.start("merged_data_prep")
        q_cats = [q_ct[b_idx, :, s:s + nf * C, :] for s, nf, _ in full_ranges]
        k_cats = [k_ct[b_idx, :, s:s + nf * C, :] for s, nf, _ in full_ranges]
        v_cats = [v_ct[b_idx, :, s:s + nf * C, :] for s, nf, _ in full_ranges]
        do_cats = [do_ct[b_idx, :, s:s + nf * C, :] for s, nf, _ in full_ranges]
        dv_cats = [dv_ct[b_idx, :, s:s + nf * C, :] for s, nf, _ in full_ranges]
        g_cats = [g_ct[b_idx, :, s:s + nf * C] for s, nf, _ in full_ranges]
        h_cats = [h[b_idx, ho:ho + nf, :, :, :] for _, nf, ho in full_ranges]
        dh_cats = [dh[b_idx, ho:ho + nf, :, :, :] for _, nf, ho in full_ranges]

        # 拼接后按 chunk 重排
        # q/k: [HK, total_N*C, K] -> [HK, total_N, C, K] -> [total_N, HK, C, K]
        q_all = torch.cat(q_cats, dim=1).reshape(HK, total_N, C, K).permute(1, 0, 2, 3)
        if n_ratio > 1:
            q_all = q_all.repeat_interleave(n_ratio, dim=1)
        q_b = q_all.reshape(total_N * H, C, K)

        k_all = torch.cat(k_cats, dim=1).reshape(HK, total_N, C, K).permute(1, 0, 2, 3)
        if n_ratio > 1:
            k_all = k_all.repeat_interleave(n_ratio, dim=1)
        k_b = k_all.reshape(total_N * H, C, K)

        v_b = torch.cat(v_cats, dim=1).reshape(H, total_N, C, V).permute(1, 0, 2, 3).reshape(total_N * H, C, V)
        do_b = torch.cat(do_cats, dim=1).reshape(H, total_N, C, V).permute(1, 0, 2, 3).reshape(total_N * H, C, V)
        dv_b = torch.cat(dv_cats, dim=1).reshape(H, total_N, C, V).permute(1, 0, 2, 3).reshape(total_N * H, C, V)
        g_b = torch.cat(g_cats, dim=1).reshape(H, total_N, C).permute(1, 0, 2).reshape(total_N * H, C)

        h_b = torch.cat(h_cats, dim=0).reshape(total_N * H, K, V)
        dh_b = torch.cat(dh_cats, dim=0).reshape(total_N * H, K, V)
        h_b_t = h_b.transpose(-1, -2).to(calc_type)
        dh_b_t = dh_b.transpose(-1, -2).to(calc_type)
        timer.stop("merged_data_prep")

        N = total_N

        # ---- Phase 1: 合并所有无依赖的 bmm（按 V/K 形状自动分组）----
        timer.start("merged_bmm_phase1")
        # V 组 A：右侧最后维为 K (h_b_t, dh_b_t 均为 [N*H, V, K])
        v_lhs_A = [do_b, v_b, dv_b]
        v_rhs_A = [h_b_t, dh_b_t, h_b_t]
        dq_state, dk_state, dw_c = stacked_bmm(v_lhs_A, v_rhs_A)
        # V 组 B：右侧最后维为 C (v_b^T 为 [N*H, V, C])，单独处理
        ds = stacked_bmm([do_b], [v_b.transpose(1, 2).contiguous()])[0]
        # K 组：lhs 最后一维为 K
        qk_t = stacked_bmm([q_b], [k_b.transpose(1, 2).contiguous()])[0]
        timer.stop("merged_bmm_phase1")

        timer.start("merged_mask")
        mask_f = get_causal_mask_f(C, q.device)
        timer.stop("merged_mask")

        # ---- 2. Decay scale ----
        timer.start("merged_decay_scale")
        g_last = g_b.reshape(N, H, C)[:, :, C - 1].reshape(N * H)
        exp_gc = torch.exp(g_b)
        exp_neg_gc_glast = torch.exp(-g_b + g_last[:, None])
        dq_state = dq_state * (exp_gc[:, :, None] * scale)
        dk_state = dk_state * exp_neg_gc_glast[:, :, None]
        timer.stop("merged_decay_scale")

        # ---- 3. Intra-chunk attention ----
        timer.start("merged_decay_apply")
        g_diff = g_b[:, :, None] - g_b[:, None, :]
        decay_mat = torch.clamp(g_diff, max=0).exp_()
        ds.mul_(decay_mat).mul_(mask_f).mul_(scale)
        timer.stop("merged_decay_apply")

        timer.start("merged_bmm_qk")
        ds2 = ds * qk_t
        dg_c = ds2.sum(dim=-1)
        dg_c = dg_c - ds2.sum(dim=-2)
        timer.stop("merged_bmm_qk")

        # ---- dg_state ----
        timer.start("merged_dg_state")
        dg_c = cast_round(dg_c)
        dg_c += (dq_state * q_b).sum(dim=-1)
        dg_c = cast_round(dg_c)
        k_dk_prod = k_b * dk_state
        dg_c = dg_c - k_dk_prod.sum(dim=-1)
        dg_last_accum = (h_b * dh_b).sum(dim=(-1, -2)) * torch.exp(g_last)
        dg_last_accum = dg_last_accum + k_dk_prod.sum(dim=(-1, -2))
        timer.stop("merged_dg_state")

        # ---- dg_last_accum + 写回 dg（按序列拆分）----
        timer.start("merged_dg_write_back")
        dg_c = dg_c.reshape(N, H, C)
        dg_last_reshaped = dg_last_accum.reshape(N, H)
        dg_c[:, :, C - 1] += dg_last_reshaped
        # 拆分写回到各自的原始位置
        offset = 0
        for s, nf, _ in full_ranges:
            n_tokens = nf * C
            dg_chunk = dg_c[offset:offset + nf].to(gtype).permute(0, 2, 1).reshape(n_tokens, H)
            dg[b_idx, s:s + n_tokens, :] = dg_chunk
            offset += nf
        timer.stop("merged_dg_write_back")

        # ---- 4. dq/dk intra ----
        timer.start("merged_bmm_dqdk_intra")
        ds = cast_round(ds)
        dq_intra = cast_round(torch.bmm(ds, k_b))
        dk_intra = cast_round(torch.bmm(ds.transpose(1, 2), q_b))
        timer.stop("merged_bmm_dqdk_intra")

        timer.start("merged_accumulate")
        dq_total = dq_state + dq_intra
        dk_total = dk_state + dk_intra
        timer.stop("merged_accumulate")

        # ---- 写回 dq_hv/dk_hv/dw（按序列拆分）----
        timer.start("merged_write_back")
        dq_total = dq_total.reshape(N, H, C, K)
        dk_total = dk_total.reshape(N, H, C, K)
        dw_c = dw_c.reshape(N, H, C, K)
        offset = 0
        for s, nf, _ in full_ranges:
            n_tokens = nf * C
            if datatype == calc_type:
                dq_hv[b_idx, s:s + n_tokens, :, :] = dq_total[offset:offset + nf].permute(0, 2, 1, 3).reshape(n_tokens, H, K)
                dk_hv[b_idx, s:s + n_tokens, :, :] = dk_total[offset:offset + nf].permute(0, 2, 1, 3).reshape(n_tokens, H, K)
            else:
                dq_hv[b_idx, s:s + n_tokens, :, :] = dq_total[offset:offset + nf].to(datatype).permute(0, 2, 1, 3).reshape(n_tokens, H, K)
                dk_hv[b_idx, s:s + n_tokens, :, :] = dk_total[offset:offset + nf].to(datatype).permute(0, 2, 1, 3).reshape(n_tokens, H, K)
            dw[b_idx, s:s + n_tokens, :, :] = (-dw_c[offset:offset + nf]).permute(0, 2, 1, 3).reshape(n_tokens, H, K)
            offset += nf
        timer.stop("merged_write_back")

        timer.stop("merged_total")

    # ------------------------------------------------------------------
    # 子批量调度：将 n_full 个满 chunk 按 _MAX_DENSE_BATCH_CHUNKS 拆分为多次
    # process_dense_batched 调用。chunk 间无数据依赖（h/dh 只读，dq_hv 等只
    # 写不跨 chunk 读），故拆分不改变结果。仅最后一个子批量保留 ragged_len
    # 修正（padded tail 的 dg_last_accum 位置）。
    # ------------------------------------------------------------------
    def _run_dense_subbatched(b_idx, n_full, t_offset, h_offset, ragged_len=0):
        remaining = n_full
        s = t_offset
        ho = h_offset
        if n_full > _MAX_DENSE_BATCH_CHUNKS and verbose_timing:
            n_sub = (n_full + _MAX_DENSE_BATCH_CHUNKS - 1) // _MAX_DENSE_BATCH_CHUNKS
            print(f"[chunk_bwd_dqkwg_cpu] sub-batching: n_full={n_full} > max={_MAX_DENSE_BATCH_CHUNKS}, "
                  f"splitting into {n_sub} sub-batches")
        while remaining > 0:
            sub_n = min(remaining, _MAX_DENSE_BATCH_CHUNKS)
            is_last = (sub_n == remaining)
            sub_ragged = ragged_len if is_last else 0
            process_dense_batched(b_idx, sub_n, s, ho, ragged_len=sub_ragged)
            s += sub_n * C
            ho += sub_n
            remaining -= sub_n

    # Main Loop
    mode = "varlen" if cu_seqlens is not None else "dense"
    if verbose_timing:
        print(f"[chunk_bwd_dqkwg_cpu] start: B={B} T={T} T_orig={T_orig} ragged_len={ragged_len} "
              f"HK={HK} HV={HV} K={K} V={V} "
              f"n_ratio={n_ratio} chunk_size={chunk_size} mode={mode} benchmark={benchmark}")
    # 小矩阵 bmm 场景：多线程 BLAS 的 spawn/join 开销远超实际计算。
    # 降为 1 线程消除开销，单线程 BLAS 对小矩阵 cache 更友好。
    _saved_num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    timer.start("main_loop")
    if cu_seqlens is None:
        # T 已 padding 到 chunk_size 的整数倍，所有 chunk 统一走 batched 路径
        for b in range(B):
            n_full = T // C
            if n_full > 0:
                _run_dense_subbatched(b, n_full, 0, 0, ragged_len=ragged_len)
    else:
        # Variable length B=1
        # 计算每个序列的 chunk 起始位置
        chunk_location_list = [0]
        seq_infos = []
        for i in range(len(cu_seqlens) - 1):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i+1].item()
            seq_len = end - start
            n_chunks = (seq_len + C - 1) // C
            seq_infos.append((start, seq_len, chunk_location_list[-1]))
            chunk_location_list.append(chunk_location_list[-1] + n_chunks)

        # Phase 1: 合并所有序列的满 chunk 为一次 batched 调用
        # 收集每个序列的满 chunk token 范围和 h/dh 偏移
        full_ranges = []
        for start, seq_len, h_off in seq_infos:
            n_full = seq_len // C
            if n_full > 0:
                full_ranges.append((start, n_full, h_off))

        total_n_full = sum(nf for _, nf, _ in full_ranges)
        if total_n_full <= _MAX_DENSE_BATCH_CHUNKS:
            if len(full_ranges) == 1:
                # 只有一个序列有满 chunk，直接调用
                ts, nf, th = full_ranges[0]
                process_dense_batched(0, nf, ts, th, ragged_len=0)
            elif len(full_ranges) > 1:
                # 合并多个序列的满 chunk 为一次大 batched 调用
                process_varlen_merged(0, full_ranges)
        else:
            # 总 chunk 数过多，按序列分别走 sub-batch 避免内存爆炸
            for ts, nf, th in full_ranges:
                _run_dense_subbatched(0, nf, ts, th, ragged_len=0)

        # Phase 2: 跨序列批量处理 ragged tails
        tails = []
        for start, seq_len, h_off in seq_infos:
            n_full = seq_len // C
            ragged = seq_len % C
            if ragged > 0:
                tails.append((start + n_full * C, ragged, h_off + n_full))

        if len(tails) == 1:
            ts, tr, th = tails[0]
            process_sequence(0, ts, ts + tr, 0, th)
        elif len(tails) > 1:
            if len(tails) <= _MAX_DENSE_BATCH_CHUNKS:
                process_tails_batched(0, tails)
            else:
                # tail 数量过多，逐个处理避免内存爆炸
                for ts, tr, th in tails:
                    process_sequence(0, ts, ts + tr, 0, th)
    timer.stop("main_loop")

    timer.start("reduce")
    dq = dq_hv.view(B, T, HK, n_ratio, K).sum(dim=3).to(datatype)
    dk = dk_hv.view(B, T, HK, n_ratio, K).sum(dim=3).to(datatype)
    timer.stop("reduce")

    # 截断到原始 T（去除 padding 部分）
    if pad_T > 0:
        dq = dq[:, :T_orig]
        dk = dk[:, :T_orig]
        dw = dw[:, :T_orig]
        dg = dg[:, :T_orig]

    t_total = time.perf_counter() - t_total_start
    summary = timer.summary(t_total, chunk_count)
    if summary:
        print(summary)

    torch.set_num_threads(_saved_num_threads)
    return dq, dk, dw, dg

def prepare_chunk_indices(
    cu_seqlens: list[int],
    chunk_size: int
) -> list[int]: 
    """
    基于 cu_seqlens (list[int]) 生成 chunk 索引。
    
    注意：原 PyTorch 版本返回的是 shape [N, 2] 的 Tensor。
    为了保持纯 Python 兼容性，这里返回 list[tuple[start_seq_idx, chunk_idx_in_seq]]。
    如果算子需要扁平化的 list[int] (如 [s0, c0, s1, c1, ...])，请在调用前展开。
    
    逻辑复刻原代码：
    1. 计算每个序列的长度: lens[i] = cu_seqlens[i+1] - cu_seqlens[i]
    2. 计算每个序列需要的 chunk 数: ceil(lens[i] / chunk_size)
    3. 生成对应的 (sequence_id, chunk_id) 对
    """
    indices = []
    
    # 遍历每个序列段
    for i in range(len(cu_seqlens) - 1):
        start = cu_seqlens[i]
        end = cu_seqlens[i+1]
        length = end - start
        
        if length <= 0:
            continue
            
        # 计算该序列需要多少个 chunk
        num_chunks = (length + chunk_size - 1) // chunk_size
        
        for chunk_id in range(num_chunks):
            # 原逻辑: indices.eq(0).cumsum(0) - 1 对应的是序列索引 i
            # 原逻辑: indices 对应的是 chunk_id
            indices.append((i))
            indices.append((chunk_id))
            
    return indices

# -------------------------------------------------------------------------
# 使用示例 / 验证
# -------------------------------------------------------------------------
if __name__ == "__main__":
    RANDOM_DATA = True
    torch.manual_seed(1)
    case_number = 21
    if len(sys.argv) > 1:
        regen = sys.argv[1]
        if regen == "random":
            print("[test.py] regenerate all random data!")
            RANDOM_DATA=True

    # 简单的形状参数
    K, V = 128, 128
    calc_type = torch.float32
    isVarLen = False
    chunk_size = 128
    cases = [   #B,HK,HV,T,chunk_size,dtype,Gtype,scale,cu_seqlens
        [64,8,8,1024,64,torch.float16,torch.float16,0.088,None],
        [32,16,16,2048,64,torch.bfloat16,torch.bfloat16,0.0625,None],
        [16,32,32,4096,64,torch.float16,torch.float16,0.0442,None],
        [8,32,32,8192,64,torch.bfloat16,torch.bfloat16,0.03125,None],
        [128,4,4,1024,64,torch.float16,torch.float16,0.088,None],
        [64,4,4,4096,128,torch.bfloat16,torch.bfloat16,0.0625,None],
        [32,16,16,8192,64,torch.float16,torch.float16,0.0442,None],
        [16,32,32,16384,64,torch.bfloat16,torch.bfloat16,0.03125,None],
        [64,8,8,2048,128,torch.float16,torch.float16,0.0625,None],
        [32,16,16,4096,128,torch.bfloat16,torch.bfloat16,0.0442,None],
        [16,32,32,8192,128,torch.float16,torch.float16,0.03125,None],
        [8,32,32,8192,128,torch.bfloat16,torch.bfloat16,0.0221,None],  #C12
        [1,4,4,1024,64,torch.float16,torch.float16,0.088,None],
        [48,8,8,2048,64,torch.bfloat16,torch.bfloat16,0.0625,None],
        [24,16,16,4096,64,torch.float16,torch.float16,0.0442,None],
        [12,32,32,8192,64,torch.bfloat16,torch.bfloat16,0.03125,None],
        [1,16,16,32768,64,torch.float16,torch.float32,0.0625,torch.tensor([0,16,20000,30000,32768])],      # V1
        [1,8,8,65536,64,torch.bfloat16,torch.bfloat16,0.0625,torch.tensor([0,16,20000,65536])],
        [1,32,32,65536,64,torch.float16,torch.float32,0.0442,torch.tensor([0,16,20000,50000,65536])],
        [1,32,32,262144,64,torch.bfloat16,torch.bfloat16,0.03125,torch.tensor([0,16,20000,50000,65536,210000,262144])],
        [8,8,8,4096,64,torch.float16,torch.float16,0.088,None],  #21 [0,16,128] [0,16,135,512]
        [1,32,32,16384,64,torch.bfloat16,torch.float32,0.088,None],  #21 [0,16,128]
    ]
    device_id = int(os.environ.get("TEST_DEVICE_ID", 0))
    

    dtype = torch.float16
    Gtype = torch.float16
    B, HK, HV = 4, 8, 8
    T = 1024
    scale = 0.088
    if isVarLen:
        cu_seqlens = torch.cumsum(torch.tensor([0, 3, 64, 63, 66, 260]), dim=0)
    else:
        cu_seqlens = None
    if case_number != -1:
        single_case = cases[case_number-1]  #case_01 => cases[0]
        dtype = single_case[5]
        Gtype = single_case[6]
        B, HK, HV = single_case[0], single_case[1], single_case[2]
        chunk_size = single_case[4]
        cu_seqlens = single_case[8]
        cu_seqlens_torch = torch.tensor(cu_seqlens) if cu_seqlens is not None else None

        if single_case[8] is None:
            isVarLen = False
        else:
            isVarLen = True
        T = single_case[3]
        scale = single_case[7]

    if isVarLen:
        B = 1  ##变长只支持B=1
        T = cu_seqlens_torch[-1]
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        num_chunks = len(chunk_indices) // 2
        print("chunk_indices",chunk_indices)
    else:
        chunk_indices = None
        num_chunks = (T + chunk_size - 1) // chunk_size
    
    if RANDOM_DATA:
        q = torch.randn(B,T,HK,K, dtype=dtype) * 5e-2 # std≈5e-6#torch.randn([B, T, H, K], dtype=dtype)
        k = torch.randn(B,T,HK,K, dtype=dtype) * 5e-2 # torch.randn([B, T, H, K], dtype=dtype)
        v = torch.randn(B,T,HV,V, dtype=dtype) * 5e-2 # torch.randn([B, T, H, V], dtype=dtype)

        g = -torch.sort(torch.rand(B*T*HV), descending=False)[0].reshape((B,T,HV)).to(Gtype)    #G必须递减且为负数
        # print("g",g)
        do = torch.randn(B,T,HV,V, dtype=dtype) * 5e-2 # torch.randn([B, T, H, V], dtype=dtype)

        dv = torch.randn(B,T,HV,V, dtype=dtype) * 5e-1 # torch.randn([B, T, H, V], dtype=dtype)
        w = torch.randn(B,T,HV,K, dtype=dtype) * 5e-2 # torch.randn([B, T, H, K], dtype=dtype)

        h = torch.randn(B, num_chunks, HV, K, V, dtype=dtype) * 5e-2  # torch.randn([B, num_chunks, H, K, V], dtype=dtype)
        dh = torch.randn(B, num_chunks, HV, K, V, dtype=dtype) * 5e-2 # torch.randn([B, num_chunks, H, K, V], dtype=dtype)

    q = q.to(dtype).to(calc_type)
    k = k.to(dtype).to(calc_type)
    v = v.to(dtype).to(calc_type)
    h = h.to(dtype).to(calc_type)
    g = g.to(Gtype).to(calc_type)
    do = do.to(dtype).to(calc_type)
    dh = dh.to(dtype).to(calc_type)
    dv = dv.to(dtype).to(calc_type)
    w = w.to(dtype).to(calc_type)
    print("entering chunk_bwd_dqkwg")
    print(f"q: {q.shape} {dtype} => {q.dtype}")
    print(f"k: {k.shape} {dtype} => {k.dtype}")
    print(f"v: {v.shape} {dtype} => {v.dtype}")
    print(f"w: {w.shape} {dtype} => {w.dtype}")
    print(f"g: {g.shape} {Gtype} => {g.dtype}")
    print(f"h: {h.shape} {dtype} => {h.dtype}")
    print(f"dv: {dv.shape} {dtype} => {dv.dtype}")
    print(f"do: {do.shape} {dtype} => {do.dtype}")
    print(f"dh: {dh.shape} {dtype} => {dh.dtype}")
    if cu_seqlens == None:
        print("cu_seqlens is None")
    else:
        print(f"cu_seqlens: {cu_seqlens_torch.shape} {cu_seqlens_torch.dtype} {cu_seqlens_torch}")
        # print(f"chunk_indices: {chunk_indices.shape} {chunk_indices.dtype} {chunk_indices}")
    print(f"scale: {scale}")
    print(f"chunk_size: {chunk_size}")


    print("==============start NPU=============")
    torch.npu.set_device(device_id)
    print("dtype")
    q_npu = torch.transpose(q, 1, 2).to(dtype).npu()
    print("q_npu", q_npu.shape, q_npu.dtype)
    k_npu = torch.transpose(k, 1, 2).to(dtype).npu()
    print("k_npu", k_npu.shape, k_npu.dtype)
    v_npu = torch.transpose(v, 1, 2).to(dtype).npu()
    print("v_npu", v_npu.shape, v_npu.dtype)
    w_npu = torch.transpose(w, 1, 2).to(dtype).npu()
    print("w_npu", w_npu.shape, w_npu.dtype)
    g_npu = torch.transpose(g, 1, 2).to(Gtype).npu()
    print("g_npu", g_npu.shape, g_npu.dtype)
    h_npu = torch.transpose(h, 1, 2).to(dtype).npu()
    print("h_npu", h_npu.shape, h_npu.dtype)
    dv_npu = torch.transpose(dv, 1, 2).to(dtype).npu()
    print("dv_npu", dv_npu.shape, dv_npu.dtype)
    do_npu = torch.transpose(do, 1, 2).to(dtype).npu()
    print("do_npu", do_npu.shape, do_npu.dtype)
    dh_npu = torch.transpose(dh, 1, 2).to(dtype).npu()
    print("dh_npu", dh_npu.shape, dh_npu.dtype)
    # cu_seqlens_npu = cu_seqlens if cu_seqlens is not None else None
    chunk_indices_npu = chunk_indices if cu_seqlens is not None else None
    print("chunk_indices_npu")
    down_tri = q_npu

    print("qqqqqqqq")
    dq_npu, dk_npu, dw_npu, dg_npu = ascendc_ops.npu_chunk_bwd_dqkwg(
        q_npu, k_npu, v_npu, g_npu, h_npu, do_npu, dh_npu, dv_npu, chunk_size, cu_seqlens=cu_seqlens, w=None, g_gamma=None, chunk_indices=chunk_indices_npu, scale=scale, use_exp2=None, transpose_state_layout=None
    )
    print("custom_ops.npu_chunk_bwd_dqkwg done")
    dq_npu = dq_npu.cpu()
    dk_npu = dk_npu.cpu()
    dw_npu = dw_npu.cpu()
    dg_npu = dg_npu.cpu()

    # print("Output shapes:", dq.shape, dk.shape, dg.shape, dw.shape)
    print("dq_npu", dq_npu.shape, dq_npu.dtype)
    print("dk_npu", dk_npu.shape, dk_npu.dtype)
    print("dw_npu", dw_npu.shape, dw_npu.dtype)
    print("dg_npu", dg_npu.shape, dg_npu.dtype)

    # print("dq_npu[0][0][-1]", dq_npu[0][0][-1])

    print("=====start cpu=========")


    dq, dk, dw, dg = chunk_bwd_dqkwg_cpu(
        q, k, v, do, h, dh, w, g, dv, scale, cu_seqlens_torch, chunk_size
    )
    # dq = dq.to(dtype)
    # dk = dk.to(dtype)
    # dw = dw.to(dtype)
    # dg = dg.to(Gtype)
    dq = torch.transpose(dq, 1, 2).cpu()
    dk = torch.transpose(dk, 1, 2).cpu()
    dw = torch.transpose(dw, 1, 2).cpu()
    dg = torch.transpose(dg, 1, 2).cpu()
    # print("dq[0][0][-1]", dq[0][0][-1])
    # print("dk", dk)
    # print("dw", dw)
    # print("dg", dg)

    print("dq", dq.cpu().shape, dq.cpu().dtype)
    print("dk", dk.cpu().shape, dk.cpu().dtype)
    print("dw", dw.cpu().shape, dw.cpu().dtype)
    print("dg", dg.cpu().shape, dg.cpu().dtype)

    print("=====compare dq/dk/dw/dg=====")
    rtol, atol = 1e-2, 1e-2
    for name, cpu_val, npu_val in [
        ("dq", dq, dq_npu),
        ("dk", dk, dk_npu),
        ("dw", dw, dw_npu),
        ("dg", dg, dg_npu),
    ]:
        cpu_f = cpu_val.float()
        npu_f = npu_val.float()
        diff = (cpu_f - npu_f).abs()
        max_abs = diff.max().item()
        flat_idx = int(diff.argmax().item())
        max_abs_coord = tuple(int(v) for v in torch.unravel_index(torch.tensor(flat_idx), diff.shape))
        significant = cpu_f.abs() > atol
        max_rel = (diff[significant] / cpu_f[significant].abs()).max().item() if significant.any() else 0.0
        allclose = torch.allclose(npu_f, cpu_f, rtol=rtol, atol=atol)
        status = "PASS" if allclose else "FAIL"
        print(f"[{status}] {name}: shape={tuple(cpu_val.shape)} max_abs={max_abs:.6e} max_rel={max_rel:.6e} max_abs_coord={max_abs_coord}")
        if not allclose:
            print(f"cpu_val={cpu_f.flatten()[flat_idx].item():.8e} npu_val={npu_f.flatten()[flat_idx].item():.8e}")

    print("All done!")
