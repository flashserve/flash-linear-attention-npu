import torch
import torch_npu
import os
from typing import Tuple

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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CPU Equivalent of chunk_bwd_kernel_dqkwg.

    优化说明：将原按 HV 头展开的 Python 循环改为对 HV 维做批量 (bmm) 计算。
    CPU 上 bmm 对每个 batch 调用与单次 mm 相同的 gemm，因此每个头的矩阵乘
    与原循环逐位等价；所有 dtype 往返 (.to(datatype).to(calc_type)) 和运算顺序
    均原样保留，维持与 kernel 一致的精度模拟。
    """
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

    q.to(calc_type)
    k.to(calc_type)
    v.to(calc_type)
    do.to(calc_type)
    h.to(calc_type)
    dh.to(calc_type)
    
    g.to(gtype).to(calc_type)
    dv.to(calc_type)

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
    q_ct = q.permute(0, 2, 1, 3).to(calc_type).contiguous()   # [B, HK, T, K]
    k_ct = k.permute(0, 2, 1, 3).to(calc_type).contiguous()   # [B, HK, T, K]
    v_ct = v.permute(0, 2, 1, 3).to(calc_type).contiguous()   # [B, HV, T, V]
    do_ct = do.permute(0, 2, 1, 3).to(calc_type).contiguous() # [B, HV, T, V]
    dv_ct = dv.permute(0, 2, 1, 3).to(calc_type).contiguous() # [B, HV, T, V]
    g_ct = g.permute(0, 2, 1).contiguous()                    # [B, HV, T], gtype

    # Keep per-value-head contributions first, then reduce them into key heads.
    dq_hv = torch.zeros((B, T, HV, K), dtype=datatype)
    dk_hv = torch.zeros((B, T, HV, K), dtype=datatype)
    dg = torch.zeros_like(g)
    dw = torch.zeros((B, T, HV, K), dtype=datatype)

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

    def process_sequence(b_idx, t_start, t_end, seq_idx_in_batch, chunk_start_idx):
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

            # ---- 取当前 chunk 全部头的数据，统一以 head 作为 batch 维 ----
            # q/k: [L, HK, K] -> [HK, L, K] -> 复制 n_ratio 份 -> [HV, L, K]
            #   head h_idx 对应 hk_idx = h_idx // n_ratio（与原循环一致）
            q_h = expand_heads(q_ct[b_idx, :, s:e, :])
            k_h = expand_heads(k_ct[b_idx, :, s:e, :])
            v_h = v_ct[b_idx, :, s:e, :]
            do_h = do_ct[b_idx, :, s:e, :]
            # h/dh: [HV, K, V]（保留原始 datatype，供 dg_last_accum 使用）
            h_prev = h[b_idx, i_t + chunk_start_idx, :, :, :]       # [HV, K, V]
            dh_curr = dh[b_idx, i_t + chunk_start_idx, :, :, :]     # [HV, K, V]
            h_prev_t = h_prev.transpose(-1, -2).to(calc_type)       # [HV, V, K]
            dh_curr_t = dh_curr.transpose(-1, -2).to(calc_type)     # [HV, V, K]

            # -----------------------------------------------------------
            # 1. State Contributions (Inter-chunk)
            # -----------------------------------------------------------
            # b_dq += dot(b_do, b_h); b_dk += dot(b_v, b_dh)
            dq_from_state = cast_round(torch.bmm(do_h, h_prev_t))   # [HV, L, K]
            dk_from_state = cast_round(torch.bmm(v_h, dh_curr_t))   # [HV, L, K]

            # b_dw += dot(b_dv, b_h) (kernel 存 -b_dw)
            dv_h = dv_ct[b_idx, :, s:e, :]
            dw_c = cast_round(torch.bmm(dv_h, h_prev_t))       # [HV, L, K]
            dw[b_idx, s:e, :, :] = (-dw_c).permute(1, 0, 2)

            mask_f = get_causal_mask_f(L, q.device)

            # -----------------------------------------------------------
            # 2. Gating / Decay Logic Preparation
            # -----------------------------------------------------------
            g_h = g_ct[b_idx, :, s:e]                            # [HV, L] (gtype)
            g_last = g_ct[b_idx, :, min(s + chunk_size, t_end) - 1]  # [HV]

            exp_gc = torch.exp(g_h)                              # [HV, L]
            exp_neg_gc_glast = torch.exp(-g_h + g_last[:, None]) # [HV, L]

            dq_from_state = dq_from_state * (exp_gc[:, :, None] * scale)
            dk_from_state = dk_from_state * exp_neg_gc_glast[:, :, None]

            # -----------------------------------------------------------
            # 3. Intra-chunk Attention
            # -----------------------------------------------------------
            ds = cast_round(torch.bmm(do_h, v_h.transpose(1, 2)))  # [HV, L, L]
            decay_mat = torch.exp(torch.min(g_h[:, :, None] - g_h[:, None, :], torch.tensor(0)))  # [HV, L, L]
            # 融合：避免 where(scalar) 类型提升与多次临时分配
            ds = ds * decay_mat
            ds = ds * mask_f                  # [1, L, L] 广播掩码
            ds = ds * scale

            qk_t = cast_round(torch.bmm(q_h, k_h.transpose(1, 2)))   # [HV, L, L]

            ds2 = ds * qk_t
            dg_c = ds2.sum(dim=-1)
            dg_c = dg_c - ds2.sum(dim=-2)
            if datatype == gtype:
                dg_c = dg_c.to(gtype)                               # 等价 .to(datatype).to(gtype)
            else:
                dg_c = dg_c.to(datatype).to(gtype)                  # [HV, L]

            # b_dg += sum(b_dq * b_q) ; b_dg -= sum(b_k * b_dk)
            # 保留显式 product+sum（与原参考一致，避免 einsum 改变归约顺序）
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

            # b_ds2 = b_ds * (q @ k.T)
            # 仅块最后一个有效 token 累加 dg_last
            dg_c[:, L - 1] = dg_c[:, L - 1] + dg_last_accum
            dg[b_idx, s:e, :] = dg_c.to(gtype).permute(1, 0)                   # [L, HV]

            # -----------------------------------------------------------
            # 4. Final Accumulation for dq, dk
            # -----------------------------------------------------------
            ds = cast_round(ds)
            dq_intra = cast_round(torch.bmm(ds, k_h))               # [HV, L, K]
            dk_intra = cast_round(torch.bmm(ds.transpose(1, 2), q_h))# [HV, L, K]

            dq_total = dq_from_state + dq_intra
            dk_total = dk_from_state + dk_intra

            if datatype == calc_type:
                dq_hv[b_idx, s:e, :, :] = dq_total.permute(1, 0, 2)
                dk_hv[b_idx, s:e, :, :] = dk_total.permute(1, 0, 2)
            else:
                dq_hv[b_idx, s:e, :, :] = dq_total.to(datatype).permute(1, 0, 2)
                dk_hv[b_idx, s:e, :, :] = dk_total.to(datatype).permute(1, 0, 2)

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
        N = n_full
        if N <= 0:
            return
        C = chunk_size
        H = HV
        s0 = t_offset

        # ---- 数据准备：从 pre-transposed tensors 切片，无需 permute+cast ----
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

        # ---- 1. State contributions ----
        dq_state = cast_round(torch.bmm(do_b, h_b_t))             # [N*H, C, K]
        dk_state = cast_round(torch.bmm(v_b, dh_b_t))            # [N*H, C, K]

        dv_b = dv_ct[b_idx, :, s0:s0 + N * C, :].reshape(H, N, C, V).permute(1, 0, 2, 3).reshape(N * H, C, V)
        dw_c = cast_round(torch.bmm(dv_b, h_b_t))            # [N*H, C, K]
        # [N*H, C, K] -> [N, H, C, K] -> [N, C, H, K] -> [N*C, H, K]
        dw_neg = (-dw_c).reshape(N, H, C, K).permute(0, 2, 1, 3).reshape(N * C, H, K)
        dw[b_idx, s0:s0 + N * C, :, :] = dw_neg

        mask_f = get_causal_mask_f(C, q.device)                  # [1, C, C]

        # ---- 2. Decay scale ----
        # g: pre-transposed [B, HV, T] -> [HV, N*C] -> [HV, N, C] -> [N, HV, C] -> [N*HV, C]
        g_b = g_ct[b_idx, :, s0:s0 + N * C].reshape(H, N, C).permute(1, 0, 2).reshape(N * H, C)  # gtype
        # 每个 chunk 的最后有效 token = chunk 内第 C-1 个位置（满 chunk 下成立）
        g_last = g_b.reshape(N, H, C)[:, :, C - 1].reshape(N * H)                          # [N*H]
        exp_gc = torch.exp(g_b)                                                             # [N*H, C]
        exp_neg_gc_glast = torch.exp(-g_b + g_last[:, None])                               # [N*H, C]
        dq_state = dq_state * (exp_gc[:, :, None] * scale)
        dk_state = dk_state * exp_neg_gc_glast[:, :, None]

        # ---- 3. Intra-chunk attention ----
        ds = cast_round(torch.bmm(do_b, v_b.transpose(1, 2)))                              # [N*H, C, C]
        # 用 clamp 替代 min(tensor, scalar)，避免每 chunk 创建标量张量
        # in-place exp_ 节省一次 [N*H, C, C] 分配
        g_diff = g_b[:, :, None] - g_b[:, None, :]
        decay_mat = torch.clamp(g_diff, max=0).exp_()                                      # [N*H, C, C]
        ds = ds * decay_mat
        ds = ds * mask_f
        ds = ds * scale

        qk_t = cast_round(torch.bmm(q_b, k_b.transpose(1, 2)))                             # [N*H, C, C]
        ds2 = ds * qk_t
        dg_c = ds2.sum(dim=-1)
        dg_c = dg_c - ds2.sum(dim=-2)

        # ---- dg_state（显式 product+sum，与原参考一致）----
        dg_c = cast_round(dg_c)
        dg_c += (dq_state * q_b).sum(dim=-1)                                              # [N*H, C]
        dg_c = cast_round(dg_c)
        # k_b * dk_state 在 dg_c 和 dg_last_accum 中各用一次，提取公共子表达式
        k_dk_prod = k_b * dk_state                                                       # [N*H, C, K]
        dg_c = dg_c - k_dk_prod.sum(dim=-1)
        # h_prev*dh_curr 保留原始 datatype（与原实现一致）
        dg_last_accum = (h_b * dh_b).sum(dim=(-1, -2)) * torch.exp(g_last)                # [N*H]
        dg_last_accum = dg_last_accum + k_dk_prod.sum(dim=(-1, -2))                      # [N*H]

        # 每个 chunk 累加 dg_last：
        #   满 chunk → 加到 C-1；padded tail → 加到 ragged_len-1（原始最后有效 token）
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

        # ---- 4. dq/dk intra ----
        ds = cast_round(ds).to(datatype)
        dq_intra = cast_round(torch.bmm(ds.to(datatype), k_b.to(datatype)))                                          # [N*H, C, K]
        dk_intra = cast_round(torch.bmm(ds.transpose(1, 2).to(datatype), q_b.to(datatype)))                          # [N*H, C, K]

        dq_total = dq_state + dq_intra
        dk_total = dk_state + dk_intra

        # ---- 写回 dq_hv/dk_hv ----
        # [N*H, C, K] -> [N, H, C, K] -> [N, C, H, K] -> [N*C, H, K]
        if datatype == calc_type:
            dq_hv[b_idx, s0:s0 + N * C, :, :] = dq_total.reshape(N, H, C, K).permute(0, 2, 1, 3).reshape(N * C, H, K)
            dk_hv[b_idx, s0:s0 + N * C, :, :] = dk_total.reshape(N, H, C, K).permute(0, 2, 1, 3).reshape(N * C, H, K)
        else:
            dq_hv[b_idx, s0:s0 + N * C, :, :] = dq_total.to(datatype).reshape(N, H, C, K).permute(0, 2, 1, 3).reshape(N * C, H, K)
            dk_hv[b_idx, s0:s0 + N * C, :, :] = dk_total.to(datatype).reshape(N, H, C, K).permute(0, 2, 1, 3).reshape(N * C, H, K)


    # ------------------------------------------------------------------
    # 跨序列批量处理 ragged tails（varlen 模式下多个序列的尾部 chunk）。
    # 每个 tail 有不同的有效长度 ragged_i，pad 到 C 后批量计算。
    # padding 位置贡献为零（do=0/v=0），唯一需修正的是 dg_last_accum 位置：
    # 每个 tail 的 dg_last_accum 加到 ragged_i - 1 而非 C - 1。
    # ------------------------------------------------------------------
    def process_tails_batched(b_idx: int, tails: list):
        N = len(tails)
        C = chunk_size
        H = HV

        # ---- 收集并 pad 每个 tail ----
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

        # ---- 1. State contributions ----
        dq_state = cast_round(torch.bmm(do_b, h_b_t))
        dk_state = cast_round(torch.bmm(v_b, dh_b_t))

        dw_c = cast_round(torch.bmm(dv_b, h_b_t))

        mask_f = get_causal_mask_f(C, q.device)

        # ---- 2. Decay scale ----
        g_last = g_b.reshape(N, H, C)[:, :, C - 1].reshape(N * H)
        exp_gc = torch.exp(g_b)
        exp_neg_gc_glast = torch.exp(-g_b + g_last[:, None])
        dq_state = dq_state * (exp_gc[:, :, None] * scale)
        dk_state = dk_state * exp_neg_gc_glast[:, :, None]

        # ---- 3. Intra-chunk attention ----
        ds = cast_round(torch.bmm(do_b, v_b.transpose(1, 2)))
        g_diff = g_b[:, :, None] - g_b[:, None, :]
        decay_mat = torch.clamp(g_diff, max=0).exp_()
        ds = ds * decay_mat
        ds = ds * mask_f
        ds = ds * scale

        qk_t = cast_round(torch.bmm(q_b, k_b.transpose(1, 2)))
        ds2 = ds * qk_t
        dg_c = ds2.sum(dim=-1)
        dg_c = dg_c - ds2.sum(dim=-2)

        # ---- dg_state ----
        dg_c = cast_round(dg_c)
        dg_c += (dq_state * q_b).sum(dim=-1)
        dg_c = cast_round(dg_c)
        k_dk_prod = k_b * dk_state
        dg_c = dg_c - k_dk_prod.sum(dim=-1)
        dg_last_accum = (h_b * dh_b).sum(dim=(-1, -2)) * torch.exp(g_last)
        dg_last_accum = dg_last_accum + k_dk_prod.sum(dim=(-1, -2))

        # ---- dg_last_accum 位置修正 + 写回 ----
        dg_c = dg_c.reshape(N, H, C)
        dg_last_reshaped = dg_last_accum.reshape(N, H)
        for i, (ts, tr, _) in enumerate(tails):
            # 加到 ragged_i - 1（原始最后有效 token），而非 C - 1
            dg_c[i, :, tr - 1] += dg_last_reshaped[i]
            # 写回到原始位置 [ts, ts + tr)
            dg[b_idx, ts:ts + tr, :] = dg_c[i, :, :tr].to(gtype).permute(1, 0)

        # ---- 4. dq/dk intra ----
        ds = cast_round(ds)
        dq_intra = cast_round(torch.bmm(ds, k_b))
        dk_intra = cast_round(torch.bmm(ds.transpose(1, 2), q_b))

        dq_total = dq_state + dq_intra
        dk_total = dk_state + dk_intra

        # ---- 写回 dq_hv/dk_hv（只写有效长度部分）----
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


    # ------------------------------------------------------------------
    # 合并多个序列的满 chunk 为一次大 batched 调用。
    # 不同序列的满 chunk token 范围不连续，需要 cat 拼接后统一处理，
    # 完成后拆分写回到各自的原始位置。
    # ------------------------------------------------------------------
    def process_varlen_merged(b_idx: int, full_ranges: list):
        C = chunk_size
        H = HV
        total_N = sum(nf for _, nf, _ in full_ranges)

        # ---- 数据准备：cat 拼接所有序列的满 chunk ----
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

        N = total_N

        # ---- 1. State contributions ----
        dq_state = cast_round(torch.bmm(do_b, h_b_t))
        dk_state = cast_round(torch.bmm(v_b, dh_b_t))

        dw_c = cast_round(torch.bmm(dv_b, h_b_t))

        mask_f = get_causal_mask_f(C, q.device)

        # ---- 2. Decay scale ----
        g_last = g_b.reshape(N, H, C)[:, :, C - 1].reshape(N * H)
        exp_gc = torch.exp(g_b)
        exp_neg_gc_glast = torch.exp(-g_b + g_last[:, None])
        dq_state = dq_state * (exp_gc[:, :, None] * scale)
        dk_state = dk_state * exp_neg_gc_glast[:, :, None]

        # ---- 3. Intra-chunk attention ----
        ds = cast_round(torch.bmm(do_b, v_b.transpose(1, 2)))
        g_diff = g_b[:, :, None] - g_b[:, None, :]
        decay_mat = torch.clamp(g_diff, max=0).exp_()
        ds = ds * decay_mat
        ds = ds * mask_f
        ds = ds * scale

        qk_t = cast_round(torch.bmm(q_b, k_b.transpose(1, 2)))
        ds2 = ds * qk_t
        dg_c = ds2.sum(dim=-1)
        dg_c = dg_c - ds2.sum(dim=-2)

        # ---- dg_state ----
        dg_c = cast_round(dg_c)
        dg_c += (dq_state * q_b).sum(dim=-1)
        dg_c = cast_round(dg_c)
        k_dk_prod = k_b * dk_state
        dg_c = dg_c - k_dk_prod.sum(dim=-1)
        dg_last_accum = (h_b * dh_b).sum(dim=(-1, -2)) * torch.exp(g_last)
        dg_last_accum = dg_last_accum + k_dk_prod.sum(dim=(-1, -2))

        # ---- dg_last_accum + 写回 dg（按序列拆分）----
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

        # ---- 4. dq/dk intra ----
        ds = cast_round(ds)
        dq_intra = cast_round(torch.bmm(ds, k_b))
        dk_intra = cast_round(torch.bmm(ds.transpose(1, 2), q_b))

        dq_total = dq_state + dq_intra
        dk_total = dk_state + dk_intra

        # ---- 写回 dq_hv/dk_hv/dw（按序列拆分）----
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


    # Main Loop
    if cu_seqlens is None:
        # T 已 padding 到 chunk_size 的整数倍，所有 chunk 统一走 batched 路径
        for b in range(B):
            n_full = T // C
            if n_full > 0:
                process_dense_batched(b, n_full, 0, 0, ragged_len=ragged_len)
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

        if len(full_ranges) == 1:
            # 只有一个序列有满 chunk，直接调用
            ts, nf, th = full_ranges[0]
            process_dense_batched(0, nf, ts, th, ragged_len=0)
        elif len(full_ranges) > 1:
            # 合并多个序列的满 chunk 为一次大 batched 调用
            process_varlen_merged(0, full_ranges)

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
            process_tails_batched(0, tails)

    dq = dq_hv.view(B, T, HK, n_ratio, K).sum(dim=3).to(datatype)
    dk = dk_hv.view(B, T, HK, n_ratio, K).sum(dim=3).to(datatype)

    # 截断到原始 T（去除 padding 部分）
    if pad_T > 0:
        dq = dq[:, :T_orig]
        dk = dk[:, :T_orig]
        dw = dw[:, :T_orig]
        dg = dg[:, :T_orig]

    return dq, dk, dw, dg