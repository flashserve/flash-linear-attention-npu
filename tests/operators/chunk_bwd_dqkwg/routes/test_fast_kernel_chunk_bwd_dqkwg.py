#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the BSD 3-Clause License (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
from tests.operators._shared.legacy_cases import legacy_param_values
import pytest
from tests.operators._shared.route_requirements import require_fast_kernel_route

require_fast_kernel_route()

import ascend_ops
import math
import random
from typing import Optional, Tuple
import ct

def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def cdiv(a: torch.LongTensor, b: int):
    return (a + b - 1) // b


def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int
) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def generate_cu_seqlens(
    cu_seqlens_len: int,
    total_length: int,
    device: Optional[torch.device] = None
) -> torch.LongTensor:
    batchsize = cu_seqlens_len - 1
    remaining = total_length
    seq_lengths = []
    for i in range(batchsize - 1):
        min_len = 1
        max_len = remaining - (batchsize - 1 - i)
        if max_len < min_len:
            max_len = min_len
        seq_len = random.randint(min_len, max_len)
        seq_lengths.append(seq_len)
        remaining -= seq_len
    seq_lengths.append(remaining)

    cu_seqlens = [0]
    for seq_len in seq_lengths:
        cu_seqlens.append(cu_seqlens[-1] + seq_len)

    tensor = torch.tensor(cu_seqlens, dtype=torch.long)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def create_tensor(shape, dtype=torch.float16):
    return torch.rand(shape, dtype=dtype)


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
    benchmark = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CPU Equivalent of chunk_bwd_kernel_dqkwg.
    """
    if benchmark:
        calc_type = torch.float64
    else:
        calc_type = torch.float32
    datatype = q.dtype
    gtype = g.dtype
    q.to(calc_type)
    k.to(calc_type)
    v.to(calc_type)
    do.to(calc_type)
    h.to(calc_type)
    dh.to(calc_type)
    
    g.to(calc_type)
    dv.to(calc_type)
    B, T, HK, K = q.shape
    HV = v.shape[2]
    V = v.shape[-1]
    n_ratio = HV // HK  # HV = n_ratio * HK

    if benchmark:
        datatype = torch.float64
        gtype = torch.float64
    g_gamma = None
    
    # 输出使用 HV 维度
    dq = torch.zeros((B, T, HV, K), dtype=datatype)
    dk = torch.zeros((B, T, HV, K), dtype=datatype)
    dg = torch.zeros_like(g) if g is not None else None
    dw = torch.zeros((B, T, HV, K), dtype=datatype)
    w = torch.zeros((B, T, HV, K), dtype=datatype)

    
    # 辅助函数：处理单个序列的逻辑
    def process_sequence(b_idx, t_start, t_end, seq_idx_in_batch, chunk_start_idx):
        # 计算该序列有多少个块
        seq_len = t_end - t_start
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        # print("H(head)", H, "num_chunks", num_chunks, "b_idx", b_idx, "t_start", t_start, "t_end", t_end, "seq_idx_in_batch", seq_idx_in_batch, "chunk_start_idx", chunk_start_idx)

        
        for h_idx in range(HV):
            # h_idx is hv_idx; compute hk_idx for q/k access
            hk_idx = h_idx // n_ratio
            # 获取当前头的 gamma (如果 USE_G_GAMMA)
            gamma_val = None
            if g_gamma is not None:
                gamma_val = g_gamma[h_idx].item()

            for i_t in range(num_chunks):
                # 块的绝对起始位置
                chunk_start_token_idx = t_start + i_t * chunk_size
                chunk_end_token_idx = min(t_start + (i_t + 1) * chunk_size, t_end)
                actual_chunk_len = chunk_end_token_idx - chunk_start_token_idx

                # 当前块在 h/dh 中的索引 (NT 维度)
                # Triton 代码逻辑: i_tg = i_b * NT + i_t (定长) 或 i_t (变长且 chunk_indices 处理)
                # 这里我们假设 h 形状为 (B, H, NT, K, V) 或者兼容的扁平结构。
                # 为简化，假设标准 FLA 布局 (B, H, NT, K, V)
                # 注意：Triton 中 h 是指向第 i_t 个块的*起始*状态 (即上一个块的输出)
                
                # 切片当前块的数据
                
                q_c = q[b_idx, chunk_start_token_idx:chunk_end_token_idx, hk_idx, :]  # [BT, K]
                k_c = k[b_idx, chunk_start_token_idx:chunk_end_token_idx, hk_idx, :]  # [BT, K]

                v_c = v[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx, :]  # [BT, V]
                do_c = do[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx, :] # [BT, V]

                # 获取状态 (h_prev) 和 状态梯度 (dh_curr)
                # h[..., i_t, ...] 存储的是第 i_t 块之前的状态 (即第 i_t-1 块的输出)
                # print("\th ", h.shape, f"h[{b_idx}, {i_t} + {chunk_start_idx}, {h_idx}, :, :]")
                h_prev = h[b_idx, i_t + chunk_start_idx, h_idx, :, :]  # [K, V]  ## 不对齐的情况??
                dh_curr = dh[b_idx, i_t + chunk_start_idx, h_idx, :, :] # [K, V]

                # -----------------------------------------------------------
                # 1. State Contributions (Inter-chunk)
                # -----------------------------------------------------------
                # Triton: b_dq += dot(b_do, b_h) -> do @ h_prev.T
                # h_prev 是 [K, V], do_c 是 [BT, V] -> [BT, K]
                dq_from_state = do_c.to(calc_type) @ h_prev.transpose(-1, -2).to(calc_type)

                dq_from_state = dq_from_state.to(datatype).to(calc_type)

                # Triton: b_dk += dot(b_v, b_dh) -> v @ dh_curr.T
                # dh_curr 是 [K, V], v_c 是 [BT, V] -> [BT, K]
                dk_from_state = v_c.to(calc_type) @ dh_curr.transpose(-1, -2).to(calc_type)
                dk_from_state = dk_from_state.to(datatype).to(calc_type)
                # Triton: if USE_DW -> b_dw += dot(b_dv, b_h)
                if w is not None and dv is not None:
                    dv_c = dv[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx, :] # [BT, V]
                    # dw_c: [BT, K]
                    dw_c_val = dv_c.to(calc_type) @ h_prev.transpose(-1, -2).to(calc_type)
                    dw_c_val = dw_c_val.to(datatype).to(calc_type)
                    # Triton stores -b_dw
                    dw[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx, :] = -dw_c_val

                # -----------------------------------------------------------
                # 2. Gating / Decay Logic Preparation
                # -----------------------------------------------------------
                # 构建 g_c (decay values)
                if g is not None:
                    g_c = g[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx] # [BT]

                    g_last = g[b_idx, min(chunk_start_token_idx + chunk_size, t_end) - 1, h_idx]
                    
                    # Triton: b_dg_last += sum(h * dh)
                    dg_last_accum = (h_prev * dh_curr).sum()

                    dg_last_accum = dg_last_accum * torch.exp(g_last)
                    # Apply decay to state contributions

                    dq_from_state = dq_from_state * torch.exp(g_c)[:, None] * scale
                    dk_from_state = dk_from_state * torch.exp(-g_c + g_last)[:, None]

                    # Accumulate gradients into dg (from state terms)
                    # b_dg += sum(b_dq * b_q)
                    dg_c = (dq_from_state * q_c).sum(dim=-1)
                    # print("ADD0.A", dg_c.to(datatype).to(calc_type))
                    dg_c = dg_c.to(datatype).to(calc_type)         #ADD0.A

                    # b_dg -= sum(b_k * b_dk)
                    dg_c -= (k_c * dk_from_state).sum(dim=-1)           #ADD0.B
                    # print("k_c",k_c)
                    # print("dk_from_state",dk_from_state)
                    # print("k_c * dk_from_state",( k_c * dk_from_state)[0])
                    # print("Add0.B", -(k_c * dk_from_state).sum(dim=-1))
                    dg_c = dg_c.to(datatype).to(calc_type)

                    # b_dg_last += sum(b_dk * b_k)
                    # print(f"dg_last_accum {dg_last_accum} += (dk_from_state * k_c).sum() {(dk_from_state * k_c).sum()}")
                    # if h_idx == 0 and i_t == 31:
                    #     print("     sum0 result", (dk_from_state * k_c).sum())
                    dg_last_accum += (dk_from_state * k_c).sum()
                    # print("dg_last_accum += (dk_from_state * k_c).sum()", dg_last_accum)
                    # pause()
                    

                elif g_gamma is not None:
                    # Scalar decay
                    # b_g = b_gamma * (arange + 1)
                    # b_g_last = b_gamma * actual_chunk_len
                    # 这里模拟 Triton 里的相对 decay 逻辑
                    arange = torch.arange(actual_chunk_len, device=q.device, dtype=q.dtype)
                    g_c = gamma_val * (arange + 1)
                    g_last = gamma_val * actual_chunk_len
                    
                    dq_from_state = dq_from_state * torch.exp(g_c)[:, None] * scale
                    dk_from_state = dk_from_state * torch.exp(-g_c + g_last)[:, None]
                    # USE_G_GAMMA 模式下不需要计算 dg
                else:
                    # No decay
                    # Triton: b_dk *= scale (else block)
                    dk_from_state = dk_from_state * scale
                    dq_from_state = dq_from_state * scale

                # -----------------------------------------------------------
                # 3. Intra-chunk Attention
                # -----------------------------------------------------------
                ds = do_c.to(calc_type) @ v_c.transpose(-1, -2).to(calc_type) # [BT, BT]
                ds = ds.to(datatype).to(calc_type)

                
                # Causal Mask
                i_indices = torch.arange(actual_chunk_len, device=q.device)[:, None]
                j_indices = torch.arange(actual_chunk_len, device=q.device)[None, :]
                mask = i_indices >= j_indices
                
                if g is not None:
                    # Decay: exp(g[i] - g[j])

                    decay_mat = torch.exp(g_c[:, None] - g_c[None, :])
                    # if h_idx == 0 and i_t == 7:
                    #     print("g_c[:, None] - g_c[None, :]", g_c[:, None] - g_c[None, :])
                    #     print("decay_mat = torch.exp(g_c[:, None] - g_c[None, :])", decay_mat)

                    ds = torch.where(mask, ds * decay_mat, torch.zeros_like(ds)) * scale
                    # print("decay_mat",decay_mat)
                    # print("ds",ds)

                    
                    # DG Calculation Part 2 (Intra-chunk)
                    # b_ds2 = b_ds * (q @ k.T)
                    qk_t = q_c.to(calc_type) @ k_c.transpose(-1, -2).to(calc_type)
                    qk_t = qk_t.to(datatype).to(calc_type)


                    ds2 = ds * qk_t

                    # print("ADD0.C : +ds2.sum(dim=1)", ds2.sum(dim=1))
                    # print("ADD0.D : -ds2.sum(dim=0)", ds2.sum(dim=0))
                    dg_c += ds2.sum(dim=1)
                    dg_c -= ds2.sum(dim=0)

                    # dg_c = dg_c_C.to(torch.float16) + dg_c_D.to(torch.float16) + dg_c_A.to(torch.float16) + dg_c_B.to(torch.float16)
                    dg_c = dg_c.to(gtype)

                    # print("dg_c after", dg_c.shape)
                    # pause()
                    
                    # Finalize dg: revcumsum-like logic
                    # Triton: b_dg = where(o_t < T-1, b_dg, b_dg + b_dg_last)
                    # 只有块的最后一个有效 token 加上 dg_last_accum
                    # 注意：Triton 内核中的 revcumsum 通常在单独内核或最后处理，
                    # 但这里代码片段显示的是直接加上。
                    # 实际上 dg 在时间轴上是累积的梯度。
                    # 根据 Triton 代码: b_dg = ... + (idx == last ? b_dg_last : 0)
                    # 这里的 dg_c 仅仅是该位置的梯度 contribution。
                    # 为了完全匹配 Triton 的输出，我们需要把 dg_last_accum 加到块的最后。
                    if actual_chunk_len > 0:
                        dg_c[actual_chunk_len - 1] += dg_last_accum.to(gtype)  ## 实际上是is_last_mask

                    #     print(f"dg_c[{actual_chunk_len - 1}] += {dg_last_accum}")
                    # print("dg_c", dg_c)
                    dg[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx] = dg_c
                    # print("dg_c",dg_c)

                elif g_gamma is not None:

                    decay_mat = torch.exp(g_c[:, None] - g_c[None, :])
                    
                    ds = torch.where(mask, ds * decay_mat, torch.zeros_like(ds)) * scale

                else:
                    ds = torch.where(mask, ds, torch.zeros_like(ds))
                    # 在 else 分支，triton 代码: b_dq *= scale (最后)
                    # 但前面 state part 已经 scale 了。
                    # ds 计算时不乘 scale，最后 dq 乘 scale。
                    # 为了统一，这里先不乘 scale，下面加完后再处理，或者这里乘了下面不再乘。
                    # Triton 代码: b_dk += dot(trans(b_ds), b_q) * scale
                    # b_dq += dot(b_ds, b_k); b_dq *= scale
                    pass # logic handled below

                # -----------------------------------------------------------
                # 4. Final Accumulation for dq, dk
                # -----------------------------------------------------------
                # dq += ds @ k

                dq_intra = ds.to(calc_type) @ k_c.to(calc_type)
                # if h_idx == 0 and i_t == 7:
                #     print("ds.to(torch.float32)",ds.to(torch.float32))
                #     print("k_c.to(torch.float32)",k_c.to(torch.float32))
                dq_intra = dq_intra.to(datatype).to(calc_type)
                # dk += ds.T @ q
                dk_intra = ds.transpose(-1, -2).to(calc_type) @ q_c.to(calc_type)
                dk_intra = dk_intra.to(datatype).to(calc_type)

                
                if g is None and g_gamma is None:
                    # Special scaling for "No Decay" mode based on Triton code
                    dk_intra = dk_intra * scale
                    dq_total = (dq_from_state + dq_intra) * scale # Triton: b_dq *= scale at end
                    dk_total = dk_from_state + dk_intra
                else:
                    dq_total = dq_from_state + dq_intra
                    # if h_idx == 0 and i_t == 7:
                    #     print("dq_from_state",dq_from_state[-1])
                    #     print("dq_intra",dq_intra[-1])
                    #     print("dq_total",dq_total.shape,dq_total[-1])
                    dk_total = dk_from_state + dk_intra

                    # print(h_idx,i_t,"dk_from_state",dk_from_state)
                    # print(h_idx,i_t,"dk_intra",dk_intra)
                    # print(h_idx,i_t,"dk_total",dk_total)


                dq[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx, :] = dq_total
                dk[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx, :] = dk_total

                # if RANDOM_DATA == False:
                #     pass
                # if h_idx == 0 and i_t == 31:
                #     pass
                #     exit()
                

    # Main Loop
    if cu_seqlens is None:
        # Fixed length padding assumed or B*T
        for b in range(B):
            process_sequence(b, 0, T, b, 0)
    else:
        # Variable length
        chunk_location = torch.zeros(cu_seqlens.shape[0], dtype=torch.int64) #每个seq的chunk起始位置
        #chunk_location tensor([0, 64, 96, 128]) 代表：[0,63] [64,95] [96,127]

        for i in range(len(cu_seqlens) - 1):
            start, end = cu_seqlens[i].item(), cu_seqlens[i+1].item()
            seq_length = end - start
            # print("seq_length", seq_length)
            if i == 0:
                chunk_start_token_idx = 0
            else:
                chunk_start_token_idx = chunk_location[i]
            # print("chunk_start_token_idx before", chunk_start_token_idx)
            chunk_end_token_idx = chunk_start_token_idx + (seq_length + chunk_size - 1) // chunk_size
            # print("chunk_end_token_idx after", chunk_end_token_idx)
            chunk_location[i + 1] = chunk_end_token_idx

            # 在 Varlen 模式下，q/k/v 通常已经是 (Total_T, ...) 或者是 (1, Total_T, ...)
            # 但这里输入还是 (B, T, ...)，我们需要确认输入格式。
            # 通常 Triton varlen kernel 的输入 q 是 (Total_T, H, K)。
            # 如果输入是 packed (1, Total_T, ...)，b_idx 永远是 0。
            # 如果输入是 padded (B, T, ...)，则需要根据 cu_seqlens 切分。
            # 假设输入已根据 varlen 展平 (Batch=1) 或保持 Padded 格式。
            # 鉴于 Triton 代码 `i_b = i_bh // H`，如果 IS_VARLEN，逻辑略有不同。
            # 为保证通用性，这里假设输入是 Padded (B, T) 且 cu_seqlens 描述有效区域，
            # 或者 B=1 的 Packed 模式。
            if B == 1:
                print(f"start {start}, end {end}")
                # if (i == 0):
                #     continue
                process_sequence(0, start, end, i, chunk_location[i])
            else:
                # 如果是 Padded Batch 且提供了 cu_seqlens，这通常不常见，
                # 但如果发生，通常 cu_seqlens[i] 是第 i 个样本的长度。
                # 简化起见，我们假设 input 是 packed flat tensor 如果 cu_seqlens 存在。
                pass 

    return dq, dk, dw, dg


def chunk_bwd_dqkwg_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
    n_ratio: int = 1,
    benchmark = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_ref = q.permute(0, 2, 1, 3).contiguous()
    k_ref = k.permute(0, 2, 1, 3).contiguous()
    v_ref = v.permute(0, 2, 1, 3).contiguous()
    do_ref = do.permute(0, 2, 1, 3).contiguous()
    dv_ref = dv.permute(0, 2, 1, 3).contiguous()
    g_ref = g.permute(0, 2, 1).contiguous()
    h_ref = h.permute(0, 2, 1, 3, 4).contiguous()
    dh_ref = dh.permute(0, 2, 1, 3, 4).contiguous()

    dq_ref, dk_ref, dw_ref, dg_ref = chunk_bwd_dqkwg_cpu(
        q_ref, k_ref, v_ref, do_ref, h_ref, dh_ref,
        None, g_ref, dv_ref,
        scale, cu_seqlens, chunk_size, benchmark = benchmark
    )

    B, HK, T, K = q.shape
    HV = v.shape[1]
    if HV != HK:
        dq_ref = dq_ref.reshape(B, T, HK, n_ratio, K).sum(dim=3)
        dk_ref = dk_ref.reshape(B, T, HK, n_ratio, K).sum(dim=3)

    dq_out = dq_ref.permute(0, 2, 1, 3).contiguous()#.to(torch.float32)
    dk_out = dk_ref.permute(0, 2, 1, 3).contiguous()#.to(torch.float32)
    dw_out = dw_ref.permute(0, 2, 1, 3).contiguous()#.to(torch.float32)
    dg_out = dg_ref.permute(0, 2, 1).contiguous()#.to(torch.float32)

    return dq_out, dk_out, dw_out, dg_out


def test_chunk_bwd_dqkwg_interface_exist():
    """
    Test that the 'ascend_ops.chunk_bwd_dqkwg' operator is present in torch.ops.
    """
    print(torch.ops.ascend_ops.chunk_bwd_dqkwg)
    assert hasattr(torch.ops.ascend_ops, "chunk_bwd_dqkwg"), \
        "The 'chunk_bwd_dqkwg' operator is not registered in the 'torch.ops.ascend_ops' namespace."


FIX_TEST_CONFIGS = [
    pytest.param(*values, id=case_id)
    for case_id, values in legacy_param_values(
        "chunk_bwd_dqkwg", "fast_kernel_launch", group="fixed", dtype_module=torch
    )
]

VARIABLE_TEST_CONFIGS = [
    pytest.param(*values, id=case_id)
    for case_id, values in legacy_param_values(
        "chunk_bwd_dqkwg", "fast_kernel_launch", group="variable", dtype_module=torch
    )
]

@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("B,HK,HV,T,K,V,chunk_size,scale,ktype,gtype,n_ratio", FIX_TEST_CONFIGS)
def test_chunk_bwd_dqkwg_fix(B, HK, HV, T, K, V, chunk_size, scale, ktype, gtype, n_ratio):
    """
    Test the chunk_bwd_dqkwg operator with fixed-length sequences.
    Compares NPU kernel launch result against PyTorch reference implementation.
    Supports GVA: HV = HK * n_ratio.
    """
    torch.manual_seed(0)

    num_chunks = (T + chunk_size - 1) // chunk_size

    q = create_tensor((B, HK, T, K), dtype=ktype)
    k = create_tensor((B, HK, T, K), dtype=ktype)
    v = create_tensor((B, HV, T, V), dtype=ktype)
    # g = torch.arange(B * HV * T, 0, -1).reshape((B, HV, T)).to(gtype) * (-0.01)
    g = -torch.sort(torch.rand(B*T*HV) * 10, descending=False)[0].reshape((B,HV,T)).to(gtype)
    h = create_tensor((B, HV, num_chunks, K, V), dtype=ktype)
    do = create_tensor((B, HV, T, V), dtype=ktype)
    dh = create_tensor((B, HV, num_chunks, K, V), dtype=ktype)
    dv = create_tensor((B, HV, T, V), dtype=ktype)

    print(f" q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}, "
          f"g.shape={g.shape}, h.shape={h.shape}, do.shape={do.shape}, "
          f"dh.shape={dh.shape}, dv.shape={dv.shape}, n_ratio={n_ratio}")

    dq_ref, dk_ref, dw_ref, dg_ref = chunk_bwd_dqkwg_ref(
        q, k, v, g, h, do, dh, dv, scale,
        cu_seqlens=None, chunk_size=chunk_size, n_ratio=n_ratio
    )

    dq_ref_bench, dk_ref_bench, dw_ref_bench, dg_ref_bench = chunk_bwd_dqkwg_ref(
        q, k, v, g, h, do, dh, dv, 0.0625,
        cu_seqlens=None, chunk_size=chunk_size, n_ratio=n_ratio, benchmark=True
    )

    q_npu = q.npu()
    k_npu = k.npu()
    v_npu = v.npu()
    g_npu = g.npu()
    h_npu = h.npu()
    do_npu = do.npu()
    dh_npu = dh.npu()
    dv_npu = dv.npu()

    dq, dk, dw, dg = torch.ops.ascend_ops.chunk_bwd_dqkwg(
        q_npu, k_npu, v_npu, g_npu, h_npu, do_npu, dh_npu, dv_npu,
        scale, chunk_size,
        w=None, g_gamma=None, cu_seqlens=None, chunk_indices=None
    )

    dq_cpu = dq.cpu().to(torch.float32)
    dk_cpu = dk.cpu().to(torch.float32)
    dw_cpu = dw.cpu().to(torch.float32)
    dg_cpu = dg.cpu().to(torch.float32) if dg.dtype != torch.float32 else dg.cpu()

    assert ct.dual(dq_cpu.to(torch.float16), dq_ref.to(torch.float16), dq_ref_bench)['success'],"Failed dual"
    assert ct.dual(dk_cpu.to(torch.float16), dk_ref.to(torch.float16), dk_ref_bench)['success'],"Failed dual"
    assert ct.dual(dw_cpu.to(torch.float16), dw_ref.to(torch.float16), dw_ref_bench)['success'],"Failed dual"
    assert ct.dual(dg_cpu.to(torch.float16), dg_ref.to(torch.float16), dg_ref_bench)['success'],"Failed dual"

    print(f"✓ fix test passed: B={B}, HK={HK}, HV={HV}, T={T}, K={K}, V={V}, "
          f"chunk_size={chunk_size}, ktype={ktype}, gtype={gtype}, n_ratio={n_ratio}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("B,HK,HV,T,K,V,chunk_size,scale,cu_seqlens_len,ktype,gtype,n_ratio", VARIABLE_TEST_CONFIGS)
def test_chunk_bwd_dqkwg_variable(B, HK, HV, T, K, V, chunk_size, scale, cu_seqlens_len, ktype, gtype, n_ratio):
    """
    Test the chunk_bwd_dqkwg operator with variable-length sequences.
    Compares NPU kernel launch result against PyTorch reference implementation.
    Supports GVA: HV = HK * n_ratio.
    """
    torch.manual_seed(0)

    cu_seqlens = generate_cu_seqlens(cu_seqlens_len, T)
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    num_chunks = chunk_indices.shape[0]

    q = create_tensor((B, HK, T, K), dtype=ktype)
    k = create_tensor((B, HK, T, K), dtype=ktype)
    v = create_tensor((B, HV, T, V), dtype=ktype)
    # g = torch.arange(B * HV * T, 0, -1).reshape((B, HV, T)).to(gtype) * (-0.01)
    g = -torch.sort(torch.rand(B*T*HV) * 10, descending=False)[0].reshape((B,HV,T)).to(gtype)
    h = create_tensor((B, HV, num_chunks, K, V), dtype=ktype)
    do = create_tensor((B, HV, T, V), dtype=ktype)
    dh = create_tensor((B, HV, num_chunks, K, V), dtype=ktype)
    dv = create_tensor((B, HV, T, V), dtype=ktype)

    dq_ref, dk_ref, dw_ref, dg_ref = chunk_bwd_dqkwg_ref(
        q, k, v, g, h, do, dh, dv, scale,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size, n_ratio=n_ratio
    )
    
    dq_ref_bench, dk_ref_bench, dw_ref_bench, dg_ref_bench = chunk_bwd_dqkwg_ref(
        q, k, v, g, h, do, dh, dv, 0.0625,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size, n_ratio=n_ratio, benchmark=True
    )

    q_npu = q.npu()
    k_npu = k.npu()
    v_npu = v.npu()
    g_npu = g.npu()
    h_npu = h.npu()
    do_npu = do.npu()
    dh_npu = dh.npu()
    dv_npu = dv.npu()

    cu_seqlens_list = cu_seqlens.tolist()
    chunk_indices_list = chunk_indices.flatten().tolist()

    dq, dk, dw, dg = torch.ops.ascend_ops.chunk_bwd_dqkwg(
        q_npu, k_npu, v_npu, g_npu, h_npu, do_npu, dh_npu, dv_npu,
        scale, chunk_size,
        w=None, g_gamma=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices_list
    )

    dq_cpu = dq.cpu().to(torch.float32)
    dk_cpu = dk.cpu().to(torch.float32)
    dw_cpu = dw.cpu().to(torch.float32)
    dg_cpu = dg.cpu().to(torch.float32) if dg.dtype != torch.float32 else dg.cpu()

    assert ct.dual(dq_cpu.to(torch.float16), dq_ref.to(torch.float16), dq_ref_bench)['success'],"Failed dual"
    assert ct.dual(dk_cpu.to(torch.float16), dk_ref.to(torch.float16), dk_ref_bench)['success'],"Failed dual"
    assert ct.dual(dw_cpu.to(torch.float16), dw_ref.to(torch.float16), dw_ref_bench)['success'],"Failed dual"
    assert ct.dual(dg_cpu.to(torch.float16), dg_ref.to(torch.float16), dg_ref_bench)['success'],"Failed dual"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_chunk_bwd_dqkwg_output_shapes():
    """
    Test that the chunk_bwd_dqkwg operator returns tensors with correct shapes.
    """
    torch.manual_seed(0)

    B, HK, HV, T, K, V, chunk_size = 1, 2, 2, 65, 128, 128, 64
    num_chunks = (T + chunk_size - 1) // chunk_size

    q = create_tensor((B, HK, T, K), dtype=torch.float16).npu()
    k = create_tensor((B, HK, T, K), dtype=torch.float16).npu()
    v = create_tensor((B, HV, T, V), dtype=torch.float16).npu()
    g = torch.randn((B, HV, T), dtype=torch.float16).npu()
    h = create_tensor((B, HV, num_chunks, K, V), dtype=torch.float16).npu()
    do = create_tensor((B, HV, T, V), dtype=torch.float16).npu()
    dh = create_tensor((B, HV, num_chunks, K, V), dtype=torch.float16).npu()
    dv = create_tensor((B, HV, T, V), dtype=torch.float16).npu()

    dq, dk, dw, dg = torch.ops.ascend_ops.chunk_bwd_dqkwg(
        q, k, v, g, h, do, dh, dv,
        0.0625, chunk_size,
        w=None, g_gamma=None, cu_seqlens=None, chunk_indices=None
    )

    assert dq.shape == (B, HK, T, K), f"dq shape mismatch: expected {(B, HK, T, K)}, got {dq.shape}"
    assert dk.shape == (B, HK, T, K), f"dk shape mismatch: expected {(B, HK, T, K)}, got {dk.shape}"
    assert dw.shape == (B, HV, T, K), f"dw shape mismatch: expected {(B, HV, T, K)}, got {dw.shape}"
    assert dg.shape == (B, HV, T), f"dg shape mismatch: expected {(B, HV, T)}, got {dg.shape}"

    print(f"✓ shape test passed: dq={dq.shape}, dk={dk.shape}, dw={dw.shape}, dg={dg.shape}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_chunk_bwd_dqkwg_gva_output_shapes():
    """
    Test output shapes with GVA (grouped-value attention): HV = HK * n_ratio.
    """
    torch.manual_seed(0)

    B, HK, HV, T, K, V, chunk_size = 1, 2, 4, 65, 128, 128, 64
    n_ratio = HV // HK
    num_chunks = (T + chunk_size - 1) // chunk_size

    q = create_tensor((B, HK, T, K), dtype=torch.float16).npu()
    k = create_tensor((B, HK, T, K), dtype=torch.float16).npu()
    v = create_tensor((B, HV, T, V), dtype=torch.float16).npu()
    g = torch.randn((B, HV, T), dtype=torch.float16).npu()
    h = create_tensor((B, HV, num_chunks, K, V), dtype=torch.float16).npu()
    do = create_tensor((B, HV, T, V), dtype=torch.float16).npu()
    dh = create_tensor((B, HV, num_chunks, K, V), dtype=torch.float16).npu()
    dv = create_tensor((B, HV, T, V), dtype=torch.float16).npu()

    dq, dk, dw, dg = torch.ops.ascend_ops.chunk_bwd_dqkwg(
        q, k, v, g, h, do, dh, dv,
        0.0625, chunk_size,
        w=None, g_gamma=None, cu_seqlens=None, chunk_indices=None
    )

    assert dq.shape == (B, HK, T, K), f"dq shape mismatch: expected {(B, HK, T, K)}, got {dq.shape}"
    assert dk.shape == (B, HK, T, K), f"dk shape mismatch: expected {(B, HK, T, K)}, got {dk.shape}"
    assert dw.shape == (B, HV, T, K), f"dw shape mismatch: expected {(B, HV, T, K)}, got {dw.shape}"
    assert dg.shape == (B, HV, T), f"dg shape mismatch: expected {(B, HV, T)}, got {dg.shape}"

    print(f"✓ GVA shape test passed: HK={HK}, HV={HV}, n_ratio={n_ratio}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_chunk_bwd_dqkwg_dtypes():
    """
    Test that the chunk_bwd_dqkwg operator returns tensors with correct dtypes.
    """
    torch.manual_seed(0)

    B, HK, HV, T, K, V, chunk_size = 1, 2, 2, 65, 128, 128, 64
    num_chunks = (T + chunk_size - 1) // chunk_size

    for ktype, gtype in [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.float32),
    ]:
        q = create_tensor((B, HK, T, K), dtype=ktype).npu()
        k = create_tensor((B, HK, T, K), dtype=ktype).npu()
        v = create_tensor((B, HV, T, V), dtype=ktype).npu()
        g = torch.randn((B, HV, T), dtype=gtype).npu()
        h = create_tensor((B, HV, num_chunks, K, V), dtype=ktype).npu()
        do = create_tensor((B, HV, T, V), dtype=ktype).npu()
        dh = create_tensor((B, HV, num_chunks, K, V), dtype=ktype).npu()
        dv = create_tensor((B, HV, T, V), dtype=ktype).npu()

        dq, dk, dw, dg = torch.ops.ascend_ops.chunk_bwd_dqkwg(
            q, k, v, g, h, do, dh, dv,
            0.0625, chunk_size,
            w=None, g_gamma=None, cu_seqlens=None, chunk_indices=None
        )

        assert dq.dtype == ktype, f"dq dtype mismatch: expected {ktype}, got {dq.dtype}"
        assert dk.dtype == ktype, f"dk dtype mismatch: expected {ktype}, got {dk.dtype}"
        assert dw.dtype == ktype, f"dw dtype mismatch: expected {ktype}, got {dw.dtype}"
        assert dg.dtype == gtype, f"dg dtype mismatch: expected {gtype}, got {dg.dtype}"

        print(f"✓ dtype test passed: ktype={ktype}, gtype={gtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_chunk_bwd_dqkwg_chunk_size_128():
    """
    Test the chunk_bwd_dqkwg operator with chunk_size=128.
    """
    torch.manual_seed(0)

    B, HK, HV, T, K, V, chunk_size = 1, 4, 8, 512, 128, 128, 128
    num_chunks = (T + chunk_size - 1) // chunk_size
    n_ratio = HV // HK

    q = create_tensor((B, HK, T, K), dtype=torch.float16)
    k = create_tensor((B, HK, T, K), dtype=torch.float16)
    v = create_tensor((B, HV, T, V), dtype=torch.float16)
    g = -torch.sort(torch.rand(B*T*HV) * 10, descending=False)[0].reshape((B,HV,T)).to(torch.float32)
    h = create_tensor((B, HV, num_chunks, K, V), dtype=torch.float16)
    do = create_tensor((B, HV, T, V), dtype=torch.float16)
    dh = create_tensor((B, HV, num_chunks, K, V), dtype=torch.float16)
    dv = create_tensor((B, HV, T, V), dtype=torch.float16)

    dq_ref, dk_ref, dw_ref, dg_ref = chunk_bwd_dqkwg_ref(
        q, k, v, g, h, do, dh, dv, 0.0625,
        cu_seqlens=None, chunk_size=chunk_size, n_ratio=n_ratio
    )

    dq_ref_bench, dk_ref_bench, dw_ref_bench, dg_ref_bench = chunk_bwd_dqkwg_ref(
        q, k, v, g, h, do, dh, dv, 0.0625,
        cu_seqlens=None, chunk_size=chunk_size, n_ratio=n_ratio, benchmark=True
    )

    dq, dk, dw, dg = torch.ops.ascend_ops.chunk_bwd_dqkwg(
        q.npu(), k.npu(), v.npu(), g.npu(), h.npu(), do.npu(), dh.npu(), dv.npu(),
        0.0625, chunk_size,
        w=None, g_gamma=None, cu_seqlens=None, chunk_indices=None
    )

    dq_cpu = dq.cpu()#.to(torch.float32)
    dk_cpu = dk.cpu()#.to(torch.float32)
    dw_cpu = dw.cpu()#.to(torch.float32)
    dg_cpu = dg.cpu()#.to(torch.float32)

    assert ct.dual(dq_cpu.to(torch.float16), dq_ref.to(torch.float16), dq_ref_bench)['success'],"Failed dual"
    assert ct.dual(dk_cpu.to(torch.float16), dk_ref.to(torch.float16), dk_ref_bench)['success'],"Failed dual"
    assert ct.dual(dw_cpu.to(torch.float16), dw_ref.to(torch.float16), dw_ref_bench)['success'],"Failed dual"
    assert ct.dual(dg_cpu.to(torch.float32), dg_ref.to(torch.float32), dg_ref_bench)['success'],"Failed dual"

    print(f"✓ chunk_size=128 test passed: T={T}, num_chunks={num_chunks}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_chunk_bwd_dqkwg_v256():
    """
    Test the chunk_bwd_dqkwg operator with V=256.
    """
    torch.manual_seed(0)

    B, HK, HV, T, K, V, chunk_size = 1, 2, 2, 65, 128, 256, 64
    num_chunks = (T + chunk_size - 1) // chunk_size
    n_ratio = HV // HK

    q = create_tensor((B, HK, T, K), dtype=torch.float16)
    k = create_tensor((B, HK, T, K), dtype=torch.float16)
    v = create_tensor((B, HV, T, V), dtype=torch.float16)
    g = -torch.sort(torch.rand(B*T*HV) * 10, descending=False)[0].reshape((B,HV,T)).to(torch.float32)
    h = create_tensor((B, HV, num_chunks, K, V), dtype=torch.float16)
    do = create_tensor((B, HV, T, V), dtype=torch.float16)
    dh = create_tensor((B, HV, num_chunks, K, V), dtype=torch.float16)
    dv = create_tensor((B, HV, T, V), dtype=torch.float16)

    dq_ref, dk_ref, dw_ref, dg_ref = chunk_bwd_dqkwg_ref(
        q, k, v, g, h, do, dh, dv, 0.0625,
        cu_seqlens=None, chunk_size=chunk_size, n_ratio=n_ratio
    )

    dq_ref_bench, dk_ref_bench, dw_ref_bench, dg_ref_bench = chunk_bwd_dqkwg_ref(
        q, k, v, g, h, do, dh, dv, 0.0625,
        cu_seqlens=None, chunk_size=chunk_size, n_ratio=n_ratio, benchmark=True
    )

    dq, dk, dw, dg = torch.ops.ascend_ops.chunk_bwd_dqkwg(
        q.npu(), k.npu(), v.npu(), g.npu(), h.npu(), do.npu(), dh.npu(), dv.npu(),
        0.0625, chunk_size,
        w=None, g_gamma=None, cu_seqlens=None, chunk_indices=None
    )

    dq_cpu = dq.cpu().to(torch.float32)
    dk_cpu = dk.cpu().to(torch.float32)
    dw_cpu = dw.cpu().to(torch.float32)
    dg_cpu = dg.cpu().to(torch.float32)

    rtol = 1e-2
    atol = 1e-2

    assert ct.dual(dq_cpu.to(torch.float16), dq_ref.to(torch.float16), dq_ref_bench)['success'],"Failed dual"
    assert ct.dual(dk_cpu.to(torch.float16), dk_ref.to(torch.float16), dk_ref_bench)['success'],"Failed dual"
    assert ct.dual(dw_cpu.to(torch.float16), dw_ref.to(torch.float16), dw_ref_bench)['success'],"Failed dual"
    assert ct.dual(dg_cpu.to(torch.float32), dg_ref.to(torch.float32), dg_ref_bench)['success'],"Failed dual"

    print(f"✓ V=256 test passed")
