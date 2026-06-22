#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import ascend_ops
import pytest
import math
import random
from typing import Optional

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


def chunk_bwd_dv_local_fix(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[float],
    cu_seqlens: torch.LongTensor,
    chunk_size: int = 64,
    h_ratio: int = 1
) -> torch.Tensor:
    B, H_qk, T, K = k.shape
    H_do = do.shape[1]
    V = do.shape[3]
    assert H_do == H_qk * h_ratio, f"H_do ({H_do}) must equal H_qk * h_ratio ({H_qk} * {h_ratio} = {H_qk * h_ratio})"
    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if cu_seqlens is not None:
        batch_idx = 0
    BT = min(chunk_size, max(16, 2 ** math.ceil(math.log2(T))))
    chunk_per_T = (T + chunk_size - 1) // chunk_size
    NT = chunk_per_T * B
    dv = torch.zeros_like(do).to(torch.float32)
    g_t = g
    for chunk_idx in range(NT):
        i_n = chunk_idx // chunk_per_T
        batch_idx = i_n
        i_t = chunk_idx % chunk_per_T

        chunk_start_token = i_t * chunk_size
        chunk_end_token = min(chunk_start_token + chunk_size, T)
        chunk_len = chunk_end_token - chunk_start_token
        if chunk_len <= 0:
            continue
        for qk_head in range(H_qk):
            b_A = torch.zeros(BT, BT, device=q.device, dtype=torch.float32)
            BK = 128
            BK = min(BK, K)
            for i_k in range(0, K, BK):
                k_end = min(i_k + BK, K)
                b_k = k[batch_idx, qk_head, chunk_start_token:chunk_start_token+chunk_len, i_k:k_end].to(torch.float32)
                q_normal = q[batch_idx, qk_head, chunk_start_token:chunk_start_token+chunk_len, i_k:k_end].to(torch.float32)
                b_q = q_normal.transpose(0, 1)
                if chunk_len == 1:
                    matmul_result = torch.sum(b_k * q_normal)
                    b_A[:chunk_len, :chunk_len] += matmul_result
                else:
                    b_A[:chunk_len, :chunk_len] += torch.matmul(b_k, b_q)
            for do_group in range(h_ratio):
                do_head = qk_head * h_ratio + do_group
                b_g = g_t[batch_idx, do_head, chunk_start_token:chunk_start_token+chunk_len]
                o_t = i_t * BT + torch.arange(0, BT)
                m_t = o_t < T
                o_t_col = o_t.unsqueeze(1)
                o_t_row = o_t.unsqueeze(0)
                pos_mask = o_t_col <= o_t_row
                m_t_col = m_t.unsqueeze(1)
                valid_mask = m_t_col & m_t
                m_A = pos_mask & valid_mask
                g_i = b_g.unsqueeze(1)
                g_j = b_g.unsqueeze(0)
                g_factor = torch.exp(g_j - g_i) * scale
                b_A_gated = torch.zeros_like(b_A)
                b_A_gated[:chunk_len, :chunk_len] = b_A[:chunk_len, :chunk_len] * g_factor
                b_A_masked = torch.where(m_A, b_A_gated, torch.zeros_like(b_A_gated))
                b_A_masked = b_A_masked.to(torch.float32)
                BV = 128
                BV = min(BV, V)
                for i_v in range(0, V, BV):
                    v_end = min(i_v + BV, V)
                    b_do = do[batch_idx, do_head, chunk_start_token:chunk_start_token+chunk_len, i_v:v_end].to(torch.float32)
                    b_dv = torch.matmul(b_A_masked[:chunk_len, :chunk_len], b_do)
                    dv[batch_idx, do_head, chunk_start_token:chunk_start_token+chunk_len, i_v:v_end] += b_dv
    return dv


def chunk_bwd_dv_local_variable(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[float],
    cu_seqlens: torch.LongTensor,
    chunk_size: int = 64,
    h_ratio: int = 1
) -> torch.Tensor:
    B, H_qk, T, K = k.shape
    H_do = do.shape[1]
    V = do.shape[3]
    assert H_do == H_qk * h_ratio, f"H_do ({H_do}) must equal H_qk * h_ratio ({H_qk} * {h_ratio} = {H_qk * h_ratio})"
    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if cu_seqlens is not None:
        batch_idx = 0
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    chunk_indices = chunk_indices.view(-1)
    BT = min(chunk_size, max(16, 2 ** math.ceil(math.log2(T))))
    NT = len(chunk_indices) // 2
    dv = torch.zeros_like(do).to(torch.float32)
    g_t = g
    for chunk_idx in range(NT):
        i_n = chunk_indices[chunk_idx * 2].item()
        i_t = chunk_indices[chunk_idx * 2 + 1].item()
        bos = cu_seqlens[i_n].item()
        eos = cu_seqlens[i_n + 1].item()
        T = eos - bos
        chunk_start_token = i_t * chunk_size
        chunk_end_token = min(chunk_start_token + chunk_size, T)
        chunk_len = chunk_end_token - chunk_start_token
        if chunk_len <= 0:
            continue
        global_start = bos + chunk_start_token
        for qk_head in range(H_qk):
            b_A = torch.zeros(BT, BT, device=q.device, dtype=torch.float32)
            BK = 128
            BK = min(BK, K)
            for i_k in range(0, K, BK):
                k_end = min(i_k + BK, K)
                b_k = k[batch_idx, qk_head, global_start:global_start+chunk_len, i_k:k_end].to(torch.float32)
                q_normal = q[batch_idx, qk_head, global_start:global_start+chunk_len, i_k:k_end].to(torch.float32)
                b_q = q_normal.transpose(0, 1)
                if chunk_len == 1:
                    matmul_result = torch.sum(b_k * q_normal)
                    b_A[:chunk_len, :chunk_len] += matmul_result
                else:
                    b_A[:chunk_len, :chunk_len] += torch.matmul(b_k, b_q)
            for do_group in range(h_ratio):
                do_head = qk_head * h_ratio + do_group
                b_g = g_t[batch_idx, do_head, global_start:global_start+chunk_len]
                o_t = i_t * BT + torch.arange(0, BT)
                m_t = o_t < T
                o_t_col = o_t.unsqueeze(1)
                o_t_row = o_t.unsqueeze(0)
                pos_mask = o_t_col <= o_t_row
                m_t_col = m_t.unsqueeze(1)
                valid_mask = m_t_col & m_t
                m_A = pos_mask & valid_mask
                g_i = b_g.unsqueeze(1)
                g_j = b_g.unsqueeze(0)
                g_factor = torch.exp(g_j - g_i)
                b_A_gated = torch.zeros_like(b_A)
                b_A_gated[:chunk_len, :chunk_len] = b_A[:chunk_len, :chunk_len] * g_factor * scale
                b_A_masked = torch.where(m_A, b_A_gated, torch.zeros_like(b_A_gated))
                b_A_masked = b_A_masked.to(torch.float32)
                BV = 128
                BV = min(BV, V)
                for i_v in range(0, V, BV):
                    v_end = min(i_v + BV, V)
                    b_do = do[batch_idx, do_head, global_start:global_start+chunk_len, i_v:v_end].to(torch.float32)
                    b_dv = torch.matmul(b_A_masked[:chunk_len, :chunk_len], b_do)
                    dv[batch_idx, do_head, global_start:global_start+chunk_len, i_v:v_end] += b_dv
    return dv


def test_chunk_bwd_dv_local_interface_exist():
    """
    Test that the 'ascend_ops.chunk_bwd_dv_local' operator is present in torch.ops.
    This existence test asserts that the custom operator registered under the
    'ascend_ops' namespace is discoverable from Python via torch.ops.ascend_ops.chunk_bwd_dv_local.
    It does not exercise operator functionality — only that the Python binding
    and registration are available.
    """
    print(torch.ops.ascend_ops.chunk_bwd_dv_local)
    assert hasattr(torch.ops.ascend_ops, "chunk_bwd_dv_local"), \
        "The 'chunk_bwd_dv_local' operator is not registered in the 'torch.ops.ascend_ops' namespace."


FIX_TEST_CONFIGS = [
    (2, 2, 65, 128, 128, 64, 0.0625, torch.float16, torch.float16, 1),
    (2, 2, 65, 128, 128, 64, 0.0625, torch.bfloat16, torch.bfloat16, 1),
    (2, 2, 65, 128, 128, 64, 0.0625, torch.bfloat16, torch.float32, 1),
    (2, 2, 65, 128, 128, 64, 0.0625, torch.float16, torch.float32, 1),
    (2, 2, 65, 128, 128, 64, 0.0625, torch.bfloat16, torch.bfloat16, 2),
    (2, 2, 65, 128, 128, 64, 0.0625, torch.bfloat16, torch.bfloat16, 4),
]

VARIABLE_TEST_CONFIGS = [
    (1, 2, 64, 128, 128, 64, 0.011, 2, torch.float16, torch.float16, 1),
    (1, 2, 64, 128, 128, 64, 0.011, 2, torch.bfloat16, torch.bfloat16, 1),
    (1, 2, 64, 128, 128, 64, 0.011, 2, torch.bfloat16, torch.float32, 1),
    (1, 2, 64, 128, 128, 64, 0.011, 2, torch.float16, torch.float32, 1),
    (1, 2, 64, 128, 128, 64, 0.011, 2, torch.bfloat16, torch.bfloat16, 2),
    (1, 2, 64, 128, 128, 64, 0.011, 2, torch.bfloat16, torch.bfloat16, 4),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("B,H_qk,T,K,V,chunk_size,scale,ktype,gtype,h_ratio", FIX_TEST_CONFIGS)
def test_chunk_bwd_dv_local_fix(B, H_qk, T, K, V, chunk_size, scale, ktype, gtype, h_ratio):
    """
    Test the chunk_bwd_dv_local operator with fixed-length sequences.
    Compares NPU kernel launch result against PyTorch golden implementation.
    Supports GVA: H_do = H_qk * h_ratio.
    """
    torch.manual_seed(0)

    H_do = H_qk * h_ratio
    q = create_tensor((B, H_qk, T, K), dtype=ktype)
    k = create_tensor((B, H_qk, T, K), dtype=ktype)
    d_o = create_tensor((B, H_do, T, V), dtype=ktype)
    g = torch.arange(B * H_do * T, 0, -1).reshape((B, H_do, T)).to(gtype)
    print(" q.shape = ",q.shape, " d_o.shape = ", d_o.shape, " h_ratio = ", h_ratio)
    dv_golden = chunk_bwd_dv_local_fix(q, k, d_o, g, scale, None, chunk_size, h_ratio)
    print(" dv_golden.shape = ",dv_golden.shape)
    q_npu = q.npu()
    k_npu = k.npu()
    d_o_npu = d_o.npu()
    g_npu = g.npu()

    dv = torch.ops.ascend_ops.chunk_bwd_dv_local(
        q_npu, k_npu, d_o_npu, g_npu,
        scale, chunk_size,
        g_gamma=None, A=None, cu_seqlens=None, chunk_indices=None
    )

    dv_cpu = dv.cpu()
    dv_golden_f32 = dv_golden.to(torch.float32)
    dv_cpu_f32 = dv_cpu.to(torch.float32)

    max_diff = torch.max(torch.abs(dv_cpu_f32 - dv_golden_f32)).item()
    rtol = 1e-2
    atol = 1e-2
    assert torch.allclose(dv_cpu_f32, dv_golden_f32, rtol=rtol, atol=atol), \
        f"chunk_bwd_dv_local_fix failed for B={B}, H_qk={H_qk}, H_do={H_do}, T={T}, K={K}, V={V}, " \
        f"chunk_size={chunk_size}, scale={scale}, ktype={ktype}, gtype={gtype}, h_ratio={h_ratio}. " \
        f"Max diff: {max_diff:.6f}"

    print(f"✓ fix test passed: B={B}, H_qk={H_qk}, H_do={H_do}, T={T}, K={K}, V={V}, "
          f"chunk_size={chunk_size}, ktype={ktype}, gtype={gtype}, h_ratio={h_ratio}, max_diff={max_diff:.6f}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("B,H_qk,T,K,V,chunk_size,scale,cu_seqlens_len,ktype,gtype,h_ratio", VARIABLE_TEST_CONFIGS)
def test_chunk_bwd_dv_local_variable(B, H_qk, T, K, V, chunk_size, scale, cu_seqlens_len, ktype, gtype, h_ratio):
    """
    Test the chunk_bwd_dv_local operator with variable-length sequences.
    Compares NPU kernel launch result against PyTorch golden implementation.
    Supports GVA: H_do = H_qk * h_ratio.
    """
    torch.manual_seed(0)

    H_do = H_qk * h_ratio
    q = create_tensor((B, H_qk, T, K), dtype=ktype)
    k = create_tensor((B, H_qk, T, K), dtype=ktype)
    d_o = create_tensor((B, H_do, T, V), dtype=ktype)
    g = torch.arange(B * H_do * T, 0, -1).reshape((B, H_do, T)).to(gtype)

    cu_seqlens = generate_cu_seqlens(cu_seqlens_len, T)
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    dv_golden = chunk_bwd_dv_local_variable(q, k, d_o, g, scale, cu_seqlens, chunk_size, h_ratio)

    q_npu = q.npu()
    k_npu = k.npu()
    d_o_npu = d_o.npu()
    g_npu = g.npu()

    cu_seqlens_list = cu_seqlens.tolist()
    chunk_indices_list = chunk_indices.flatten().tolist()

    dv = torch.ops.ascend_ops.chunk_bwd_dv_local(
        q_npu, k_npu, d_o_npu, g_npu,
        scale, chunk_size,
        g_gamma=None, A=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices_list
    )

    dv_cpu = dv.cpu()
    dv_golden_f32 = dv_golden.to(torch.float32)
    dv_cpu_f32 = dv_cpu.to(torch.float32)

    max_diff = torch.max(torch.abs(dv_cpu_f32 - dv_golden_f32)).item()
    rtol = 1e-2
    atol = 1e-2
    assert torch.allclose(dv_cpu_f32, dv_golden_f32, rtol=rtol, atol=atol), \
        f"chunk_bwd_dv_local_variable failed for B={B}, H_qk={H_qk}, H_do={H_do}, T={T}, K={K}, V={V}, " \
        f"chunk_size={chunk_size}, scale={scale}, cu_seqlens_len={cu_seqlens_len}, " \
        f"ktype={ktype}, gtype={gtype}, h_ratio={h_ratio}. Max diff: {max_diff:.6f}"

    print(f"✓ variable test passed: B={B}, H_qk={H_qk}, H_do={H_do}, T={T}, K={K}, V={V}, "
          f"chunk_size={chunk_size}, cu_seqlens_len={cu_seqlens_len}, "
          f"ktype={ktype}, gtype={gtype}, h_ratio={h_ratio}, max_diff={max_diff:.6f}")
