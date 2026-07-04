#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import random

import pytest
import torch
import torch_npu

import ascend_ops


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def get_bos_eos(idx, T, chunk_size, cu_seqlens, chunk_indices):
    if cu_seqlens is not None:
        seq_idx = chunk_indices[idx * 2]
        chunk_idx = chunk_indices[idx * 2 + 1]
        bos = cu_seqlens[seq_idx] + chunk_idx * chunk_size
        eos = bos + chunk_size
        if eos > cu_seqlens[seq_idx + 1]:
            eos = cu_seqlens[seq_idx + 1]
    else:
        bos = idx * chunk_size
        eos = bos + chunk_size
        if eos > T:
            eos = T
    return bos, eos


def compute_w_golden(k, v, beta, A, g, cu_seqlens, chunk_indices, B, H, T, D, chunk_size, NT, Hk=0):
    if Hk == 0:
        Hk = H
    hv_per_hk = H // Hk
    w = torch.zeros(B, H, T, D, dtype=v.dtype, device=v.device)
    for i_b in range(B):
        for idx in range(NT):
            bos, eos = get_bos_eos(idx, T, chunk_size, cu_seqlens, chunk_indices)
            for i_h in range(H):
                hk = i_h // hv_per_hk
                a_chunk = A[i_b, i_h, bos:eos, : eos - bos]
                k_chunk = k[i_b, hk, bos:eos, :]
                beta_chunk = beta[i_b, i_h, bos:eos]
                g_chunk = g[i_b, i_h, bos:eos]
                g_exp_chunk = torch.exp(g_chunk.to(torch.float32))
                beta_g_exp_chunk = beta_chunk.to(torch.float32) * g_exp_chunk
                kbgexp_chunk = k_chunk * beta_g_exp_chunk[:, None]
                b_w = torch.matmul(a_chunk.to(torch.float32), kbgexp_chunk.to(torch.float32))
                w[i_b, i_h, bos:eos, :] = b_w.to(w.dtype)
    return w


def compute_u_golden(v, beta, A, cu_seqlens, chunk_indices, B, Hv, T, chunk_size, NT):
    u = torch.zeros_like(v)
    for i_b in range(B):
        for idx in range(NT):
            bos, eos = get_bos_eos(idx, T, chunk_size, cu_seqlens, chunk_indices)
            for i_h in range(Hv):
                a_chunk = A[i_b, i_h, bos:eos, : eos - bos]
                v_chunk = v[i_b, i_h, bos:eos, :]
                beta_chunk = beta[i_b, i_h, bos:eos]
                vb_chunk = v_chunk.to(torch.float32) * beta_chunk[:, None].to(torch.float32)
                u_chunk = torch.matmul(a_chunk.to(torch.float32), vb_chunk.to(torch.float32))
                u[i_b, i_h, bos:eos, :] = u_chunk.to(u.dtype)
    return u


def prepare_cu_seqlens(T: int, L: int = 3, seed: int = 42):
    random.seed(seed)
    if L == 2:
        return [0, T]
    middle_points = random.sample(range(1, T), L - 2)
    middle_points.sort()
    return [0] + middle_points + [T]


def prepare_chunk_indices(cu_seqlens, chunk_size: int):
    indices = []
    for i in range(len(cu_seqlens) - 1):
        length = cu_seqlens[i + 1] - cu_seqlens[i]
        if length <= 0:
            continue
        num_chunks = cdiv(length, chunk_size)
        for chunk_id in range(num_chunks):
            indices.append(i)
            indices.append(chunk_id)
    return indices


def run_recompute_wu_fwd(k, v, beta, A, g, chunk_size, cu_seqlens=None, chunk_indices=None):
    return torch.ops.ascend_ops.recompute_wu_fwd(
        k.npu(),
        v.npu(),
        beta.npu(),
        A.npu(),
        chunk_size,
        g=g.npu(),
        gk=None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )


def test_recompute_wu_fwd_interface_exist():
    assert hasattr(torch.ops.ascend_ops, "recompute_wu_fwd")


FIX_TEST_CONFIGS = [
    (1, 2, 2, 129, 128, 128, 128, torch.float16, torch.float16),
    (2, 2, 2, 65, 128, 128, 64, torch.float16, torch.float16),
    (2, 2, 2, 65, 128, 128, 64, torch.bfloat16, torch.bfloat16),
    (2, 2, 2, 65, 128, 128, 64, torch.float16, torch.float32),
    (2, 2, 2, 65, 128, 128, 64, torch.bfloat16, torch.float32),
    (1, 2, 4, 256, 128, 256, 64, torch.float16, torch.float16),
    (2, 2, 4, 128, 128, 128, 64, torch.float16, torch.float16),
]

VARIABLE_TEST_CONFIGS = [
    (1, 2, 4, 128, 128, 128, 64, 3, torch.float16, torch.float16),
    (1, 2, 4, 128, 128, 128, 64, 3, torch.bfloat16, torch.bfloat16),
    (1, 2, 4, 256, 128, 256, 64, 2, torch.float16, torch.float16),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("B,Hk,Hv,T,K,V,chunk_size,ktype,btype", FIX_TEST_CONFIGS)
def test_recompute_wu_fwd_fix(B, Hk, Hv, T, K, V, chunk_size, ktype, btype):
    torch.manual_seed(0)
    k = torch.randn(B, Hk, T, K, dtype=ktype)
    v = torch.randn(B, Hv, T, V, dtype=ktype)
    beta = torch.randn(B, Hv, T, dtype=btype)
    A = torch.randn(B, Hv, T, chunk_size, dtype=ktype)
    g = torch.randn(B, Hv, T, dtype=btype)
    NT = cdiv(T, chunk_size)

    cpu_w = compute_w_golden(k, v, beta, A, g, None, None, B, Hv, T, K, chunk_size, NT, Hk=Hk)
    cpu_u = compute_u_golden(v, beta, A, None, None, B, Hv, T, chunk_size, NT)

    w, u = run_recompute_wu_fwd(k, v, beta, A, g, chunk_size)

    rtol = 1e-1
    atol = 1e-1
    assert torch.allclose(w.cpu().float(), cpu_w.float(), rtol=rtol, atol=atol)
    assert torch.allclose(u.cpu().float(), cpu_u.float(), rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("B,Hk,Hv,T,K,V,chunk_size,cu_seqlens_len,ktype,btype", VARIABLE_TEST_CONFIGS)
def test_recompute_wu_fwd_variable(B, Hk, Hv, T, K, V, chunk_size, cu_seqlens_len, ktype, btype):
    torch.manual_seed(0)
    k = torch.randn(B, Hk, T, K, dtype=ktype)
    v = torch.randn(B, Hv, T, V, dtype=ktype)
    beta = torch.randn(B, Hv, T, dtype=btype)
    A = torch.randn(B, Hv, T, chunk_size, dtype=ktype)
    g = torch.randn(B, Hv, T, dtype=btype)

    cu_seqlens = prepare_cu_seqlens(T, cu_seqlens_len)
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = len(chunk_indices) // 2

    cpu_w = compute_w_golden(k, v, beta, A, g, cu_seqlens, chunk_indices, B, Hv, T, K, chunk_size, NT, Hk=Hk)
    cpu_u = compute_u_golden(v, beta, A, cu_seqlens, chunk_indices, B, Hv, T, chunk_size, NT)

    w, u = run_recompute_wu_fwd(k, v, beta, A, g, chunk_size, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)

    rtol = 1e-1
    atol = 1e-1
    assert torch.allclose(w.cpu().float(), cpu_w.float(), rtol=rtol, atol=atol)
    assert torch.allclose(u.cpu().float(), cpu_u.float(), rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_recompute_wu_fwd_variable_metadata_pair_required():
    torch.manual_seed(0)
    B, Hk, Hv, T, K, V, chunk_size = 1, 2, 4, 128, 128, 128, 64
    k = torch.randn(B, Hk, T, K, dtype=torch.float16)
    v = torch.randn(B, Hv, T, V, dtype=torch.float16)
    beta = torch.randn(B, Hv, T, dtype=torch.float16)
    A = torch.randn(B, Hv, T, chunk_size, dtype=torch.float16)
    g = torch.randn(B, Hv, T, dtype=torch.float16)
    cu_seqlens = prepare_cu_seqlens(T, 3)

    with pytest.raises(RuntimeError, match="cu_seqlens and chunk_indices"):
        run_recompute_wu_fwd(k, v, beta, A, g, chunk_size, cu_seqlens=cu_seqlens, chunk_indices=None)

    with pytest.raises(RuntimeError, match="cu_seqlens and chunk_indices"):
        run_recompute_wu_fwd(k, v, beta, A, g, chunk_size, cu_seqlens=None, chunk_indices=[0, 0])
