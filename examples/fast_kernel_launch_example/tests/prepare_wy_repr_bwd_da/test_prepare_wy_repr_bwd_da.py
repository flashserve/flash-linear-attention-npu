#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest
import torch
import torch_npu

import ascend_ops


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def prepare_chunk_indices(cu_seqlens, chunk_size):
    indices = []
    for seq_idx in range(len(cu_seqlens) - 1):
        seq_len = cu_seqlens[seq_idx + 1] - cu_seqlens[seq_idx]
        for chunk_idx in range(cdiv(seq_len, chunk_size)):
            indices.extend([seq_idx, chunk_idx])
    return indices


def get_bos_eos(idx, T, chunk_size, cu_seqlens, chunk_indices):
    if cu_seqlens is None:
        bos = idx * chunk_size
        eos = min(bos + chunk_size, T)
        return bos, eos
    seq_idx = chunk_indices[idx * 2]
    chunk_idx = chunk_indices[idx * 2 + 1]
    bos = cu_seqlens[seq_idx] + chunk_idx * chunk_size
    eos = min(bos + chunk_size, cu_seqlens[seq_idx + 1])
    return bos, eos


def make_inputs(B, HK, HV, T, K, V, chunk_size, k_dtype, beta_dtype, variable=False):
    torch.manual_seed(0)
    actual_B = 1 if variable else B
    k = torch.rand(actual_B, HK, T, K, dtype=k_dtype) * 0.1
    v = torch.rand(actual_B, HV, T, V, dtype=k_dtype) * 0.1
    beta = torch.rand(actual_B, HV, T, dtype=beta_dtype) * 0.1
    A = torch.rand(actual_B, HV, T, chunk_size, dtype=k_dtype) * 0.1
    dw = torch.rand(actual_B, HV, T, K, dtype=k_dtype) * 0.1
    du = torch.rand(actual_B, HV, T, V, dtype=k_dtype) * 0.1
    g = torch.rand(actual_B, HV, T, dtype=beta_dtype) * 0.1
    if not variable:
        return k, v, beta, A, dw, du, g, None, None
    cu_seqlens = [0, T // 3, T]
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    return k, v, beta, A, dw, du, g, cu_seqlens, chunk_indices


def prepare_wy_repr_bwd_da_ref(k, v, beta, A, dw, du, g, chunk_size, cu_seqlens=None, chunk_indices=None):
    out_dtype = A.dtype
    k_f = k.to(torch.float64)
    v_f = v.to(torch.float64)
    beta_f = beta.to(torch.float64)
    A_f = A.to(torch.float64)
    dw_f = dw.to(torch.float64)
    du_f = du.to(torch.float64)
    g_f = g.to(torch.float64)

    B, HK, T, K = k.shape
    _, HV, _, _ = v.shape
    heads_per_kv = HV // HK
    NT = len(chunk_indices) // 2 if chunk_indices is not None else cdiv(T, chunk_size)
    dA = torch.zeros_like(A_f)

    for b in range(B):
        for idx in range(NT):
            bos, eos = get_bos_eos(idx, T, chunk_size, cu_seqlens, chunk_indices)
            chunk_len = eos - bos
            if chunk_len <= 0:
                continue
            local_t = torch.arange(0, chunk_size, dtype=torch.int64)
            if cu_seqlens is None:
                chunk_idx = idx
                cur_t = T
            else:
                seq_idx = chunk_indices[idx * 2]
                chunk_idx = chunk_indices[idx * 2 + 1]
                cur_t = cu_seqlens[seq_idx + 1] - cu_seqlens[seq_idx]
            o_t = chunk_idx * chunk_size + local_t
            m_t = o_t < cur_t
            m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)

            for hv in range(HV):
                hk = hv // heads_per_kv
                A_chunk = A_f[b, hv, bos:eos, :chunk_len]
                dw_chunk = dw_f[b, hv, bos:eos, :]
                k_chunk = k_f[b, hk, bos:eos, :]
                beta_chunk = beta_f[b, hv, bos:eos]
                g_chunk = g_f[b, hv, bos:eos]
                du_chunk = du_f[b, hv, bos:eos, :]
                v_chunk = v_f[b, hv, bos:eos, :]

                g_exp_chunk = torch.exp(g_chunk)
                b_k_beta_g = k_chunk * (beta_chunk * g_exp_chunk)[:, None]
                b_dA_1 = dw_chunk @ b_k_beta_g.T
                b_v_beta = v_chunk * beta_chunk[:, None]
                b_dA_2 = du_chunk @ b_v_beta.T
                b_dA_3 = b_dA_1 + b_dA_2
                b_dA_4 = torch.where(m_A[:chunk_len, :chunk_len], b_dA_3, 0.0)
                b_dA_5 = b_dA_4 @ A_chunk.T
                b_dA_6 = A_chunk.T @ b_dA_5
                b_g_sub_exp = torch.exp(g_chunk[:, None] - g_chunk[None, :])
                b_dA_7 = -b_dA_6 * b_g_sub_exp
                b_dA = torch.where(m_A[:chunk_len, :chunk_len], b_dA_7, 0.0)
                dA[b, hv, bos:eos, :chunk_len] = b_dA.T

    return dA.to(out_dtype)


def assert_close(actual, expected):
    actual_f32 = actual.cpu().to(torch.float32)
    expected_f32 = expected.cpu().to(torch.float32)
    max_diff = torch.max(torch.abs(actual_f32 - expected_f32)).item()
    assert torch.allclose(actual_f32, expected_f32, rtol=1e-1, atol=1e-1), f"max diff: {max_diff:.6f}"


def test_prepare_wy_repr_bwd_da_interface_exist():
    print(torch.ops.ascend_ops.prepare_wy_repr_bwd_da)
    assert hasattr(torch.ops.ascend_ops, "prepare_wy_repr_bwd_da")


FIXED_CONFIGS = [
    (1, 2, 2, 65, 128, 128, 64, torch.float16, torch.float16),
    (1, 2, 4, 65, 128, 128, 64, torch.float16, torch.float32),
    (1, 2, 2, 129, 128, 256, 128, torch.bfloat16, torch.float32),
]

VARIABLE_CONFIGS = [
    (1, 2, 2, 65, 128, 128, 64, torch.float16, torch.float16),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("B,HK,HV,T,K,V,chunk_size,k_dtype,beta_dtype", FIXED_CONFIGS)
def test_prepare_wy_repr_bwd_da_fixed(B, HK, HV, T, K, V, chunk_size, k_dtype, beta_dtype):
    inputs = make_inputs(B, HK, HV, T, K, V, chunk_size, k_dtype, beta_dtype, variable=False)
    k, v, beta, A, dw, du, g, cu_seqlens, chunk_indices = inputs
    expected = prepare_wy_repr_bwd_da_ref(k, v, beta, A, dw, du, g, chunk_size, cu_seqlens, chunk_indices)
    actual = torch.ops.ascend_ops.prepare_wy_repr_bwd_da(
        k.npu(), v.npu(), beta.npu(), A.npu(), dw.npu(), du.npu(), g.npu(), chunk_size
    )
    assert_close(actual, expected)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("B,HK,HV,T,K,V,chunk_size,k_dtype,beta_dtype", VARIABLE_CONFIGS)
def test_prepare_wy_repr_bwd_da_variable(B, HK, HV, T, K, V, chunk_size, k_dtype, beta_dtype):
    inputs = make_inputs(B, HK, HV, T, K, V, chunk_size, k_dtype, beta_dtype, variable=True)
    k, v, beta, A, dw, du, g, cu_seqlens, chunk_indices = inputs
    expected = prepare_wy_repr_bwd_da_ref(k, v, beta, A, dw, du, g, chunk_size, cu_seqlens, chunk_indices)
    actual = torch.ops.ascend_ops.prepare_wy_repr_bwd_da(
        k.npu(),
        v.npu(),
        beta.npu(),
        A.npu(),
        dw.npu(),
        du.npu(),
        g.npu(),
        chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    assert_close(actual, expected)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_prepare_wy_repr_bwd_da_variable_metadata_pair_required():
    inputs = make_inputs(1, 2, 2, 65, 128, 128, 64, torch.float16, torch.float16, variable=True)
    k, v, beta, A, dw, du, g, cu_seqlens, chunk_indices = inputs
    npu_args = (k.npu(), v.npu(), beta.npu(), A.npu(), dw.npu(), du.npu(), g.npu(), 64)

    with pytest.raises(RuntimeError, match="cu_seqlens and chunk_indices"):
        torch.ops.ascend_ops.prepare_wy_repr_bwd_da(*npu_args, cu_seqlens=cu_seqlens, chunk_indices=None)

    with pytest.raises(RuntimeError, match="cu_seqlens and chunk_indices"):
        torch.ops.ascend_ops.prepare_wy_repr_bwd_da(*npu_args, cu_seqlens=None, chunk_indices=chunk_indices)
