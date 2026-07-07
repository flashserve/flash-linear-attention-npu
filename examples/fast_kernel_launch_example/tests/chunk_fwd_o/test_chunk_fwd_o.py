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

import pytest
import torch
import torch_npu

import ascend_ops


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def prepare_chunk_offsets(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    offsets = []
    for batch_idx in range(cu_seqlens.numel() - 1):
        seq_len = int(cu_seqlens[batch_idx + 1] - cu_seqlens[batch_idx])
        for chunk_idx in range(cdiv(seq_len, chunk_size)):
            offsets.append([batch_idx, chunk_idx])
    return torch.tensor(offsets, dtype=torch.long)


def generate_cu_seqlens(total_length: int, token_batch: int) -> torch.LongTensor:
    if token_batch == 1:
        return torch.tensor([0, total_length], dtype=torch.long)
    base = total_length // token_batch
    lengths = [base] * token_batch
    lengths[-1] += total_length - sum(lengths)
    cu_seqlens = [0]
    for length in lengths:
        cu_seqlens.append(cu_seqlens[-1] + length)
    return torch.tensor(cu_seqlens, dtype=torch.long)


def make_inputs(
    shape_batch,
    k_heads,
    v_heads,
    seq_len,
    k_dim,
    v_dim,
    chunk_size,
    input_dtype,
    g_dtype,
    variable,
    token_batch=1,
):
    torch.manual_seed(0)
    actual_shape_batch = 1 if variable else shape_batch
    cu_seqlens = generate_cu_seqlens(seq_len, token_batch) if variable else None
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, chunk_size) if variable else None
    num_chunks = chunk_offsets.shape[0] if variable else cdiv(seq_len, chunk_size)

    q = torch.randn((actual_shape_batch, k_heads, seq_len, k_dim), dtype=input_dtype) * 0.1
    k = torch.randn((actual_shape_batch, k_heads, seq_len, k_dim), dtype=input_dtype) * 0.1
    v = torch.randn((actual_shape_batch, v_heads, seq_len, v_dim), dtype=input_dtype) * 0.1
    h = torch.randn((actual_shape_batch, v_heads, num_chunks, k_dim, v_dim), dtype=input_dtype) * 0.1
    g_base = torch.linspace(-0.25, 0.25, seq_len, dtype=torch.float32).reshape(1, 1, seq_len)
    g = g_base.repeat(actual_shape_batch, v_heads, 1).to(g_dtype)
    return q, k, v, h, g, cu_seqlens, chunk_offsets


def chunk_fwd_o_ref(q, k, v, h, g, scale, chunk_size, cu_seqlens=None, chunk_offsets=None):
    out_dtype = v.dtype
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    h = h.to(torch.float32)
    g = g.to(torch.float32)

    shape_batch, k_heads, seq_len, _ = q.shape
    _, v_heads, _, v_dim = v.shape
    head_ratio = v_heads // k_heads
    output = torch.zeros((shape_batch, v_heads, seq_len, v_dim), dtype=torch.float32)

    if cu_seqlens is None:
        for batch_idx in range(shape_batch):
            for v_head_idx in range(v_heads):
                k_head_idx = v_head_idx // head_ratio
                for chunk_idx in range(cdiv(seq_len, chunk_size)):
                    start = chunk_idx * chunk_size
                    end = min(start + chunk_size, seq_len)
                    q_block = q[batch_idx, k_head_idx, start:end]
                    k_block = k[batch_idx, k_head_idx, start:end]
                    v_block = v[batch_idx, v_head_idx, start:end]
                    g_block = g[batch_idx, v_head_idx, start:end]
                    h_block = h[batch_idx, v_head_idx, chunk_idx]
                    output[batch_idx, v_head_idx, start:end] = compute_block(
                        q_block, k_block, v_block, h_block, g_block, scale
                    )
    else:
        flat_offsets = chunk_offsets.reshape(-1, 2)
        for global_chunk_idx, (token_batch_idx, batch_chunk_idx) in enumerate(flat_offsets.tolist()):
            bos = int(cu_seqlens[token_batch_idx])
            eos = int(cu_seqlens[token_batch_idx + 1])
            start = bos + batch_chunk_idx * chunk_size
            end = min(start + chunk_size, eos)
            for v_head_idx in range(v_heads):
                k_head_idx = v_head_idx // head_ratio
                q_block = q[0, k_head_idx, start:end]
                k_block = k[0, k_head_idx, start:end]
                v_block = v[0, v_head_idx, start:end]
                g_block = g[0, v_head_idx, start:end]
                h_block = h[0, v_head_idx, global_chunk_idx]
                output[0, v_head_idx, start:end] = compute_block(q_block, k_block, v_block, h_block, g_block, scale)

    return output.to(out_dtype)


def compute_block(q_block, k_block, v_block, h_block, g_block, scale):
    attn = q_block @ k_block.transpose(-1, -2)
    decay = torch.exp(g_block.unsqueeze(-1) - g_block.unsqueeze(-2))
    causal = torch.tril(torch.ones_like(attn, dtype=torch.bool))
    attn = torch.where(causal, attn * decay, torch.zeros_like(attn))
    o_inter = (q_block * torch.exp(g_block).unsqueeze(-1)) @ h_block
    return (o_inter + attn @ v_block) * scale


def assert_close(actual, expected):
    actual_f32 = actual.cpu().to(torch.float32)
    expected_f32 = expected.cpu().to(torch.float32)
    max_diff = torch.max(torch.abs(actual_f32 - expected_f32)).item()
    assert torch.allclose(actual_f32, expected_f32, rtol=2e-2, atol=2e-2), f"max diff: {max_diff:.6f}"


def test_chunk_fwd_o_interface_exist():
    print(torch.ops.ascend_ops.chunk_fwd_o)
    assert hasattr(torch.ops.ascend_ops, "chunk_fwd_o")


FIX_TEST_CONFIGS = [
    (2, 2, 2, 65, 128, 128, 64, 0.08838834764831845, torch.float16, torch.float16),
    (2, 2, 2, 65, 128, 128, 64, 0.08838834764831845, torch.bfloat16, torch.bfloat16),
    (2, 2, 2, 65, 128, 128, 64, 0.08838834764831845, torch.float16, torch.float32),
    (2, 2, 2, 65, 128, 128, 64, 0.08838834764831845, torch.bfloat16, torch.float32),
]

VARIABLE_TEST_CONFIGS = [
    (1, 2, 2, 65, 128, 128, 64, 0.08838834764831845, 3, torch.float16, torch.float16),
    (1, 2, 2, 65, 128, 128, 64, 0.08838834764831845, 3, torch.bfloat16, torch.bfloat16),
    (1, 2, 2, 65, 128, 128, 64, 0.08838834764831845, 3, torch.float16, torch.float32),
    (1, 2, 2, 65, 128, 128, 64, 0.08838834764831845, 3, torch.bfloat16, torch.float32),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("B,HK,HV,T,K,V,chunk_size,scale,input_dtype,g_dtype", FIX_TEST_CONFIGS)
def test_chunk_fwd_o_fix(B, HK, HV, T, K, V, chunk_size, scale, input_dtype, g_dtype):
    q, k, v, h, g, cu_seqlens, chunk_offsets = make_inputs(
        B, HK, HV, T, K, V, chunk_size, input_dtype, g_dtype, variable=False
    )
    expected = chunk_fwd_o_ref(q, k, v, h, g, scale, chunk_size, cu_seqlens, chunk_offsets)

    actual = torch.ops.ascend_ops.chunk_fwd_o(
        q.npu(), k.npu(), v.npu(), h.npu(), g.npu(), scale, chunk_size, cu_seqlens=None, chunk_offsets=None
    )

    assert_close(actual, expected)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("B,HK,HV,T,K,V,chunk_size,scale,token_batch,input_dtype,g_dtype", VARIABLE_TEST_CONFIGS)
def test_chunk_fwd_o_variable(B, HK, HV, T, K, V, chunk_size, scale, token_batch, input_dtype, g_dtype):
    q, k, v, h, g, cu_seqlens, chunk_offsets = make_inputs(
        B, HK, HV, T, K, V, chunk_size, input_dtype, g_dtype, variable=True, token_batch=token_batch
    )
    expected = chunk_fwd_o_ref(q, k, v, h, g, scale, chunk_size, cu_seqlens, chunk_offsets)

    actual = torch.ops.ascend_ops.chunk_fwd_o(
        q.npu(),
        k.npu(),
        v.npu(),
        h.npu(),
        g.npu(),
        scale,
        chunk_size,
        cu_seqlens=cu_seqlens.tolist(),
        chunk_offsets=chunk_offsets.flatten().tolist(),
    )

    assert_close(actual, expected)
