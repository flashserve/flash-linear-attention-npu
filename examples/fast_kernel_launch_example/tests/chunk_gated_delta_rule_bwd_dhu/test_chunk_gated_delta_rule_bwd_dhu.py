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

import math
import random
from typing import List, Optional, Tuple

import ascend_ops
import pytest
import torch
import torch_npu

_LOW_PRECISION_INPUT_HALF_RANGE_QK = 6.5e-3
_LOW_PRECISION_INPUT_HALF_RANGE_WO = 6.5e-3
_LOW_PRECISION_INPUT_HALF_RANGE_DV = 9e-3
_LOW_PRECISION_SCALE_FACTOR = 0.92


def prepare_chunk_indices(cu_seqlens: List[int], chunk_size: int = 64) -> List[int]:
    chunk_indices = []
    for seq_idx in range(len(cu_seqlens) - 1):
        seq_len = cu_seqlens[seq_idx + 1] - cu_seqlens[seq_idx]
        chunk_num = (seq_len + chunk_size - 1) // chunk_size
        for chunk_idx in range(chunk_num):
            chunk_indices.append(seq_idx)
            chunk_indices.append(chunk_idx)
    return chunk_indices


def create_gate_g(B: int, Hv: int, T: int, gtype, narrow: bool = False):
    if narrow:
        lo, hi = -1e-2, -1e-6
    else:
        lo, hi = -5e-2, -5e-5
    span = hi - lo
    margin = max(span * 1e-7, 1e-12)
    g_t = torch.linspace(float(hi) - margin, float(lo) + margin, T, dtype=torch.float64)
    g = g_t.unsqueeze(0).unsqueeze(0).expand(B, Hv, T).contiguous().to(gtype)
    return g


def generate_cu_seqlens(cu_seqlens_len: int, total_length: int, seg_min: int = 64, seg_max: int = 128) -> List[int]:
    batchsize = cu_seqlens_len - 1
    if batchsize <= 0:
        return [0, total_length]

    B = batchsize
    T = total_length
    lengths = [(T * (i + 1)) // B - (T * i) // B for i in range(B)]
    for i in range(B):
        lengths[i] = max(seg_min, min(seg_max, lengths[i]))

    diff = T - sum(lengths)
    while diff > 0:
        cand = [i for i in range(B) if lengths[i] < seg_max]
        if not cand:
            break
        i = min(cand, key=lambda j: lengths[j])
        lengths[i] += 1
        diff -= 1
    while diff < 0:
        cand = [i for i in range(B) if lengths[i] > seg_min]
        if not cand:
            break
        i = max(cand, key=lambda j: lengths[j])
        lengths[i] -= 1
        diff += 1

    sorted_l = sorted(lengths)
    seq_lengths: List[int] = []
    i, j = 0, len(sorted_l) - 1
    while i <= j:
        if i == j:
            seq_lengths.append(sorted_l[i])
        else:
            seq_lengths.append(sorted_l[i])
            seq_lengths.append(sorted_l[j])
        i += 1
        j -= 1

    cu_seqlens = [0]
    for seq_len in seq_lengths:
        cu_seqlens.append(cu_seqlens[-1] + seq_len)
    return cu_seqlens


def rand_symmetric_uniform(shape, dtype: torch.dtype, half_range: float) -> torch.Tensor:
    x = torch.rand(shape, dtype=torch.float32, device="cpu")
    x = (x * 2.0 - 1.0) * float(half_range)
    return x.to(dtype=dtype)


def create_bwd_dhu_random_inputs(
    B: int, Hk: int, Hv: int, T: int, K: int, V: int, ktype: torch.dtype, gtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    low = ktype in (torch.float16, torch.bfloat16)
    if low:
        hr_qk = _LOW_PRECISION_INPUT_HALF_RANGE_QK
        hr_wo = _LOW_PRECISION_INPUT_HALF_RANGE_WO
        hr_dv = _LOW_PRECISION_INPUT_HALF_RANGE_DV
        narrow_g = True
    else:
        hr_qk = 2e-2
        hr_wo = 2e-2
        hr_dv = 3e-2
        narrow_g = False

    q = rand_symmetric_uniform((B, Hk, T, K), ktype, half_range=hr_qk)
    k = rand_symmetric_uniform((B, Hk, T, K), ktype, half_range=hr_qk)
    w = rand_symmetric_uniform((B, Hv, T, K), ktype, half_range=hr_qk)
    d_o = rand_symmetric_uniform((B, Hv, T, V), ktype, half_range=hr_wo)
    dv = rand_symmetric_uniform((B, Hv, T, V), ktype, half_range=hr_dv)
    g = create_gate_g(B, Hv, T, gtype, narrow=narrow_g)
    return q, k, w, d_o, dv, g


def effective_scale(scale: float, K: int) -> float:
    cap = 1.0 / math.sqrt(float(K))
    return float(min(scale, cap))


def scale_for_compute_dtype(scale: float, ktype: torch.dtype) -> float:
    if ktype in (torch.float16, torch.bfloat16):
        return float(scale * _LOW_PRECISION_SCALE_FACTOR)
    return float(scale)


def chunk_gated_delta_rule_bwd_dhu_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    cu_seqlens: Optional[List[int]] = None,
    chunk_indices: Optional[List[int]] = None,
    g: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    device = q.device
    B, Hk, T, K = q.shape
    Hv = do.shape[1]
    V = do.shape[-1]
    BT = chunk_size
    hv_per_hk = Hv // Hk

    if cu_seqlens is not None:
        NT = len(chunk_indices) // 2
    else:
        NT = (T + BT - 1) // BT

    if scale is None:
        scale = 1.0

    chunk_info = []
    for i_t in range(NT):
        if cu_seqlens is not None:
            i_n = chunk_indices[i_t * 2]
            block_idx_in_token = chunk_indices[i_t * 2 + 1]
            bos = cu_seqlens[i_n]
            token_length = cu_seqlens[i_n + 1] - bos
        else:
            i_n = 0
            block_idx_in_token = i_t
            bos = 0
            token_length = T

        start_t = block_idx_in_token * BT
        end_t = min((block_idx_in_token + 1) * BT, token_length)
        block_size_t = end_t - start_t
        global_start_t = bos + start_t
        global_end_t = bos + end_t

        chunk_info.append({
            "i_t": i_t,
            "i_n": i_n,
            "block_idx_in_token": block_idx_in_token,
            "bos": bos,
            "token_length": token_length,
            "block_size_t": block_size_t,
            "global_start_t": global_start_t,
            "global_end_t": global_end_t,
        })

    dh = torch.zeros(B, Hv, NT, K, V, device=device, dtype=torch.float32)
    if cu_seqlens is not None:
        dv2 = dv.clone()
    else:
        dv2 = torch.zeros(B, Hv, T, V, device=device, dtype=dv.dtype)

    if cu_seqlens is None:
        hq = torch.arange(Hv, device=device, dtype=torch.long) // hv_per_hk
        b_dh = torch.zeros(B, Hv, K, V, device=device, dtype=torch.float32)
        for i_t in range(NT - 1, -1, -1):
            info = chunk_info[i_t]
            global_start_t = info["global_start_t"]
            global_end_t = info["global_end_t"]
            block_size_t = info["block_size_t"]

            dh[:, :, i_t, :, :] = b_dh

            last_idx = min((info["block_idx_in_token"] + 1) * BT, info["token_length"]) - 1
            global_last_idx = info["bos"] + last_idx

            k_blk = k[:, :, global_start_t:global_end_t, :].index_select(1, hq)
            q_blk = q[:, :, global_start_t:global_end_t, :].index_select(1, hq)
            w_blk = w[:, :, global_start_t:global_end_t, :]
            b_do = do[:, :, global_start_t:global_end_t, :]
            b_dv_existing = dv[:, :, global_start_t:global_end_t, :]

            b_dv = torch.matmul(k_blk.to(torch.float), b_dh.to(torch.float))

            if g is not None:
                bg_last = g[:, :, global_last_idx]
                b_g = g[:, :, global_start_t:global_end_t]
                gate_factor = torch.exp(bg_last.unsqueeze(-1) - b_g).unsqueeze(-1)
                m_t = torch.arange(block_size_t, device=device, dtype=torch.float32) < float(block_size_t)
                mask_expanded = m_t.view(1, 1, block_size_t, 1)
                b_dv = b_dv * gate_factor * mask_expanded

            b_dv = b_dv + b_dv_existing.to(torch.float32)
            dv2[:, :, global_start_t:global_end_t, :] = b_dv.to(dv2.dtype)

            b_q_t = q_blk.transpose(-1, -2)
            b_w_t = w_blk.transpose(-1, -2)

            if g is not None:
                bg_last_exp = torch.exp(bg_last)
                b_g_exp = torch.exp(b_g)
                b_dh_for_update = b_dh * bg_last_exp.unsqueeze(-1).unsqueeze(-1)
                b_q_gated = b_q_t * b_g_exp.unsqueeze(-2)
            else:
                b_dh_for_update = b_dh.clone()
                b_q_gated = b_q_t

            term1 = torch.matmul(b_q_gated.to(torch.float), b_do.to(torch.float)) * scale
            term2 = torch.matmul(b_w_t.to(torch.float), b_dv.to(torch.float))
            b_dh = b_dh_for_update + term1 - term2
    else:
        for b in range(B):
            for i_h in range(Hv):
                hq = i_h // hv_per_hk
                num_tokens = len(cu_seqlens) - 1
                b_dh_buffers = {i_n: torch.zeros(K, V, device=device, dtype=torch.float32) for i_n in range(num_tokens)}

                for i_t in range(NT - 1, -1, -1):
                    info = chunk_info[i_t]
                    i_n = info["i_n"]
                    b_dh = b_dh_buffers[i_n]
                    dh[b, i_h, i_t] = b_dh

                    global_start_t = info["global_start_t"]
                    global_end_t = info["global_end_t"]
                    block_size_t = info["block_size_t"]
                    last_idx = min((info["block_idx_in_token"] + 1) * BT, info["token_length"]) - 1
                    global_last_idx = info["bos"] + last_idx

                    bg_last = bg_last_exp = b_g = b_g_exp = None
                    if g is not None:
                        bg_last = g[b, i_h, global_last_idx]
                        b_g = g[b, i_h, global_start_t:global_end_t]
                        bg_last_exp = torch.exp(bg_last)
                        b_g_exp = torch.exp(b_g)

                    b_do = do[b, i_h, global_start_t:global_end_t, :]
                    b_dv_existing = dv[b, i_h, global_start_t:global_end_t, :]
                    b_k = k[b, hq, global_start_t:global_end_t, :]
                    b_dv = torch.matmul(b_k.to(torch.float), b_dh.to(torch.float))

                    if g is not None:
                        m_t = torch.arange(block_size_t, device=device) < block_size_t
                        gate_factor = torch.exp(bg_last - b_g).unsqueeze(-1)
                        mask_expanded = m_t.unsqueeze(-1).float()
                        b_dv *= gate_factor * mask_expanded

                    b_dv += b_dv_existing.to(torch.float32)
                    dv2[b, i_h, global_start_t:global_end_t, :] = b_dv.to(dv2.dtype)

                    b_q = q[b, hq, global_start_t:global_end_t, :]
                    b_w = w[b, i_h, global_start_t:global_end_t, :]
                    b_q_t = b_q.transpose(0, 1)
                    b_w_t = b_w.transpose(0, 1)
                    b_dh_for_update = b_dh.clone()
                    if g is not None:
                        b_dh_for_update = b_dh_for_update * bg_last_exp

                    b_q_gated = b_q_t
                    if g is not None:
                        b_q_gated = b_q_t * b_g_exp.unsqueeze(0)

                    term1 = torch.matmul(b_q_gated.to(torch.float), b_do.to(torch.float)) * scale
                    term2 = torch.matmul(b_w_t.to(torch.float), b_dv.to(torch.float))
                    b_dh_buffers[i_n] = b_dh_for_update + term1 - term2

    return dh, None, dv2


def run_chunk_gated_delta_rule_bwd_dhu(
    q, k, w, d_o, dv, g, scale, chunk_size, cu_seqlens=None, chunk_indices=None
):
    return torch.ops.ascend_ops.chunk_gated_delta_rule_bwd_dhu(
        q.npu(),
        k.npu(),
        w.npu(),
        d_o.npu(),
        dv.npu(),
        scale,
        chunk_size,
        g=g.npu(),
        gK=None,
        h0=None,
        dht=None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )


def test_chunk_gated_delta_rule_bwd_dhu_interface_exist():
    assert hasattr(torch.ops.ascend_ops, "chunk_gated_delta_rule_bwd_dhu")


FIX_TEST_CONFIGS = [
    (1, 2, 4, 128, 128, 128, 64, 0.088, torch.float16, torch.float16),
    (1, 2, 4, 128, 128, 128, 64, 0.088, torch.bfloat16, torch.bfloat16),
    (1, 2, 4, 128, 128, 128, 64, 0.088, torch.bfloat16, torch.float32),
    (1, 2, 4, 128, 128, 128, 64, 0.088, torch.float16, torch.float32),
    (1, 4, 4, 128, 128, 128, 64, 0.088, torch.bfloat16, torch.bfloat16),
]

VARIABLE_TEST_CONFIGS = [
    (1, 4, 8, 512, 128, 128, 64, 0.011, 4, torch.bfloat16, torch.bfloat16),
    (1, 4, 8, 512, 128, 128, 64, 0.011, 4, torch.bfloat16, torch.float32),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("B,Hk,Hv,T,K,V,chunk_size,scale,ktype,gtype", FIX_TEST_CONFIGS)
def test_chunk_gated_delta_rule_bwd_dhu_fix(B, Hk, Hv, T, K, V, chunk_size, scale, ktype, gtype):
    torch.manual_seed(0)
    scale = scale_for_compute_dtype(effective_scale(scale, K), ktype)

    q, k, w, d_o, dv, g = create_bwd_dhu_random_inputs(B, Hk, Hv, T, K, V, ktype, gtype)
    dh_golden, _, dv2_golden = chunk_gated_delta_rule_bwd_dhu_torch(
        q, k, w, d_o, dv, cu_seqlens=None, chunk_indices=None, g=g, scale=scale, chunk_size=chunk_size
    )

    dh_npu, _, dv2_npu = run_chunk_gated_delta_rule_bwd_dhu(q, k, w, d_o, dv, g, scale, chunk_size)

    rtol = 1e-2
    atol = 1e-2
    assert torch.allclose(dh_npu.cpu().float(), dh_golden.float(), rtol=rtol, atol=atol)
    assert torch.allclose(dv2_npu.cpu().float(), dv2_golden.float(), rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("B,Hk,Hv,T,K,V,chunk_size,scale,cu_seqlens_len,ktype,gtype", VARIABLE_TEST_CONFIGS)
def test_chunk_gated_delta_rule_bwd_dhu_variable(B, Hk, Hv, T, K, V, chunk_size, scale, cu_seqlens_len, ktype, gtype):
    torch.manual_seed(0)
    scale = scale_for_compute_dtype(effective_scale(scale, K), ktype)

    q, k, w, d_o, dv, g = create_bwd_dhu_random_inputs(B, Hk, Hv, T, K, V, ktype, gtype)
    cu_seqlens = generate_cu_seqlens(cu_seqlens_len, T)
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    dh_golden, _, dv2_golden = chunk_gated_delta_rule_bwd_dhu_torch(
        q, k, w, d_o, dv, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, g=g, scale=scale, chunk_size=chunk_size
    )

    dh_npu, _, dv2_npu = run_chunk_gated_delta_rule_bwd_dhu(
        q, k, w, d_o, dv, g, scale, chunk_size, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )

    rtol = 1e-2
    atol = 1e-2
    assert torch.allclose(dh_npu.cpu().float(), dh_golden.float(), rtol=rtol, atol=atol)
    assert torch.allclose(dv2_npu.cpu().float(), dv2_golden.float(), rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_chunk_gated_delta_rule_bwd_dhu_metadata_pair():
    torch.manual_seed(0)
    B, Hk, Hv, T, K, V = 1, 2, 4, 128, 128, 128
    chunk_size = 64
    scale = 0.088
    ktype = torch.bfloat16
    gtype = torch.bfloat16

    q, k, w, d_o, dv, g = create_bwd_dhu_random_inputs(B, Hk, Hv, T, K, V, ktype, gtype)
    cu_seqlens = generate_cu_seqlens(3, T)

    q_npu = q.npu()
    k_npu = k.npu()
    w_npu = w.npu()
    d_o_npu = d_o.npu()
    dv_npu = dv.npu()
    g_npu = g.npu()

    with pytest.raises(RuntimeError, match="cu_seqlens and chunk_indices"):
        torch.ops.ascend_ops.chunk_gated_delta_rule_bwd_dhu(
            q_npu, k_npu, w_npu, d_o_npu, dv_npu, scale, chunk_size,
            g=g_npu, gK=None, h0=None, dht=None,
            cu_seqlens=cu_seqlens, chunk_indices=None,
        )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_chunk_gated_delta_rule_bwd_dhu_gva_smoke():
    torch.manual_seed(0)
    B, Hk, Hv, T, K, V = 1, 2, 4, 256, 128, 128
    chunk_size = 64
    scale = scale_for_compute_dtype(effective_scale(0.088, K), torch.bfloat16)
    ktype = torch.bfloat16
    gtype = torch.bfloat16

    q, k, w, d_o, dv, g = create_bwd_dhu_random_inputs(B, Hk, Hv, T, K, V, ktype, gtype)
    dh_golden, _, dv2_golden = chunk_gated_delta_rule_bwd_dhu_torch(
        q, k, w, d_o, dv, g=g, scale=scale, chunk_size=chunk_size
    )
    dh_npu, _, dv2_npu = run_chunk_gated_delta_rule_bwd_dhu(q, k, w, d_o, dv, g, scale, chunk_size)

    assert dh_npu.shape == dh_golden.shape
    assert dv2_npu.shape == dv2_golden.shape
