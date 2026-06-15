#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import math
import random
from pathlib import Path
from typing import Optional

import pytest
import torch
import torch_npu
import ascend_ops

DATA_DIR = Path(__file__).resolve().parent / "data"


# --------------------------------------------------------------------------------------
# helpers (ported from fla/ops/.../chunk_gated_delta_rule_fwd_h/tests/pta/test_fwd_h.py)
# --------------------------------------------------------------------------------------
def cdiv(a, b):
    return (a + b - 1) // b


def prepare_lens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    indices = torch.cat([torch.arange(n) for n in cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def gen_seqlen(seqlen: int, batch: int) -> torch.Tensor:
    cu_seqlens = [0]
    avg_len = seqlen // batch
    for _ in range(batch - 1):
        diff = random.randint(max(1, avg_len // 2), max(1, avg_len * 3 // 2))
        cu_seqlens.append(min(cu_seqlens[-1] + diff, seqlen - (batch - len(cu_seqlens))))
    cu_seqlens.append(seqlen)
    return torch.tensor(cu_seqlens, dtype=torch.int64)


def gen_decay_data(shape_batch, v_num_head, seqlen, chunk_size, token_batch, cu_seqlens):
    """Cumulative-sum decay within each chunk, kept negative so exp() stays stable."""
    base = torch.randint(-15, -5, [v_num_head])
    bias = torch.empty([shape_batch, v_num_head, seqlen]).uniform_(-2, 0)
    g = base[None, :, None] + bias
    for sb in range(shape_batch):
        for vh in range(v_num_head):
            for tb in range(token_batch):
                start = int(cu_seqlens[tb]) if cu_seqlens is not None else 0
                end = int(cu_seqlens[tb + 1]) if cu_seqlens is not None else seqlen
                chunks = math.ceil((end - start) / chunk_size)
                for c in range(chunks):
                    cs = start + chunk_size * c
                    ce = min(cs + chunk_size, end)
                    g[sb, vh, cs:ce] = g[sb, vh, cs:ce].cumsum(0)
    return g


def forward_h_trans_cpu(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    dtype_ = k.dtype
    state_type_ = initial_state.dtype if initial_state is not None else torch.float32

    k = k.to(torch.float32)
    w = w.to(torch.float32)
    u = u.to(torch.float32)
    g = g.to(torch.float32)

    B, HK, T, K = k.shape
    HV, V = u.shape[1], u.shape[3]
    BT = chunk_size

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    if cu_seqlens is None:
        N, NT = B, (T + BT - 1) // BT
        chunk_offsets = None
    else:
        N, NT = len(cu_seqlens) - 1, len(chunk_indices)
        chunk_offsets = torch.cat(
            [cu_seqlens.new_tensor([0]), cdiv(prepare_lens(cu_seqlens), BT)]
        ).cumsum(-1)
    if initial_state is not None:
        initial_state = initial_state.reshape([N, HV, K, V]).contiguous().to(torch.float32)

    S = torch.zeros((B, HV, NT, K, V), dtype=torch.float32)
    v_new_output = torch.zeros((B, HV, T, V), dtype=torch.float32)
    final_state = torch.zeros((N, HV, K, V), dtype=torch.float32)

    head_ratio = HV // HK
    for n in range(N):
        if cu_seqlens is None:
            bos, eos, boh = 0, T, 0
            NT_inner = NT
        else:
            bos, eos = int(cu_seqlens[n]), int(cu_seqlens[n + 1])
            NT_inner = (eos - bos + BT - 1) // BT
            boh = int(chunk_offsets[n])
        bidx = 0 if cu_seqlens is not None else n
        for h in range(HV):
            for i in range(NT_inner):
                actual_len = min(bos + (i + 1) * BT, eos) - (bos + i * BT)
                k_sel = torch.zeros((BT, K), dtype=torch.float32)
                w_sel = torch.zeros((BT, w.shape[-1]), dtype=torch.float32)
                u_sel = torch.zeros((BT, V), dtype=torch.float32)
                g_sel = torch.zeros((BT,), dtype=torch.float32)
                k_sel[:actual_len] = k[bidx, h // head_ratio, bos + i * BT: bos + i * BT + actual_len, :]
                w_sel[:actual_len] = w[bidx, h, bos + i * BT: bos + i * BT + actual_len, :]
                u_sel[:actual_len] = u[bidx, h, bos + i * BT: bos + i * BT + actual_len, :]
                g_sel[:actual_len] = g[bidx, h, bos + i * BT: bos + i * BT + actual_len]
                if initial_state is not None and i == 0:
                    S[bidx, h, boh + i] = initial_state[n, h]
                v_new = u_sel - w_sel @ S[bidx, h, boh + i]
                decay = (g_sel[actual_len - 1, None, None]).exp()
                new_state = S[bidx, h, boh + i] * decay + k_sel.transpose(-1, -2) @ (
                    v_new * (g_sel[actual_len - 1, None] - g_sel).exp()[..., None]
                )
                if i != NT_inner - 1:
                    S[bidx, h, boh + i + 1] = new_state
                else:
                    final_state[n, h] = new_state
                v_new_output[bidx, h, bos + i * BT: bos + i * BT + actual_len, :] = v_new[:actual_len, :]

    S = S.to(dtype_)
    v_new_output = v_new_output.to(dtype_)
    final_state = final_state.to(state_type_)
    return S, v_new_output, final_state


def assert_close(real, ref, diff_thd, pct_thd=0.05, max_diff_hd=0.1, name=""):
    """Mirror the repo's data_compare semantics: allow at most pct_thd fraction of elements to
    exceed the per-element relative threshold, and no single element may exceed max_diff_hd."""
    real = real.detach().cpu().to(torch.float32).flatten()
    ref = ref.detach().cpu().to(torch.float32).flatten()
    abs_diff = (real - ref).abs()
    denom = torch.maximum(real.abs(), ref.abs()) + 1e-9
    rel = torch.where(abs_diff < diff_thd, abs_diff, abs_diff / denom)
    fail_ratio = float((rel > diff_thd).float().mean().item())
    max_err = float(rel.max().item())
    assert fail_ratio <= pct_thd, \
        f"{name} mismatch: fail ratio {fail_ratio:.4f} > {pct_thd} (diff_thd={diff_thd})"
    assert max_err < max_diff_hd, \
        f"{name} mismatch: max relative error {max_err:.4f} >= {max_diff_hd}"


# --------------------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------------------
def test_interface_exist():
    print(torch.ops.ascend_ops.chunk_gated_delta_rule_fwd_h)
    assert hasattr(torch.ops.ascend_ops, "chunk_gated_delta_rule_fwd_h"), \
        "operator chunk_gated_delta_rule_fwd_h is not registered under torch.ops.ascend_ops"


# --------------------------------------------------------------------------------------
# model-distributed fixtures (produced by dump_model_data.py)
#
# fwd_h is a chunk recurrence: random k/w/u/g make the per-chunk states blow up, so the CPU
# golden loses all significant digits and any comparison becomes meaningless. Instead we drive
# the test with inputs sampled from the real GDN pipeline (l2-normalised key, cumsum gate, WY
# representation), where the recurrence stays bounded and the fp32 reference is trustworthy.
# Regenerate with: ASCEND_RT_VISIBLE_DEVICES=4 python dump_model_data.py
# --------------------------------------------------------------------------------------
FIXTURES = sorted(DATA_DIR.glob("*.pt")) if DATA_DIR.exists() else []


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.skipif(len(FIXTURES) == 0, reason="no model-dump fixtures found, run dump_model_data.py first")
@pytest.mark.parametrize("fixture_path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_model_dump(fixture_path):
    payload = torch.load(fixture_path, map_location="cpu")
    in_dtype = getattr(torch, payload["dtype"])
    chunk_size = int(payload["chunk_size"])
    cu_list = payload["cu_seqlens"]
    chunk_list = payload["chunk_indices"]

    k = payload["k"]
    w = payload["w"]
    u = payload["u"]
    g = payload["g"].to(torch.float32)
    initial_state = payload["initial_state"]
    out_final = True

    cu_tensor = torch.tensor(cu_list, dtype=torch.int64) if cu_list is not None else None

    h_ref, v_ref, fs_ref = forward_h_trans_cpu(
        k, w, u, g, initial_state, out_final, chunk_size, cu_seqlens=cu_tensor)

    h, v_new, final_state = torch.ops.ascend_ops.chunk_gated_delta_rule_fwd_h(
        k.npu(), w.npu(), u.npu(), g.npu(),
        initial_state=initial_state.npu() if initial_state is not None else None,
        output_final_state=out_final, chunk_size=chunk_size,
        cu_seqlens=cu_list, chunk_indices=chunk_list)

    diff_thd = 1e-2 if in_dtype == torch.bfloat16 else 4e-3
    assert_close(h, h_ref, diff_thd, name="h")
    assert_close(v_new, v_ref, diff_thd, name="v_new")
    assert_close(final_state, fs_ref, diff_thd, name="final_state")

