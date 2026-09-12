#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Accuracy test for npu_recurrent_gated_delta_rule.

Compares NPU output against the canonical CPU reference from tests/atk.
"""

import os
import sys
from pathlib import Path

import ascend_ops
import pytest
import torch
import torch_npu

_ATK_DIR = (
    Path(__file__).resolve().parents[4]
    / "tests/atk/recurrent_gated_delta_rule"
)
sys.path.insert(0, str(_ATK_DIR))

from reference_recurrent_gated_delta_rule import (
    recurrent_gated_delta_rule_reference,
)


STATE_HEAD_PREFIX_PADDING = 16
STATE_LAYOUTS = ("contiguous", "noncontiguous")
API_MODES = ("functional", "mutable")


def make_inputs(
    bs,
    mtp,
    nk,
    nv,
    dk,
    dv,
    use_g=True,
    use_gk=False,
    use_accepted_tokens=False,
    seed=42,
):
    """Generate inputs on CPU, matching the kernel's expected shapes."""
    torch.manual_seed(seed)
    star_idx = 1
    star_idx_tensor = torch.tensor([star_idx], dtype=torch.int32)
    batch_tensor = torch.ones(bs, dtype=torch.int32) * mtp
    actual_seq_lengths = torch.cat([star_idx_tensor, batch_tensor])
    t = int(actual_seq_lengths.sum().item())

    state = torch.rand((t, nv, dv, dk), dtype=torch.float)
    query = torch.nn.functional.normalize(
        torch.rand((t, nk, dk), dtype=torch.bfloat16), p=2, dim=-1
    )
    key = torch.nn.functional.normalize(
        torch.rand((t, nk, dk), dtype=torch.bfloat16), p=2, dim=-1
    )
    value = torch.rand((t, nv, dv), dtype=torch.bfloat16)
    beta = torch.rand((t, nv), dtype=torch.bfloat16)
    scale = dk ** -0.5

    ssm_state_indices = torch.arange(t, dtype=torch.int32)

    g = None
    if use_g:
        g = -torch.rand((t, nv), dtype=torch.float32)

    gk = None
    if use_gk:
        gk = -torch.rand((t, nv, dk), dtype=torch.float32)

    num_accepted_tokens = None
    if use_accepted_tokens and mtp > 1:
        num_accepted_tokens = torch.randint(1, mtp + 1, (bs,), dtype=torch.int32)

    return {
        "query": query,
        "key": key,
        "value": value,
        "state": state,
        "beta": beta,
        "scale": scale,
        "actual_seq_lengths": actual_seq_lengths,
        "ssm_state_indices": ssm_state_indices,
        "num_accepted_tokens": num_accepted_tokens,
        "g": g,
        "gk": gk,
    }


def run_golden(inp):
    """Run CPU golden implementation."""
    return recurrent_gated_delta_rule_reference(
        query=inp["query"],
        key=inp["key"],
        value=inp["value"],
        state=inp["state"].clone(),
        beta=inp["beta"],
        scale=inp["scale"],
        actual_seq_lengths=inp["actual_seq_lengths"],
        ssm_state_indices=inp["ssm_state_indices"],
        num_accepted_tokens=inp["num_accepted_tokens"],
        g=inp["g"],
        gk=inp["gk"],
    )


def make_npu_state(cpu_state, device, state_layout):
    dense_state = cpu_state.to(device)
    if state_layout == "contiguous":
        return dense_state, tuple(dense_state.stride())

    block_num, nv, dv, dk = cpu_state.shape
    head_stride = dv * dk + STATE_HEAD_PREFIX_PADDING
    padded_strides = (nv * head_stride, head_stride, dk, 1)
    storage = torch.empty(
        (block_num, nv, head_stride), dtype=cpu_state.dtype, device=device
    )
    state = storage.as_strided(
        cpu_state.shape, padded_strides, STATE_HEAD_PREFIX_PADDING
    )
    state.copy_(dense_state)
    return state, padded_strides


def run_npu(inp, state_layout, api_mode):
    """Run NPU operator and return CPU tensors."""
    device = torch.device(os.getenv("FLA_NPU_TEST_DEVICE", "npu:0"))
    torch_npu.npu.set_device(device)
    q_npu = inp["query"].to(device)
    k_npu = inp["key"].to(device)
    v_npu = inp["value"].to(device)
    s_npu, expected_state_strides = make_npu_state(
        inp["state"], device, state_layout
    )
    b_npu = inp["beta"].to(device)
    asl_npu = inp["actual_seq_lengths"].to(device)
    ssi_npu = inp["ssm_state_indices"].to(device)

    g_npu = inp["g"].to(device) if inp["g"] is not None else None
    gk_npu = inp["gk"].to(device) if inp["gk"] is not None else None
    nat_npu = (
        inp["num_accepted_tokens"].to(device)
        if inp["num_accepted_tokens"] is not None
        else None
    )

    assert tuple(s_npu.stride()) == expected_state_strides
    if state_layout == "noncontiguous":
        assert not s_npu.is_contiguous()
        assert s_npu.storage_offset() == STATE_HEAD_PREFIX_PADDING

    # Finish asynchronous H2D/view initialization before isolating the custom
    # kernel launch in this accuracy test.
    torch_npu.npu.synchronize()

    kwargs = {
        "beta": b_npu,
        "scale": inp["scale"],
        "actual_seq_lengths": asl_npu,
        "ssm_state_indices": ssi_npu,
        "num_accepted_tokens": nat_npu,
        "g": g_npu,
        "gk": gk_npu,
    }
    if api_mode == "functional":
        result = torch.ops.ascend_ops.recurrent_gated_delta_rule_functional(
            q_npu, k_npu, v_npu, s_npu, **kwargs
        )
        torch_npu.npu.synchronize()
        attn_out = result[0].cpu()
        final_state = result[1].cpu()
        output_state_strides = tuple(result[1].stride())
        input_state_after = s_npu.cpu()
    else:
        result = torch.ops.ascend_ops.recurrent_gated_delta_rule(
            q_npu, k_npu, v_npu, s_npu, **kwargs
        )
        torch_npu.npu.synchronize()
        attn_out = result.cpu()
        final_state = s_npu.cpu()
        output_state_strides = tuple(s_npu.stride())
        input_state_after = None

    star_idx = int(inp["actual_seq_lengths"][0].item())
    attn_out[:star_idx] = 0
    return (
        attn_out,
        final_state,
        output_state_strides,
        expected_state_strides,
        input_state_after,
    )


def assert_compare_tensors_by_ratio(golden, actual, name, rtol=0.01, atol=0.004):
    assert golden.shape == actual.shape, (
        f"{name} shape mismatch: golden={golden.shape}, actual={actual.shape}"
    )
    golden_float = golden.float()
    actual_float = actual.float()
    close_mask = torch.isclose(actual_float, golden_float, rtol=rtol, atol=atol)
    failed_count = int((~close_mask).sum().item())
    assert failed_count == 0, (
        f"{name} comparison failed: failed={failed_count}/{golden.numel()}, "
        f"rtol={rtol}, atol={atol}"
    )


def test_recurrent_gated_delta_rule_interface_exist():
    assert hasattr(torch.ops.ascend_ops, "recurrent_gated_delta_rule")
    assert hasattr(torch.ops.ascend_ops, "recurrent_gated_delta_rule_functional")


TEST_CONFIGS = [
    pytest.param(
        2,
        2,
        4,
        8,
        128,
        128,
        True,
        False,
        False,
        42,
        0.01,
        0.004,
        id="basic_bs2_mtp2",
    ),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("state_layout", STATE_LAYOUTS)
@pytest.mark.parametrize("api_mode", API_MODES)
@pytest.mark.parametrize(
    "bs,mtp,nk,nv,dk,dv,use_g,use_gk,use_accepted_tokens,seed,rtol,atol",
    TEST_CONFIGS,
)
def test_recurrent_gated_delta_rule_accuracy(
    bs,
    mtp,
    nk,
    nv,
    dk,
    dv,
    use_g,
    use_gk,
    use_accepted_tokens,
    seed,
    rtol,
    atol,
    state_layout,
    api_mode,
):
    inp = make_inputs(
        bs,
        mtp,
        nk,
        nv,
        dk,
        dv,
        use_g=use_g,
        use_gk=use_gk,
        use_accepted_tokens=use_accepted_tokens,
        seed=seed,
    )

    golden_attn, golden_state = run_golden(inp)
    npu_attn, npu_state, output_strides, expected_strides, input_state_after = run_npu(
        inp, state_layout, api_mode
    )

    assert_compare_tensors_by_ratio(golden_attn, npu_attn, "attn_out", rtol=rtol, atol=atol)
    assert_compare_tensors_by_ratio(golden_state, npu_state, "final_state", rtol=rtol, atol=atol)
    assert output_strides == expected_strides
    if api_mode == "functional":
        assert torch.equal(input_state_after, inp["state"])
