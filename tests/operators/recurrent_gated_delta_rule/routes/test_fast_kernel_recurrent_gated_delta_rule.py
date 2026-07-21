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

Compares NPU output against CPU golden reference from PTA.
"""

import os
import sys
from pathlib import Path

import pytest
from tests.operators._shared.route_requirements import require_fast_kernel_route

require_fast_kernel_route()

import ascend_ops
import torch
import torch_npu
from tests.operators._shared.legacy_cases import legacy_param_values

_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))

from tests.operators.recurrent_gated_delta_rule.common.golden import (
    recurrent_gated_delta_rule_golden,
)
from tests.operators.recurrent_gated_delta_rule.common.legacy_compare import (
    compare_tensors_by_ratio,
)


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
    return recurrent_gated_delta_rule_golden(
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


def run_npu(inp):
    """Run NPU operator and return CPU tensors."""
    device = torch.device(f"npu:{int(os.environ.get('TEST_DEVICE_ID', 0))}")
    torch_npu.npu.set_device(device)
    q_npu = inp["query"].npu()
    k_npu = inp["key"].npu()
    v_npu = inp["value"].npu()
    s_npu = inp["state"].clone().npu()
    b_npu = inp["beta"].npu()
    asl_npu = inp["actual_seq_lengths"].npu()
    ssi_npu = inp["ssm_state_indices"].npu()

    g_npu = inp["g"].npu() if inp["g"] is not None else None
    gk_npu = inp["gk"].npu() if inp["gk"] is not None else None
    nat_npu = (
        inp["num_accepted_tokens"].npu()
        if inp["num_accepted_tokens"] is not None
        else None
    )

    result = torch.ops.ascend_ops.recurrent_gated_delta_rule_functional(
        q_npu,
        k_npu,
        v_npu,
        s_npu,
        beta=b_npu,
        scale=inp["scale"],
        actual_seq_lengths=asl_npu,
        ssm_state_indices=ssi_npu,
        num_accepted_tokens=nat_npu,
        g=g_npu,
        gk=gk_npu,
    )
    torch_npu.npu.synchronize()

    if isinstance(result, (tuple, list)):
        attn_out = result[0].cpu()
        final_state = result[1].cpu()
    else:
        attn_out = result.cpu()
        final_state = s_npu.cpu()

    star_idx = int(inp["actual_seq_lengths"][0].item())
    attn_out[:star_idx] = 0
    return attn_out, final_state


def assert_compare_tensors_by_ratio(golden, actual, name, rtol=0.01, atol=0.004):
    passed = compare_tensors_by_ratio(golden, actual, name, rtol=rtol, atol=atol)
    assert passed, f"{name} comparison failed (rtol={rtol}, atol={atol})"


def test_recurrent_gated_delta_rule_interface_exist():
    assert hasattr(torch.ops.ascend_ops, "recurrent_gated_delta_rule")
    assert hasattr(torch.ops.ascend_ops, "recurrent_gated_delta_rule_functional")


TEST_CONFIGS = [
    pytest.param(*values, id=case_id)
    for case_id, values in legacy_param_values(
        "recurrent_gated_delta_rule",
        "direct_accuracy",
        param_names=['bs', 'mtp', 'nk', 'nv', 'dk', 'dv', 'use_g', 'use_gk', 'use_accepted_tokens', 'seed', 'rtol', 'atol'],
        dtype_module=torch,
    )
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "bs,mtp,nk,nv,dk,dv,use_g,use_gk,use_accepted_tokens,seed,rtol,atol",
    TEST_CONFIGS,
)
def test_recurrent_gated_delta_rule_functional_accuracy(
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
    npu_attn, npu_state = run_npu(inp)

    assert_compare_tensors_by_ratio(golden_attn, npu_attn, "attn_out", rtol=rtol, atol=atol)
    assert_compare_tensors_by_ratio(golden_state, npu_state, "final_state", rtol=rtol, atol=atol)
