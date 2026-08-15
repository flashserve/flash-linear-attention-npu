# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import math
import os

import torch
import torch_npu

from fla_npu.ops import ascendc


DEVICE_ID = int(os.environ.get("TEST_DEVICE_ID", 0))


def canonical_chunks(cu_seqlens: list[int] | None, chunk_size: int) -> list[int] | None:
    if cu_seqlens is None:
        return None
    result = []
    for sequence, (begin, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        for local_chunk in range(math.ceil((end - begin) / chunk_size)):
            result.extend((sequence, local_chunk))
    return result


def assert_bit_exact(name: str, expected: torch.Tensor, actual: torch.Tensor) -> None:
    if torch.equal(expected, actual):
        return
    max_abs = float((expected.float() - actual.float()).abs().max().cpu())
    raise AssertionError(f"{name} is not bit exact; max_abs={max_abs:.6e}")


def assert_all_finite(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor.float()).all()):
        nonfinite = int((~torch.isfinite(tensor.float())).sum())
        raise AssertionError(f"{name} contains {nonfinite} non-finite values")


def valid_a_chunks(
    tensor: torch.Tensor,
    cu_seqlens: list[int] | None,
    chunk_size: int,
) -> list[torch.Tensor]:
    sequences = [(batch, 0, tensor.shape[2]) for batch in range(tensor.shape[0])]
    if cu_seqlens is not None:
        sequences = [(0, begin, end) for begin, end in zip(cu_seqlens, cu_seqlens[1:])]
    chunks = []
    for batch, begin, end in sequences:
        for chunk_begin in range(begin, end, chunk_size):
            chunk_end = min(chunk_begin + chunk_size, end)
            chunk_len = chunk_end - chunk_begin
            chunks.append(tensor[batch, :, chunk_begin:chunk_end, :chunk_len])
    return chunks


def assert_valid_a_finite(
    name: str,
    tensor: torch.Tensor,
    cu_seqlens: list[int] | None,
    chunk_size: int,
) -> None:
    for chunk in valid_a_chunks(tensor, cu_seqlens, chunk_size):
        assert_all_finite(name, chunk)


def assert_valid_a_bit_exact(
    name: str,
    expected: torch.Tensor,
    actual: torch.Tensor,
    cu_seqlens: list[int] | None,
    chunk_size: int,
) -> None:
    expected_chunks = valid_a_chunks(expected, cu_seqlens, chunk_size)
    actual_chunks = valid_a_chunks(actual, cu_seqlens, chunk_size)
    for expected_chunk, actual_chunk in zip(expected_chunks, actual_chunks):
        assert_bit_exact(name, expected_chunk, actual_chunk)


def run_case(case: dict) -> None:
    torch.manual_seed(20260724)
    batch = case["batch"]
    k_heads = case["k_heads"]
    v_heads = case["v_heads"]
    tokens = case["tokens"]
    chunk_size = case["chunk_size"]
    dtype = case["dtype"]
    cu_seqlens = case["cu_seqlens"]
    chunk_indices = canonical_chunks(cu_seqlens, chunk_size)
    sequence_count = len(cu_seqlens) - 1 if cu_seqlens is not None else batch

    q = (torch.randn(batch, k_heads, tokens, 128, dtype=dtype) * 0.05).npu()
    k = (torch.randn(batch, k_heads, tokens, 128, dtype=dtype) * 0.05).npu()
    if k_heads != v_heads:
        repeat = v_heads // k_heads
        q = q.repeat_interleave(repeat, dim=1).contiguous()
        k = k.repeat_interleave(repeat, dim=1).contiguous()
    v = (torch.randn(batch, v_heads, tokens, 128, dtype=dtype) * 0.05).npu()
    beta_token_first = torch.sigmoid(torch.randn(batch, tokens, v_heads, dtype=dtype)).npu()
    g_token_first = (-torch.rand(batch, tokens, v_heads, dtype=torch.float32) * 0.1).npu()
    initial_state = None
    if case["initial_state"]:
        initial_state = (
            torch.randn(sequence_count, v_heads, 128, 128, dtype=dtype) * 0.01
        ).npu()
    scale = 128**-0.5

    cu_tensor = None if cu_seqlens is None else torch.tensor(cu_seqlens, device="npu", dtype=torch.int64)
    chunk_tensor = None
    if chunk_indices is not None:
        chunk_tensor = torch.tensor(chunk_indices, device="npu", dtype=torch.int64).view(-1, 2)
    beta = beta_token_first.transpose(1, 2).contiguous().float()
    g_input = g_token_first.transpose(1, 2).contiguous()
    g = ascendc.chunk_local_cumsum(
        g_input,
        chunk_size=chunk_size,
        cu_seqlens=cu_tensor,
        chunk_indices_out=chunk_tensor,
        head_first=True,
    )
    a_raw = ascendc.chunk_scaled_dot_kkt(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    if cu_seqlens is None:
        a = ascendc.solve_tri(a_raw.to(dtype), layout="bhtd")
    else:
        a_token_first = a_raw.transpose(1, 2).contiguous().squeeze(0)
        a_token_first = ascendc.solve_tri(
            a_token_first.to(dtype),
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            layout="tnd",
        )
        a = a_token_first.unsqueeze(0).transpose(1, 2).contiguous()

    w, u = ascendc.recompute_w_u_fwd(
        k,
        v,
        beta,
        a,
        chunk_size,
        g=g,
        gk=None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    h, v_new, legacy_final_state = ascendc.chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g=g,
        gk=None,
        initial_state=initial_state,
        output_final_state=case["output_final_state"],
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=False,
        transpose_state_layout=False,
    )
    legacy_o = ascendc.chunk_fwd_o(
        q,
        k,
        v_new,
        h,
        scale,
        g=g,
        g_gamma=None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        transpose_state_layout=False,
    )

    composite_o, composite_final_state, composite_g, composite_a = ascendc.gdn_core_fwd(
        q,
        k,
        v,
        g_token_first,
        beta_token_first,
        initial_state=initial_state,
        output_final_state=case["output_final_state"],
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
    )
    torch.npu.synchronize()

    tensors = {
        "legacy output": legacy_o,
        "composite output": composite_o,
        "legacy g": g,
        "composite g": composite_g,
    }
    for tensor_name, tensor in tensors.items():
        assert_all_finite(f"{case['name']} {tensor_name}", tensor)
    assert_valid_a_finite(f"{case['name']} legacy A", a, cu_seqlens, chunk_size)
    assert_valid_a_finite(f"{case['name']} composite A", composite_a, cu_seqlens, chunk_size)
    assert_bit_exact(f"{case['name']} output", legacy_o.cpu(), composite_o.cpu())
    assert_bit_exact(
        f"{case['name']} g",
        g.transpose(1, 2).contiguous().cpu(),
        composite_g.cpu(),
    )
    assert_valid_a_bit_exact(
        f"{case['name']} A",
        a.cpu(),
        composite_a.cpu(),
        cu_seqlens,
        chunk_size,
    )
    if case["output_final_state"]:
        assert legacy_final_state is not None and composite_final_state is not None
        assert_all_finite(f"{case['name']} legacy final_state", legacy_final_state)
        assert_all_finite(f"{case['name']} composite final_state", composite_final_state)
        assert_bit_exact(
            f"{case['name']} final_state",
            legacy_final_state.cpu(),
            composite_final_state.cpu(),
        )
    elif composite_final_state is not None:
        raise AssertionError(f"{case['name']} returned an unexpected final state")
    print(f"{case['name']}: PASS")


def main() -> None:
    torch.npu.set_device(DEVICE_ID)
    torch.npu.set_compile_mode(jit_compile=False)
    cases = (
        {
            "name": "dense_mha_bf16",
            "batch": 2,
            "k_heads": 2,
            "v_heads": 2,
            "tokens": 128,
            "dtype": torch.bfloat16,
            "chunk_size": 64,
            "cu_seqlens": None,
            "initial_state": False,
            "output_final_state": False,
        },
        {
            "name": "dense_gva_fp16_tail",
            "batch": 1,
            "k_heads": 2,
            "v_heads": 4,
            "tokens": 129,
            "dtype": torch.float16,
            "chunk_size": 64,
            "cu_seqlens": None,
            "initial_state": False,
            "output_final_state": False,
        },
        {
            "name": "varlen_gva_bf16_tail",
            "batch": 1,
            "k_heads": 2,
            "v_heads": 4,
            "tokens": 130,
            "dtype": torch.bfloat16,
            "chunk_size": 64,
            "cu_seqlens": [0, 65, 130],
            "initial_state": False,
            "output_final_state": True,
        },
        {
            "name": "dense_mha_bf16_initial_state",
            "batch": 2,
            "k_heads": 2,
            "v_heads": 2,
            "tokens": 128,
            "dtype": torch.bfloat16,
            "chunk_size": 64,
            "cu_seqlens": None,
            "initial_state": True,
            "output_final_state": True,
        },
        {
            "name": "dense_mha_bf16_chunk128",
            "batch": 1,
            "k_heads": 2,
            "v_heads": 2,
            "tokens": 128,
            "dtype": torch.bfloat16,
            "chunk_size": 128,
            "cu_seqlens": None,
            "initial_state": False,
            "output_final_state": False,
        },
        {
            "name": "varlen_gva_bf16_chunk128",
            "batch": 1,
            "k_heads": 2,
            "v_heads": 4,
            "tokens": 130,
            "dtype": torch.bfloat16,
            "chunk_size": 128,
            "cu_seqlens": [0, 65, 130],
            "initial_state": False,
            "output_final_state": True,
        },
    )
    for case in cases:
        run_case(case)
    print(f"GDN core composite A/B: {len(cases)}/{len(cases)} PASS")


if __name__ == "__main__":
    main()
