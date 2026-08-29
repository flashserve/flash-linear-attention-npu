# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the BSD 3-Clause License (the "License").

"""Direct C ABI tests for aclnnChunkFwdH.

These tests intentionally bypass ``fla_npu.ops.ascendc.chunk_fwd_h``.  They
construct aclTensor descriptors and call both aclnn stages through ctypes so
that symbol resolution, the public aclnn ABI and exact host-side status codes
are covered independently from the Python argument validation layer.
"""

from __future__ import annotations

import ctypes
import json
import os
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

import fla_npu.ops.ascendc._aclnn_ctypes as aclnn_ctypes
import fla_npu.ops.ascendc._runtime as aclnn_runtime


ACLNN_ERR_PARAM_NULLPTR = 161001
ACLNN_ERR_PARAM_INVALID = 161002
PRIVATE_NPU_FORMAT = 29
CASE_FILE = Path(__file__).resolve().parents[3] / "op_cases" / "chunk_fwd_h.json"
CASE_DATA = json.loads(CASE_FILE.read_text(encoding="utf-8"))
NEGATIVE_CASES = CASE_DATA["negative_cases"]
HOST_NEGATIVE_CASES = CASE_DATA["host_negative_cases"]
CALL_PATH_CASES = {case["id"]: case for case in CASE_DATA["call_path_cases"]}


def _npu_is_available() -> bool:
    try:
        import torch_npu  # noqa: F401

        return bool(torch.npu.is_available())
    except (ImportError, AttributeError, RuntimeError):
        return False


pytestmark = pytest.mark.skipif(not _npu_is_available(), reason="NPU device not found")


def _device() -> torch.device:
    device = torch.device(os.getenv("FLA_NPU_TEST_DEVICE", "npu:0"))
    torch.npu.set_device(device)
    return device


def _nd_tensor(ctx, tensor, name: str, *, acl_format: int = aclnn_runtime.ACL_FORMAT_ND):
    if tensor is None:
        return ctx.tensor(None, name)
    storage_shape = (
        tuple(int(dim) for dim in tensor.shape)
        if tensor.is_contiguous() and int(tensor.storage_offset()) == 0
        else None
    )
    return ctx.tensor(
        tensor,
        name,
        acl_format_override=acl_format,
        storage_shape_override=storage_shape,
    )


@contextmanager
def _direct_call_args(
    *,
    private_k_format: bool = False,
    mutation: str | None = None,
    output_final_state: bool = True,
    final_state_dtype: torch.dtype = torch.float32,
    k_value: float = 0.0,
    u_value: float = 0.0,
):
    device = _device()
    runtime = aclnn_runtime.runtime()
    ctx = aclnn_runtime._CallContext(runtime, device)

    batch, k_heads, v_heads, tokens, dim = 1, 1, 1, 64, 128
    input_dtype = torch.bfloat16
    use_g = True
    use_gk = False
    gk_heads = v_heads
    chunk_size = 64
    save_new_value = True
    state_v_first = False
    initial_shape = None
    initial_dtype = torch.bfloat16
    cu_values = None
    chunk_indices_values = None
    k_rank3 = False
    w_tokens = tokens
    g_tokens = tokens
    gate_dtype = torch.float32
    w_dtype = None
    h_shape = None
    v_new_shape = None
    final_shape = None
    h_dtype = None
    final_present = output_final_state
    null_tensor = None
    non_contiguous_output = None

    if mutation == "both_gate_inputs":
        use_gk = True
    elif mutation == "unsupported_chunk_size":
        chunk_size = 32
    elif mutation == "invalid_g_head_ratio":
        k_heads, v_heads = 2, 3
    elif mutation == "invalid_gk_shape":
        k_heads, v_heads = 2, 2
        use_g = False
        use_gk = True
        gk_heads = 1
    elif mutation == "varlen_batch_not_one":
        batch = 2
        cu_values = (0, tokens)
    elif mutation == "invalid_state_v_first_shape":
        state_v_first = True
        initial_shape = (1, 1, 127, 128)
    elif mutation == "missing_gate_inputs":
        use_g = False
    elif mutation == "unsupported_input_dtype":
        input_dtype = torch.float32
    elif mutation == "save_new_value_false":
        save_new_value = False
    elif mutation == "chunk_indices_without_cu":
        chunk_indices_values = (0, 0)
    elif mutation == "invalid_input_rank":
        k_rank3 = True
    elif mutation == "non_positive_dimension":
        batch = 0
    elif mutation == "mismatched_w_u_shape":
        w_tokens = tokens + 1
    elif mutation == "unsupported_kv_dimension":
        dim = 64
    elif mutation == "invalid_g_shape":
        g_tokens = tokens + 1
    elif mutation == "unsupported_gate_dtype":
        gate_dtype = torch.float16
    elif mutation == "mismatched_input_dtype":
        w_dtype = torch.float32
    elif mutation == "invalid_cu_size":
        cu_values = (0,)
    elif mutation == "invalid_cu_start":
        cu_values = (1, tokens)
    elif mutation == "invalid_cu_end":
        cu_values = (0, tokens + 1)
    elif mutation == "non_increasing_cu":
        cu_values = (0, tokens, tokens)
    elif mutation == "invalid_chunk_indices_length":
        cu_values = (0, tokens)
        chunk_indices_values = (0,)
    elif mutation == "invalid_chunk_indices_order":
        tokens = 65
        w_tokens = tokens
        g_tokens = tokens
        cu_values = (0, tokens)
        chunk_indices_values = (0, 1, 0, 0)
    elif mutation == "unsupported_state_dtype":
        initial_shape = (1, 1, dim, dim)
        initial_dtype = torch.float16
    elif mutation in {"null_k", "null_w", "null_u", "null_h", "null_v_new"}:
        null_tensor = mutation[len("null_"):]
    elif mutation == "missing_final_output":
        final_present = False
    elif mutation == "unexpected_final_output":
        output_final_state = False
        final_present = True
    elif mutation in {"non_contiguous_h", "non_contiguous_v_new", "non_contiguous_final"}:
        non_contiguous_output = mutation[len("non_contiguous_"):]
    elif mutation == "invalid_h_shape":
        h_shape = (batch, v_heads, 2, dim, dim)
    elif mutation == "invalid_v_new_shape":
        v_new_shape = (batch, v_heads, tokens + 1, dim)
    elif mutation == "invalid_final_shape":
        final_shape = (batch, v_heads, dim, dim - 1)
    elif mutation == "invalid_output_dtype":
        h_dtype = torch.float32
    elif mutation == "invalid_final_dtype":
        final_state_dtype = torch.float16
    elif mutation == "mismatched_final_dtype":
        initial_shape = (1, 1, dim, dim)
        initial_dtype = torch.bfloat16
        final_state_dtype = torch.float32
    elif mutation is not None:
        raise AssertionError(f"unknown negative mutation: {mutation}")

    k_shape = (batch, k_heads, dim) if k_rank3 else (batch, k_heads, tokens, dim)
    k = torch.full(
        k_shape, k_value, dtype=input_dtype, device=device
    )
    w = torch.zeros(
        (batch, v_heads, w_tokens, dim),
        dtype=w_dtype if w_dtype is not None else input_dtype,
        device=device,
    )
    u = torch.full(
        (batch, v_heads, tokens, dim), u_value, dtype=input_dtype, device=device
    )
    g = (
        torch.zeros((batch, v_heads, g_tokens), dtype=gate_dtype, device=device)
        if use_g
        else None
    )
    gk = (
        torch.zeros((batch, gk_heads, tokens, dim), dtype=gate_dtype, device=device)
        if use_gk
        else None
    )
    initial_state = (
        torch.zeros(initial_shape, dtype=initial_dtype, device=device)
        if initial_shape is not None
        else None
    )
    h = torch.full(
        h_shape if h_shape is not None else (batch, v_heads, 1, dim, dim),
        3,
        dtype=h_dtype if h_dtype is not None else input_dtype,
        device=device,
    )
    v_new = torch.full(
        v_new_shape if v_new_shape is not None else tuple(u.shape),
        5,
        dtype=input_dtype,
        device=device,
    )
    final_state = (
        torch.full(
            final_shape if final_shape is not None else (batch, v_heads, dim, dim),
            7,
            dtype=final_state_dtype,
            device=device,
        )
        if final_present
        else None
    )

    def make_non_contiguous(tensor):
        padded = torch.empty(
            (*tensor.shape[:-1], tensor.shape[-1] * 2),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        padded[..., ::2].copy_(tensor)
        return padded[..., ::2]

    if non_contiguous_output == "h":
        h = make_non_contiguous(h)
    elif non_contiguous_output == "v_new":
        v_new = make_non_contiguous(v_new)
    elif non_contiguous_output == "final":
        final_state = make_non_contiguous(final_state)

    k_format = PRIVATE_NPU_FORMAT if private_k_format else aclnn_runtime.ACL_FORMAT_ND
    args = [
        _nd_tensor(ctx, None if null_tensor == "k" else k, "k", acl_format=k_format),
        _nd_tensor(ctx, None if null_tensor == "w" else w, "w"),
        _nd_tensor(ctx, None if null_tensor == "u" else u, "u"),
        _nd_tensor(ctx, g, "g"),
        _nd_tensor(ctx, gk, "gk"),
        _nd_tensor(ctx, initial_state, "initial_state"),
        ctypes.c_bool(output_final_state),
        ctypes.c_int64(chunk_size),
        ctypes.c_bool(save_new_value),
        ctx.int_array(cu_values),
        ctx.int_array(chunk_indices_values),
        ctypes.c_bool(False),
        ctypes.c_bool(state_v_first),
        _nd_tensor(ctx, None if null_tensor == "h" else h, "h"),
        _nd_tensor(ctx, None if null_tensor == "v_new" else v_new, "v_new"),
        _nd_tensor(ctx, final_state, "final_state"),
    ]
    try:
        yield runtime, device, args, (h, v_new, final_state)
    finally:
        ctx.destroy()


def _get_workspace_symbol(runtime):
    symbol = runtime.symbol("aclnnChunkFwdHGetWorkspaceSize")
    symbol.argtypes = aclnn_ctypes._GET_WORKSPACE_ARGTYPES["aclnnChunkFwdH"]
    symbol.restype = ctypes.c_int
    return symbol


def test_aclnn_chunk_fwd_h_direct_positive_zero_input():
    with _direct_call_args() as (runtime, device, args, outputs):
        assert runtime.symbol("aclnnChunkFwdH") is not None
        workspace = runtime.call(
            "aclnnChunkFwdH",
            args,
            device,
            get_workspace_argtypes=aclnn_ctypes._GET_WORKSPACE_ARGTYPES[
                "aclnnChunkFwdH"
            ],
        )
        torch.npu.synchronize()
        # Keep the workspace alive through synchronization even though the
        # runtime allocator is stream aware.
        assert workspace is not None
        for output in outputs:
            assert torch.count_nonzero(output).item() == 0


def test_aclnn_chunk_fwd_h_direct_terminal_without_final_state():
    with _direct_call_args(output_final_state=False) as (
        runtime,
        device,
        args,
        outputs,
    ):
        workspace = runtime.call(
            "aclnnChunkFwdH",
            args,
            device,
            get_workspace_argtypes=aclnn_ctypes._GET_WORKSPACE_ARGTYPES[
                "aclnnChunkFwdH"
            ],
        )
        torch.npu.synchronize()
        assert workspace is not None
        assert outputs[2] is None
        assert torch.count_nonzero(outputs[0]).item() == 0
        assert torch.count_nonzero(outputs[1]).item() == 0


def test_aclnn_chunk_fwd_h_zero_initial_bf16_final_state():
    case = CALL_PATH_CASES["aclnn_zero_initial_bf16_final"]
    with _direct_call_args(
        final_state_dtype=torch.bfloat16,
        k_value=case["k_value"],
        u_value=case["u_value"],
    ) as (runtime, device, args, outputs):
        workspace = runtime.call(
            "aclnnChunkFwdH",
            args,
            device,
            get_workspace_argtypes=aclnn_ctypes._GET_WORKSPACE_ARGTYPES[
                "aclnnChunkFwdH"
            ],
        )
        torch.npu.synchronize()
        assert workspace is not None

        h, v_new, final_state = outputs
        assert final_state is not None
        assert final_state.dtype == torch.bfloat16
        assert torch.count_nonzero(h).item() == 0
        torch.testing.assert_close(
            v_new,
            torch.full_like(v_new, case["u_value"]),
            rtol=0,
            atol=0,
        )

        k_bf16 = torch.tensor(case["k_value"], dtype=torch.bfloat16).float()
        u_bf16 = torch.tensor(case["u_value"], dtype=torch.bfloat16).float()
        expected_value = (case["seqlen"] * k_bf16 * u_bf16).to(torch.bfloat16)
        torch.testing.assert_close(
            final_state,
            torch.full_like(final_state, expected_value.item()),
            rtol=0.02,
            atol=0.002,
        )


def test_aclnn_chunk_fwd_h_null_workspace_size_returns_nullptr():
    with _direct_call_args() as (runtime, _device_value, args, _outputs):
        get_workspace = _get_workspace_symbol(runtime)
        executor = ctypes.c_void_p()
        status = get_workspace(*args, None, ctypes.byref(executor))

    assert status == ACLNN_ERR_PARAM_NULLPTR
    assert not executor.value


def test_aclnn_chunk_fwd_h_null_executor_returns_nullptr():
    with _direct_call_args() as (runtime, _device_value, args, _outputs):
        get_workspace = _get_workspace_symbol(runtime)
        workspace_size = ctypes.c_uint64(0xA5A5A5A5)
        status = get_workspace(*args, ctypes.byref(workspace_size), None)

    assert status == ACLNN_ERR_PARAM_NULLPTR
    assert workspace_size.value == 0xA5A5A5A5


def test_aclnn_chunk_fwd_h_private_format_returns_invalid():
    with _direct_call_args(private_k_format=True) as (
        runtime,
        _device_value,
        args,
        _outputs,
    ):
        get_workspace = _get_workspace_symbol(runtime)
        workspace_size = ctypes.c_uint64(0)
        executor = ctypes.c_void_p()
        status = get_workspace(
            *args, ctypes.byref(workspace_size), ctypes.byref(executor)
        )

    assert status == ACLNN_ERR_PARAM_INVALID
    assert not executor.value


@pytest.mark.parametrize("case", NEGATIVE_CASES, ids=lambda case: case["id"])
def test_aclnn_chunk_fwd_h_host_validation_returns_invalid(case):
    assert case["expected_return_code"] == "ACLNN_ERR_PARAM_INVALID"
    with _direct_call_args(mutation=case["mutation"]) as (
        runtime,
        _device_value,
        args,
        _outputs,
    ):
        get_workspace = _get_workspace_symbol(runtime)
        workspace_size = ctypes.c_uint64(0)
        executor = ctypes.c_void_p()
        status = get_workspace(
            *args, ctypes.byref(workspace_size), ctypes.byref(executor)
        )

    assert status == ACLNN_ERR_PARAM_INVALID
    assert not executor.value


@pytest.mark.parametrize("case", HOST_NEGATIVE_CASES, ids=lambda case: case["id"])
def test_aclnn_chunk_fwd_h_host_only_validation(case):
    expected_status = {
        "ACLNN_ERR_PARAM_NULLPTR": ACLNN_ERR_PARAM_NULLPTR,
        "ACLNN_ERR_PARAM_INVALID": ACLNN_ERR_PARAM_INVALID,
    }[case["expected_return_code"]]
    with _direct_call_args(mutation=case["mutation"]) as (
        runtime,
        _device_value,
        args,
        _outputs,
    ):
        get_workspace = _get_workspace_symbol(runtime)
        workspace_size = ctypes.c_uint64(0)
        executor = ctypes.c_void_p()
        status = get_workspace(
            *args, ctypes.byref(workspace_size), ctypes.byref(executor)
        )

    assert status == expected_status
    assert not executor.value
