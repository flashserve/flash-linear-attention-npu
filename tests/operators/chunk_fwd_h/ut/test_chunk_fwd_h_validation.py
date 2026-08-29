# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the BSD 3-Clause License (the "License").

import json
import re
import sys
import types
from pathlib import Path

import pytest
import torch

from fla_npu.ops.ascendc import chunk_fwd_h
import fla_npu.ops.ascendc._aclnn_ctypes as aclnn_ctypes


CASE_FILE = Path(__file__).resolve().parents[3] / "op_cases" / "chunk_fwd_h.json"
NEGATIVE_CASES = json.loads(CASE_FILE.read_text(encoding="utf-8"))["negative_cases"]


def _base_inputs(batch=1, k_heads=1, v_heads=1, tokens=1):
    return {
        "k": torch.empty(batch, k_heads, tokens, 128, dtype=torch.bfloat16),
        "w": torch.empty(batch, v_heads, tokens, 128, dtype=torch.bfloat16),
        "u": torch.empty(batch, v_heads, tokens, 128, dtype=torch.bfloat16),
        "g": torch.empty(batch, v_heads, tokens, dtype=torch.bfloat16),
    }


@pytest.mark.parametrize("case", NEGATIVE_CASES, ids=lambda case: case["id"])
def test_chunk_fwd_h_validation(case):
    assert case["expected_return_code"] == "ACLNN_ERR_PARAM_INVALID"
    mutation = case["mutation"]
    inputs = _base_inputs()
    kwargs = {}

    if mutation == "both_gate_inputs":
        kwargs["gk"] = torch.empty(1, 1, 1, 128, dtype=torch.bfloat16)
    elif mutation == "unsupported_chunk_size":
        kwargs["chunk_size"] = 32
    elif mutation == "invalid_g_head_ratio":
        inputs = _base_inputs(k_heads=2, v_heads=3)
    elif mutation == "invalid_gk_shape":
        inputs = _base_inputs(k_heads=2, v_heads=2)
        inputs.pop("g")
        kwargs["gk"] = torch.empty(1, 1, 1, 128, dtype=torch.bfloat16)
    elif mutation == "varlen_batch_not_one":
        inputs = _base_inputs(batch=2)
        kwargs["cu_seqlens"] = (0, 1)
    elif mutation == "invalid_state_v_first_shape":
        kwargs["state_v_first"] = True
        kwargs["initial_state"] = torch.empty(1, 1, 127, 128, dtype=torch.bfloat16)
    elif mutation == "missing_gate_inputs":
        inputs.pop("g")
    elif mutation == "unsupported_input_dtype":
        inputs["k"] = inputs["k"].float()
        inputs["w"] = inputs["w"].float()
        inputs["u"] = inputs["u"].float()
    elif mutation == "save_new_value_false":
        kwargs["save_new_value"] = False
    elif mutation == "chunk_indices_without_cu":
        kwargs["chunk_indices"] = (0, 0)
    elif mutation == "invalid_input_rank":
        inputs["k"] = torch.empty(1, 1, 128, dtype=torch.bfloat16)
    elif mutation == "non_positive_dimension":
        inputs = _base_inputs(batch=0)
    elif mutation == "mismatched_w_u_shape":
        inputs["w"] = torch.empty(1, 1, 2, 128, dtype=torch.bfloat16)
    elif mutation == "unsupported_kv_dimension":
        inputs["k"] = torch.empty(1, 1, 1, 64, dtype=torch.bfloat16)
        inputs["w"] = torch.empty(1, 1, 1, 64, dtype=torch.bfloat16)
    elif mutation == "invalid_g_shape":
        inputs["g"] = torch.empty(1, 1, 2, dtype=torch.bfloat16)
    elif mutation == "unsupported_gate_dtype":
        inputs["g"] = inputs["g"].to(torch.float16)
    elif mutation == "mismatched_input_dtype":
        inputs["w"] = inputs["w"].float()
    elif mutation == "invalid_cu_size":
        kwargs["cu_seqlens"] = (0,)
    elif mutation == "invalid_cu_start":
        kwargs["cu_seqlens"] = (1, 2)
    elif mutation == "invalid_cu_end":
        kwargs["cu_seqlens"] = (0, 2)
    elif mutation == "non_increasing_cu":
        kwargs["cu_seqlens"] = (0, 1, 1)
    elif mutation == "invalid_chunk_indices_length":
        kwargs["cu_seqlens"] = (0, 1)
        kwargs["chunk_indices"] = (0,)
    elif mutation == "invalid_chunk_indices_order":
        inputs = _base_inputs(tokens=65)
        kwargs["cu_seqlens"] = (0, 65)
        kwargs["chunk_indices"] = (0, 1, 0, 0)
    elif mutation == "unsupported_state_dtype":
        kwargs["initial_state"] = torch.empty(1, 1, 128, 128, dtype=torch.float16)
    else:
        raise AssertionError(f"unknown negative mutation: {mutation}")

    with pytest.raises(RuntimeError, match=re.escape(case["expected_message"])):
        chunk_fwd_h(**inputs, **kwargs)


class _FakeCallContext:
    def __init__(self):
        self.tensors = {}

    def tensor(self, tensor, name, **kwargs):
        self.tensors[name] = (tensor, kwargs)
        return object()

    def int_array(self, values):
        return object()


def _capture_direct_call(monkeypatch):
    captured = {}

    def fake_call(name, build_args, outputs):
        ctx = _FakeCallContext()
        captured["name"] = name
        captured["args"] = build_args(ctx)
        captured["ctx"] = ctx
        return outputs

    monkeypatch.setattr(aclnn_ctypes, "_call_aclnn", fake_call)
    _set_mock_npu_format(monkeypatch, aclnn_ctypes.ACL_FORMAT_ND)
    return captured


def _set_mock_npu_format(monkeypatch, acl_format):
    monkeypatch.setattr(aclnn_ctypes, "_acl_format", lambda tensor: acl_format)
    loaded_torch_npu = sys.modules.get("torch_npu")
    if loaded_torch_npu is not None:
        monkeypatch.setattr(
            loaded_torch_npu, "get_npu_format", lambda tensor: acl_format
        )


def test_non_contiguous_input_uses_truthful_storage_and_contiguous_output(monkeypatch):
    captured = _capture_direct_call(monkeypatch)
    inputs = _base_inputs()
    inputs["u"] = torch.empty(1, 1, 1, 256, dtype=torch.bfloat16)[..., ::2]

    _, v_new, _ = aclnn_ctypes.npu_chunk_fwd_h(**inputs)

    assert captured["name"] == "aclnnChunkFwdH"
    assert captured["ctx"].tensors["u"][1]["storage_shape_override"] is None
    assert v_new.is_contiguous()
    assert captured["ctx"].tensors["v_new"][1]["storage_shape_override"] == (1, 1, 1, 128)


def test_private_npu_format_is_rejected_before_descriptor_override(monkeypatch):
    _capture_direct_call(monkeypatch)
    _set_mock_npu_format(monkeypatch, 29)

    with pytest.raises(RuntimeError, match="private NPU format 29 is not supported"):
        aclnn_ctypes.npu_chunk_fwd_h(**_base_inputs())


def test_loaded_torch_npu_format_query_failure_is_not_guessed_from_rank(monkeypatch):
    _capture_direct_call(monkeypatch)

    def fail_format_query(tensor):
        raise RuntimeError("format metadata unavailable")

    fake_torch_npu = types.SimpleNamespace(get_npu_format=fail_format_query)
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)

    with pytest.raises(RuntimeError, match="cannot determine the real NPU format of k"):
        aclnn_ctypes.npu_chunk_fwd_h(**_base_inputs())


def test_loaded_torch_npu_format_is_queried_once_per_tensor(monkeypatch):
    _capture_direct_call(monkeypatch)
    queried = set()

    def one_shot_format_query(tensor):
        tensor_id = id(tensor)
        if tensor_id in queried:
            raise RuntimeError("format metadata queried twice")
        queried.add(tensor_id)
        return aclnn_ctypes.ACL_FORMAT_ND

    fake_torch_npu = types.SimpleNamespace(get_npu_format=one_shot_format_query)
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)

    aclnn_ctypes.npu_chunk_fwd_h(**_base_inputs())
    assert queried
