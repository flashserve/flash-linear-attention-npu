# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the BSD 3-Clause License (the "License").

import json
import re
from pathlib import Path

import pytest
import torch

from fla_npu.ops.ascendc import chunk_fwd_h


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
    else:
        raise AssertionError(f"unknown negative mutation: {mutation}")

    with pytest.raises(RuntimeError, match=re.escape(case["expected_message"])):
        chunk_fwd_h(**inputs, **kwargs)
