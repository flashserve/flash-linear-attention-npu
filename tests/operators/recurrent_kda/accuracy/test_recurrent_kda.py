"""Canonical JSON entry for RecurrentKda accuracy/generalization tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.operators._shared.cases import load_cases, select_cases


OP = "recurrent_kda"
ROOT = Path(__file__).resolve().parents[4]


def test_case_manifest_covers_required_matrix():
    manifest = load_cases(OP)
    cases = manifest["cases"]
    tags = {tag for case in cases for tag in case["tags"]}
    assert {"accuracy", "generalization", "negative", "route"} <= tags
    assert {"ascend910b", "ascend910_93", "ascend950"} <= set(manifest["capability"]["soc"])
    assert all("ascendc" in case["run_on"] for case in cases if "accuracy" in case["tags"])


def test_selected_case_ids_are_unique():
    selected = select_cases(OP)
    ids = [case["id"] for case in selected]
    assert len(ids) == len(set(ids))


def _shape_from_case(case):
    shape = case["shape"]
    layout = case["layout"]
    heads = int(shape["H"])
    value_heads = int(shape["H_v"])
    key_dim = int(shape["K"])
    value_dim = int(shape["V"])
    if layout == "BSND":
        batch = int(shape["B"])
        seq_len = int(shape["T"])
        return {
            "q": (batch, seq_len, heads, key_dim),
            "v": (batch, seq_len, value_heads, value_dim),
            "g": (batch, seq_len, value_heads, key_dim),
            "beta": (batch, seq_len, value_heads),
            "seq_num": batch,
            "cu_seqlens": None,
        }
    if layout == "TND":
        total_tokens = int(shape["T_total"])
        seq_num = int(shape["seq_num"])
        return {
            "q": (total_tokens, heads, key_dim),
            "v": (total_tokens, value_heads, value_dim),
            "g": (total_tokens, value_heads, key_dim),
            "beta": (total_tokens, value_heads),
            "seq_num": seq_num,
            "cu_seqlens": case["optional_inputs"].get("cu_seqlens"),
        }
    raise ValueError(layout)


def _make_inputs(case, torch):
    shapes = _shape_from_case(case)
    generator = torch.Generator().manual_seed(int(case["seed"]))
    q = torch.randn(shapes["q"], generator=generator, dtype=torch.bfloat16)
    k = torch.randn(shapes["q"], generator=generator, dtype=torch.bfloat16)
    v = torch.randn(shapes["v"], generator=generator, dtype=torch.bfloat16)
    g = torch.randn(shapes["g"], generator=generator, dtype=torch.float32) * 0.5
    beta = torch.randn(shapes["beta"], generator=generator, dtype=torch.float32)
    state_shape = (shapes["seq_num"], shapes["v"][-2], shapes["v"][-1], shapes["q"][-1])
    initial_spec = case["optional_inputs"].get("initial_state")
    initial_state = None
    if initial_spec == "present":
        initial_state = torch.randn(state_shape, generator=generator, dtype=torch.float32) * 0.02
    attrs = case["attrs"]
    A_log = None
    dt_bias = None
    if attrs.get("use_gate_in_kernel", False):
        A_log = torch.randn((shapes["v"][-2],), generator=generator, dtype=torch.float32) * 0.1
        if case["optional_inputs"].get("dt_bias") is not None:
            dt_bias = torch.randn((shapes["v"][-2], shapes["q"][-1]), generator=generator, dtype=torch.float32) * 0.1
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "initial_state": initial_state,
        "cu_seqlens": shapes["cu_seqlens"],
        "A_log": A_log,
        "dt_bias": dt_bias,
    }


@pytest.mark.npu
def test_json_accuracy_cases():
    if os.environ.get("FLA_NPU_RUN_OPERATOR_TESTS") != "1":
        pytest.skip("set FLA_NPU_RUN_OPERATOR_TESTS=1 on an NPU test host")

    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    from fla_npu.ops.ascendc import recurrent_kda
    from tests.reference.recurrent_kda_reference import recurrent_kda_reference

    device_id = int(os.environ.get("TEST_DEVICE_ID", "0"))
    device = torch.device(f"npu:{device_id}")
    torch.npu.set_device(device)
    cases = select_cases(OP, tags=("accuracy",), route="ascendc", include_negative=False)
    tol = load_cases(OP)["tolerance"]["bfloat16"]
    for case in cases:
        inputs = _make_inputs(case, torch)
        attrs = dict(case["attrs"])
        attrs["output_final_state"] = True
        expected = recurrent_kda_reference(**inputs, **attrs)
        call_kwargs = {
            key: value
            for key, value in attrs.items()
            if key not in ("scale",) or value is not None
        }
        if inputs["cu_seqlens"] is not None:
            call_kwargs["cu_seqlens"] = inputs["cu_seqlens"]
        out, final_state = recurrent_kda(
            inputs["q"].to(device),
            inputs["k"].to(device),
            inputs["v"].to(device),
            inputs["g"].to(device),
            inputs["beta"].to(device),
            inputs["initial_state"].to(device) if inputs["initial_state"] is not None else None,
            A_log=inputs["A_log"].to(device) if inputs["A_log"] is not None else None,
            dt_bias=inputs["dt_bias"].to(device) if inputs["dt_bias"] is not None else None,
            **call_kwargs,
        )
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu().float(), expected[0].float(), rtol=tol["rtol"], atol=tol["atol"])
        torch.testing.assert_close(final_state.cpu().float(), expected[1].float(), rtol=tol["rtol"], atol=tol["atol"])
