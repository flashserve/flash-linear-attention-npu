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
    positive_cases = [case for case in cases if "negative" not in case["tags"]]
    assert any(case["layout"] == "BSND" and case["optional_inputs"].get("cu_seqlens") for case in positive_cases)
    assert any(case["layout"] == "TND" and case["optional_inputs"].get("cu_seqlens") for case in positive_cases)
    assert any(case["attrs"].get("use_gate_in_kernel") for case in positive_cases)
    assert any(not case["attrs"].get("use_gate_in_kernel") for case in positive_cases)
    assert any(case["attrs"].get("safe_gate") for case in positive_cases)
    assert any(case["attrs"].get("use_gate_in_kernel") and case["optional_inputs"].get("dt_bias") is None
               for case in positive_cases)
    assert any(case["optional_inputs"].get("dt_bias") == "[H_v*K]" for case in positive_cases)
    assert any(case["attrs"].get("output_final_state") is False for case in positive_cases)
    assert any(case["optional_inputs"].get("ssm_state_indices") is not None for case in positive_cases)
    assert any(case["dtype"].get("g_beta") in ("float16", "bfloat16") for case in positive_cases)
    supported_kv = {(128, 128), (128, 256)}
    positive_kv = {(int(case["shape"]["K"]), int(case["shape"]["V"])) for case in positive_cases}
    assert positive_kv <= supported_kv
    assert supported_kv <= positive_kv
    assert any(int(case["shape"]["H"]) == 96 and int(case["shape"]["H_v"]) == 96 for case in positive_cases)
    assert any(int(case["shape"]["H_v"]) // int(case["shape"]["H"]) > 1 for case in positive_cases)
    assert any(int(case["shape"].get("max_seq_len", case["shape"].get("T", 1))) == 8 for case in positive_cases)


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
        cu_seqlens = _cu_seqlens_from_case(case)
        return {
            "q": (batch, seq_len, heads, key_dim),
            "v": (batch, seq_len, value_heads, value_dim),
            "g": (batch, seq_len, value_heads, key_dim),
            "beta": (batch, seq_len, value_heads),
            "seq_num": int(shape.get("seq_num", len(cu_seqlens) - 1 if cu_seqlens is not None else batch)),
            "cu_seqlens": cu_seqlens,
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
            "cu_seqlens": _cu_seqlens_from_case(case),
        }
    raise ValueError(layout)


def _cu_seqlens_from_case(case):
    spec = case["optional_inputs"].get("cu_seqlens")
    if spec is None or isinstance(spec, list):
        return spec
    if spec != "generated_varlen_max8_total":
        raise ValueError(f"{case['id']}: unsupported cu_seqlens generator {spec!r}")

    shape = case["shape"]
    total = int(shape["T_total"])
    max_seq_len = int(shape["max_seq_len"])
    if max_seq_len != 8:
        raise ValueError(f"{case['id']}: generated_varlen_max8_total expects max_seq_len=8")
    special_lengths = [0, 8, 1, 7, 2, 6, 3, 5, 4, 4, 0, 8]
    remaining = total - sum(special_lengths)
    if remaining < 0 or remaining % max_seq_len != 0:
        raise ValueError(f"{case['id']}: cannot generate cu_seqlens for total={total}")
    lengths = [max_seq_len] * (remaining // max_seq_len) + special_lengths
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    if cu[-1] != total or len(cu) - 1 != int(shape["seq_num"]):
        raise ValueError(f"{case['id']}: generated cu_seqlens does not match manifest shape")
    return cu


def _torch_dtype(torch, name):
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


def _randn(torch, shape, generator, dtype, scale=1.0):
    return (torch.randn(shape, generator=generator, dtype=torch.float32) * scale).to(dtype)


def _sequence_lengths(cu_seqlens):
    if cu_seqlens is None:
        return None
    return [right - left for left, right in zip(cu_seqlens, cu_seqlens[1:])]


def _generated_sequence_slots(cu_seqlens):
    slots = []
    for seq_idx, length in enumerate(_sequence_lengths(cu_seqlens)):
        slots.extend([seq_idx] * length)
    return slots


def _generated_last_token_accepts(cu_seqlens):
    return [max(1, length) for length in _sequence_lengths(cu_seqlens)]


def _make_optional_index_tensor(case, torch, key, cu_seqlens):
    spec = case["optional_inputs"].get(key)
    if spec is None:
        return None
    if isinstance(spec, list):
        return torch.tensor(spec, dtype=torch.int64)
    if key == "ssm_state_indices" and spec == "generated_sequence_slots_int32":
        return torch.tensor(_generated_sequence_slots(cu_seqlens), dtype=torch.int32)
    if key == "ssm_state_indices" and spec == "generated_sequence_slots_int64":
        return torch.tensor(_generated_sequence_slots(cu_seqlens), dtype=torch.int64)
    if key == "num_accepted_tokens" and spec == "generated_last_token_int32":
        return torch.tensor(_generated_last_token_accepts(cu_seqlens), dtype=torch.int32)
    if key == "num_accepted_tokens" and spec == "generated_last_token_int64":
        return torch.tensor(_generated_last_token_accepts(cu_seqlens), dtype=torch.int64)
    raise ValueError(f"{case['id']}: unsupported {key} generator {spec!r}")


def _make_inputs(case, torch):
    shapes = _shape_from_case(case)
    generator = torch.Generator().manual_seed(int(case["seed"]))
    qkv_dtype = _torch_dtype(torch, case["dtype"].get("q_k_v", "bfloat16"))
    g_dtype = _torch_dtype(torch, case["dtype"].get("g", case["dtype"].get("g_beta", "float32")))
    beta_dtype = _torch_dtype(torch, case["dtype"].get("beta", case["dtype"].get("g_beta", "float32")))
    q = _randn(torch, shapes["q"], generator, qkv_dtype)
    k = _randn(torch, shapes["q"], generator, qkv_dtype)
    v = _randn(torch, shapes["v"], generator, qkv_dtype)
    g = _randn(torch, shapes["g"], generator, g_dtype, scale=0.5)
    beta = _randn(torch, shapes["beta"], generator, beta_dtype)
    state_shape = (shapes["seq_num"], shapes["v"][-2], shapes["v"][-1], shapes["q"][-1])
    initial_spec = case["optional_inputs"].get("initial_state")
    initial_state = None
    if initial_spec == "present":
        state_dtype = _torch_dtype(torch, case["dtype"].get("state", "float32"))
        initial_state = _randn(torch, state_shape, generator, state_dtype, scale=0.02)
    attrs = case["attrs"]
    A_log = None
    dt_bias = None
    if attrs.get("use_gate_in_kernel", False):
        A_log = _randn(torch, (shapes["v"][-2],), generator, torch.float32, scale=0.1)
        dt_bias_spec = case["optional_inputs"].get("dt_bias")
        if dt_bias_spec == "[H_v,K]":
            dt_bias = _randn(torch, (shapes["v"][-2], shapes["q"][-1]), generator, torch.float32, scale=0.1)
        elif dt_bias_spec == "[H_v*K]":
            dt_bias = _randn(torch, (shapes["v"][-2] * shapes["q"][-1],), generator, torch.float32, scale=0.1)
        elif dt_bias_spec is not None:
            raise ValueError(f"{case['id']}: unsupported dt_bias spec {dt_bias_spec!r}")
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "initial_state": initial_state,
        "cu_seqlens": shapes["cu_seqlens"],
        "ssm_state_indices": _make_optional_index_tensor(case, torch, "ssm_state_indices", shapes["cu_seqlens"]),
        "A_log": A_log,
        "dt_bias": dt_bias,
        "num_accepted_tokens": _make_optional_index_tensor(case, torch, "num_accepted_tokens", shapes["cu_seqlens"]),
    }


@pytest.mark.npu
def test_json_accuracy_cases():
    if os.environ.get("FLA_NPU_RUN_OPERATOR_TESTS") != "1":
        pytest.skip("set FLA_NPU_RUN_OPERATOR_TESTS=1 on an NPU test host")

    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    from fla_npu.ops.ascendc import npu_recurrent_kda as recurrent_kda
    from tests.reference.recurrent_kda_reference import recurrent_kda_reference

    device_id = int(os.environ.get("TEST_DEVICE_ID", "0"))
    device = torch.device(f"npu:{device_id}")
    torch.npu.set_device(device)
    cases = select_cases(OP, tags=("accuracy",), route="ascendc", include_negative=False)
    tol = load_cases(OP)["tolerance"]["bfloat16"]
    for case in cases:
        inputs = _make_inputs(case, torch)
        attrs = dict(case["attrs"])
        expected = recurrent_kda_reference(**inputs, **attrs)
        call_kwargs = {
            key: value
            for key, value in attrs.items()
            if key not in ("scale",) or value is not None
        }
        if inputs["cu_seqlens"] is not None:
            call_kwargs["cu_seqlens"] = inputs["cu_seqlens"]
        optional_tensor_kwargs = {
            "ssm_state_indices": inputs["ssm_state_indices"],
            "A_log": inputs["A_log"],
            "dt_bias": inputs["dt_bias"],
            "num_accepted_tokens": inputs["num_accepted_tokens"],
        }
        optional_tensor_kwargs = {
            key: value.to(device) if value is not None else None
            for key, value in optional_tensor_kwargs.items()
        }
        out, final_state = recurrent_kda(
            inputs["q"].to(device),
            inputs["k"].to(device),
            inputs["v"].to(device),
            inputs["g"].to(device),
            inputs["beta"].to(device),
            inputs["initial_state"].to(device) if inputs["initial_state"] is not None else None,
            **optional_tensor_kwargs,
            **call_kwargs,
        )
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu().float(), expected[0].float(), rtol=tol["rtol"], atol=tol["atol"])
        if attrs.get("output_final_state", False):
            torch.testing.assert_close(final_state.cpu().float(), expected[1].float(), rtol=tol["rtol"], atol=tol["atol"])
        else:
            assert tuple(final_state.shape) == (0,)


def _assert_sample_finite(torch, tensor, name):
    if not tensor.is_floating_point() or tensor.numel() == 0:
        return
    flat = tensor.reshape(-1)
    sample_size = min(int(flat.numel()), 4096)
    samples = [flat[:sample_size]]
    if flat.numel() > sample_size:
        samples.append(flat[-sample_size:])
    if tensor.dim() >= 1 and tensor.shape[0] > 1:
        samples.append(tensor[0].reshape(-1)[:sample_size])
        samples.append(tensor[-1].reshape(-1)[:sample_size])
    for sample in samples:
        if sample.numel() and not bool(torch.isfinite(sample.float()).all().item()):
            raise AssertionError(f"{name} contains NaN or Inf in sampled values")


def _device_full(torch, shape, value, dtype, device):
    return torch.full(shape, value, dtype=dtype).to(device).contiguous()


def _device_zeros(torch, shape, dtype, device):
    return torch.zeros(shape, dtype=dtype).to(device).contiguous()


@pytest.mark.npu
def test_json_large_shape_cases():
    if os.environ.get("FLA_NPU_RUN_OPERATOR_TESTS") != "1":
        pytest.skip("set FLA_NPU_RUN_OPERATOR_TESTS=1 on an NPU test host")
    if os.environ.get("FLA_NPU_RUN_LARGE_SHAPE_TESTS") != "1":
        pytest.skip("set FLA_NPU_RUN_LARGE_SHAPE_TESTS=1 for long-context Kimi shape smoke")

    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    from fla_npu.ops.ascendc import npu_recurrent_kda as recurrent_kda

    device_id = int(os.environ.get("TEST_DEVICE_ID", "0"))
    device = torch.device(f"npu:{device_id}")
    torch.npu.set_device(device)
    cases = select_cases(OP, tags=("large_shape",), route="ascendc", include_negative=False)
    assert cases, f"{OP} has no executable large-shape cases"
    for case in cases:
        shapes = _shape_from_case(case)
        cu_seqlens = shapes["cu_seqlens"]
        lengths = [right - left for left, right in zip(cu_seqlens, cu_seqlens[1:])]
        assert len(cu_seqlens) == int(case["shape"]["seq_num"]) + 1
        assert cu_seqlens[-1] == int(case["shape"]["T_total"])
        assert max(lengths) <= int(case["shape"]["max_seq_len"])
        assert {0, 1, 2, 3, 4, 5, 6, 7, 8} <= set(lengths)

        q = _device_full(torch, shapes["q"], 0.125, torch.bfloat16, device)
        k = _device_full(torch, shapes["q"], 0.125, torch.bfloat16, device)
        v = _device_full(torch, shapes["v"], 0.25, torch.bfloat16, device)
        g = _device_zeros(torch, shapes["g"], torch.float32, device)
        beta = _device_zeros(torch, shapes["beta"], torch.float32, device)
        state_dtype = _torch_dtype(torch, case["dtype"]["state"])
        state_shape = (shapes["seq_num"], shapes["v"][-2], shapes["v"][-1], shapes["q"][-1])
        initial_state = _device_zeros(torch, state_shape, state_dtype, device)
        A_log = _device_zeros(torch, (shapes["v"][-2],), torch.float32, device)
        dt_bias = _device_zeros(torch, (shapes["v"][-2], shapes["q"][-1]), torch.float32, device)

        attrs = dict(case["attrs"])
        attrs["output_final_state"] = True
        call_kwargs = {
            key: value
            for key, value in attrs.items()
            if key not in ("scale",) or value is not None
        }
        out, final_state = recurrent_kda(
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            cu_seqlens=cu_seqlens,
            A_log=A_log,
            dt_bias=dt_bias,
            **call_kwargs,
        )
        torch.npu.synchronize()
        assert tuple(out.shape) == shapes["v"]
        assert tuple(final_state.shape) == state_shape
        for zero_idx in [idx for idx, length in enumerate(lengths) if length == 0][:2]:
            zero_state_max = final_state[zero_idx].float().abs().max().cpu().item()
            assert zero_state_max == 0.0, f"{case['id']} zero-length seq {zero_idx} final_state changed"
        _assert_sample_finite(torch, out, f"{case['id']}/out")
        _assert_sample_finite(torch, final_state, f"{case['id']}/final_state")
        print(
            f"[PASS] {OP}/{case['id']} T_total={case['shape']['T_total']} "
            f"H={case['shape']['H']} HV={case['shape']['H_v']} D={case['shape']['K']}"
        )

        del q, k, v, g, beta, initial_state, A_log, dt_bias, out, final_state
        torch.npu.empty_cache()
