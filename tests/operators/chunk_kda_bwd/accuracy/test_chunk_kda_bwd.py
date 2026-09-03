"""A5 fused KDA backward regression for strong safe-gate decay."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch


CASE_FILE = Path(__file__).resolve().parents[3] / "op_cases/chunk_kda_bwd.json"
with CASE_FILE.open(encoding="utf-8") as file:
    MANIFEST = json.load(file)
CASES = MANIFEST["cases"]


def test_case_manifest_contract():
    assert MANIFEST["op"] == "chunk_kda_bwd"
    assert MANIFEST["implementation"] == "ascendc"
    assert MANIFEST["capability"] == {
        "run_on": ["ascendc"],
        "soc": ["ascend950"],
        "layout": ["BNSD"],
    }
    assert len({case["id"] for case in CASES}) == len(CASES) == 3
    for case in CASES:
        shape = case["shape"]
        attrs = case["attrs"]
        assert case["soc"] == ["ascend950"]
        assert case["run_on"] == ["ascendc"]
        assert attrs["layout"] == "BNSD"
        assert shape["batch"] == 1
        assert shape["key_dim"] == shape["value_dim"] == 128
        assert shape["chunk_size"] == 64
        assert attrs["safe_gate"] is True
        assert attrs["use_gate_in_kernel"] is True
        assert attrs["use_exp2"] is True
        assert case["dtype"] == {"input": "bfloat16", "gate": "float32"}
        assert case["generator"] == "zero_forward_consistent_safe_gate"
        assert case["expect"]["outputs"] == "finite_exact_zero"
        assert _expected_anchor(case) == case["expect"]["anchor"]

    model_case = next(case for case in CASES if "model_shape" in case["tags"])
    shape = model_case["shape"]
    assert (shape["batch"], shape["heads"], shape["seqlen"]) == (
        1,
        96,
        4096,
    )
    assert model_case["attrs"]["lower_bound"] == -5.0


def _expected_anchor(case: dict[str, object]) -> str:
    shape = case["shape"]
    attrs = case["attrs"]
    max_log2_magnitude = 120.0
    min_lower_bound = (
        -max_log2_magnitude * math.log(2.0) / int(shape["chunk_size"])
    )
    can_share = (
        attrs["use_gate_in_kernel"]
        and attrs["layout"] == "BNSD"
        and int(shape["key_dim"]) == 128
        and int(shape["chunk_size"]) == 64
        and int(shape["seqlen"]) % 64 == 0
        and int(shape["seqlen"]) >= 1024
        and min_lower_bound <= float(attrs["lower_bound"]) <= 0.0
    )
    return "shared" if can_share else "local"


def _is_ascend950() -> bool:
    try:
        import torch_npu  # noqa: F401

        return "950" in torch.npu.get_device_name(0)
    except (ImportError, RuntimeError):
        return False


def _make_zero_forward_case(
    case: dict[str, object], device: torch.device
) -> dict[str, torch.Tensor]:
    shape = case["shape"]
    attrs = case["attrs"]
    batch = int(shape["batch"])
    heads = int(shape["heads"])
    seqlen = int(shape["seqlen"])
    key_dim = int(shape["key_dim"])
    value_dim = int(shape["value_dim"])
    chunk_size = int(shape["chunk_size"])
    lower_bound = float(attrs["lower_bound"])
    raw_gate_value = float(case["inputs"]["raw_gate"])
    vector_shape = (batch, heads, seqlen, key_dim)
    input_dtype = getattr(torch, str(case["dtype"]["input"]))
    gate_dtype = getattr(torch, str(case["dtype"]["gate"]))

    gate_step = lower_bound * torch.sigmoid(
        torch.tensor(raw_gate_value, dtype=gate_dtype)
    )
    chunk_gate = (
        torch.arange(1, chunk_size + 1, dtype=gate_dtype)
        .view(1, 1, 1, chunk_size, 1)
        .mul_(gate_step / math.log(2.0))
    )
    gk = (
        chunk_gate.repeat(1, heads, seqlen // chunk_size, 1, key_dim)
        .reshape(vector_shape)
        .contiguous()
        .to(device)
    )

    identity_rows = torch.eye(chunk_size, dtype=input_dtype).repeat(
        seqlen // chunk_size, 1
    )
    akk = (
        identity_rows.view(1, 1, seqlen, chunk_size)
        .repeat(batch, heads, 1, 1)
        .contiguous()
        .to(device)
    )
    zeros = torch.zeros(vector_shape, dtype=input_dtype, device=device)
    return {
        "q": zeros.clone(),
        "k": zeros.clone(),
        "v": torch.zeros(
            (batch, heads, seqlen, value_dim), dtype=input_dtype, device=device
        ),
        "beta": torch.full(
            (batch, heads, seqlen), 0.5, dtype=gate_dtype, device=device
        ),
        "gk": gk,
        "aqk": torch.zeros(
            (batch, heads, seqlen, chunk_size),
            dtype=input_dtype,
            device=device,
        ),
        "akk": akk,
        "w": zeros.clone(),
        "qg": zeros.clone(),
        "kg": zeros.clone(),
        "v_new": torch.zeros(
            (batch, heads, seqlen, value_dim), dtype=input_dtype, device=device
        ),
        "h": torch.zeros(
            (
                batch,
                seqlen // chunk_size,
                heads,
                key_dim,
                value_dim,
            ),
            dtype=input_dtype,
            device=device,
        ),
        "d_o": torch.ones(
            (batch, heads, seqlen, value_dim), dtype=input_dtype, device=device
        ),
        "raw_g": torch.full(
            vector_shape, raw_gate_value, dtype=input_dtype, device=device
        ),
        "a_log": torch.zeros(heads, dtype=gate_dtype, device=device),
        "dt_bias": torch.zeros(
            (heads, key_dim), dtype=gate_dtype, device=device
        ),
    }


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
@torch.inference_mode()
def test_a5_dense_fused_backward_strong_gate(case):
    if not _is_ascend950():
        pytest.skip("requires a built chunk_kda_bwd OPP and an A5 NPU")

    from fla_npu.ops.ascendc import chunk_kda_bwd

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    inputs = _make_zero_forward_case(case, device)
    assert torch.isfinite(inputs["gk"]).all()
    assert float(inputs["gk"].min().cpu()) < float(case["expect"]["gk_min_lt"])

    shape = case["shape"]
    attrs = case["attrs"]

    outputs = chunk_kda_bwd(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["beta"],
        inputs["gk"],
        inputs["aqk"],
        inputs["akk"],
        inputs["w"],
        inputs["qg"],
        inputs["kg"],
        inputs["v_new"],
        inputs["h"],
        inputs["d_o"],
        int(shape["key_dim"]) ** -0.5,
        raw_g=inputs["raw_g"],
        A_log=inputs["a_log"],
        dt_bias=inputs["dt_bias"],
        chunk_size=int(shape["chunk_size"]),
        safe_gate=bool(attrs["safe_gate"]),
        lower_bound=float(attrs["lower_bound"]),
        use_gate_in_kernel=bool(attrs["use_gate_in_kernel"]),
        disable_recompute=True,
        use_exp2=bool(attrs["use_exp2"]),
        state_v_first=False,
    )
    torch.npu.synchronize()

    assert len(outputs) == 8
    assert outputs[5] is None
    assert outputs[6] is not None
    assert outputs[7] is not None
    for index, output in enumerate(outputs):
        if output is not None:
            assert torch.isfinite(output).all(), f"output {index} is non-finite"
            assert torch.count_nonzero(output).item() == 0, (
                f"output {index} is not exactly zero"
            )
