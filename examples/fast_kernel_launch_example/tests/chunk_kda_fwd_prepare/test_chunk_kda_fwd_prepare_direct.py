import json
import sys
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

import ascend_ops  # noqa: F401
from fla_npu.ops.ascendc import chunk_kda_fwd_prepare


ROOT = Path(__file__).resolve().parents[4]
COMMON_DIR = ROOT / "tests/operators/chunk_kda_fwd_prepare/common"
sys.path.insert(0, str(COMMON_DIR))

from runtime import OUTPUT_NAMES, build_inputs, check_output_contract  # noqa: E402


CASE_IDS = (
    "prepare_dense_bnsd_raw",
    "prepare_dense_bsnd_fused",
    "prepare_dense_bsnd_fused_save",
)
OUTPUT_MODES = {"none": 0, "recompute": 1, "save": 2}
ACL_FORMAT_FRACTAL_NZ = 29


def _load_case(case_id):
    manifest = json.loads(
        (ROOT / "tests/op_cases/chunk_kda_fwd_prepare.json").read_text(
            encoding="utf-8"
        )
    )
    return next(case for case in manifest["cases"] if case["id"] == case_id)


def _call_direct(inputs, attrs, **overrides):
    q, k, v, g, beta, a_log, dt_bias = inputs
    arguments = {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "a_log": a_log,
        "dt_bias": dt_bias,
        "layout": str(attrs["layout"]),
        "scale": float(attrs["scale"]),
        "chunk_size": int(attrs["chunk_size"]),
        "epsilon": float(attrs["epsilon"]),
        "use_qk_l2norm_in_kernel": bool(
            attrs["use_qk_l2norm_in_kernel"]
        ),
        "use_gate_in_kernel": bool(attrs["use_gate_in_kernel"]),
        "use_beta_sigmoid_in_kernel": bool(
            attrs["use_beta_sigmoid_in_kernel"]
        ),
        "allow_neg_eigval": bool(attrs["allow_neg_eigval"]),
        "safe_gate": bool(attrs["safe_gate"]),
        "lower_bound": float(attrs["lower_bound"]),
        "use_exp2": bool(attrs["use_exp2"]),
        "output_mode": OUTPUT_MODES[str(attrs["backward_mode"])],
    }
    arguments.update(overrides)
    return torch.ops.ascend_ops.chunk_kda_fwd_prepare_direct(
        arguments["q"],
        arguments["k"],
        arguments["v"],
        arguments["g"],
        arguments["beta"],
        arguments["a_log"],
        arguments["dt_bias"],
        arguments["layout"],
        arguments["scale"],
        arguments["chunk_size"],
        arguments["epsilon"],
        arguments["use_qk_l2norm_in_kernel"],
        arguments["use_gate_in_kernel"],
        arguments["use_beta_sigmoid_in_kernel"],
        arguments["allow_neg_eigval"],
        arguments["safe_gate"],
        arguments["lower_bound"],
        arguments["use_exp2"],
        arguments["output_mode"],
    )


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_chunk_kda_fwd_prepare_direct_matches_aclnn(case_id):
    case = _load_case(case_id)
    attrs = case["attrs"]
    device = torch.device("npu:0")
    q, k, v, g, beta, a_log, dt_bias = build_inputs(torch, device, case)

    reference = chunk_kda_fwd_prepare(
        q,
        k,
        v,
        g,
        beta,
        float(attrs["scale"]),
        layout=str(attrs["layout"]),
        chunk_size=int(attrs["chunk_size"]),
        epsilon=float(attrs["epsilon"]),
        use_qk_l2norm_in_kernel=bool(attrs["use_qk_l2norm_in_kernel"]),
        use_gate_in_kernel=bool(attrs["use_gate_in_kernel"]),
        use_beta_sigmoid_in_kernel=bool(attrs["use_beta_sigmoid_in_kernel"]),
        allow_neg_eigval=bool(attrs["allow_neg_eigval"]),
        safe_gate=bool(attrs["safe_gate"]),
        lower_bound=float(attrs["lower_bound"]),
        use_exp2=bool(attrs["use_exp2"]),
        a_log=a_log,
        dt_bias=dt_bias,
        backward_mode=str(attrs["backward_mode"]),
    )
    direct = _call_direct(
        (q, k, v, g, beta, a_log, dt_bias), attrs
    )
    torch.npu.synchronize()

    assert len(direct) == len(OUTPUT_NAMES) == 13
    check_output_contract(torch, case, direct)
    for name, actual, expected in zip(OUTPUT_NAMES, direct, reference):
        if expected is None:
            assert actual is None, name
            continue
        assert actual is not None, name
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=name)


@pytest.mark.parametrize(
    ("input_index", "input_name"),
    ((0, "q"), (1, "k"), (2, "v"), (3, "g"), (4, "beta")),
)
def test_chunk_kda_fwd_prepare_direct_rejects_private_required_input_format(
    input_index, input_name
):
    case = _load_case("prepare_dense_bnsd_raw")
    attrs = case["attrs"]
    inputs = list(build_inputs(torch, torch.device("npu:0"), case))
    inputs[input_index] = torch_npu.npu_format_cast(
        inputs[input_index], ACL_FORMAT_FRACTAL_NZ
    )
    assert (
        torch_npu.get_npu_format(inputs[input_index])
        == ACL_FORMAT_FRACTAL_NZ
    )

    with pytest.raises(
        RuntimeError, match=rf"{input_name}.*private NPU format 29"
    ):
        _call_direct(tuple(inputs), attrs)


@pytest.mark.parametrize(
    ("input_index", "input_name"), ((5, "a_log"), (6, "dt_bias"))
)
def test_chunk_kda_fwd_prepare_direct_rejects_private_optional_input_format(
    input_index, input_name
):
    case = _load_case("prepare_dense_bsnd_fused")
    attrs = case["attrs"]
    inputs = list(build_inputs(torch, torch.device("npu:0"), case))
    assert inputs[input_index] is not None
    try:
        private_input = torch_npu.npu_format_cast(
            inputs[input_index], ACL_FORMAT_FRACTAL_NZ
        )
    except (RuntimeError, ValueError) as error:
        error_text = str(error).lower()
        if not any(
            marker in error_text
            for marker in ("format", "fractal", "dimension", "dim", "rank")
        ):
            raise
        pytest.skip(
            f"目标 torch_npu 不支持为 rank-{inputs[input_index].dim()} "
            f"的 {input_name} 构造 FRACTAL_NZ"
        )
    inputs[input_index] = private_input
    assert (
        torch_npu.get_npu_format(inputs[input_index])
        == ACL_FORMAT_FRACTAL_NZ
    )

    with pytest.raises(
        RuntimeError, match=rf"{input_name}.*private NPU format 29"
    ):
        _call_direct(tuple(inputs), attrs)


@pytest.mark.parametrize(
    ("case_id", "attribute", "value", "message"),
    (
        (
            "prepare_dense_bnsd_raw",
            "scale",
            1e300,
            "scale must be finite",
        ),
        (
            "prepare_dense_bnsd_raw",
            "epsilon",
            1e-300,
            "epsilon must be finite and greater than zero",
        ),
        (
            "prepare_dense_bnsd_raw",
            "lower_bound",
            1e300,
            "lower_bound must be finite",
        ),
        (
            "prepare_dense_bsnd_fused",
            "lower_bound",
            0.0,
            r"lower_bound must be in \[-5, 0\)",
        ),
    ),
)
def test_chunk_kda_fwd_prepare_direct_rejects_invalid_fp32_attributes(
    case_id, attribute, value, message
):
    case = _load_case(case_id)
    attrs = case["attrs"]
    inputs = build_inputs(torch, torch.device("npu:0"), case)

    with pytest.raises(RuntimeError, match=message):
        _call_direct(inputs, attrs, **{attribute: value})
