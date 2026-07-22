"""Static op_host contract for recurrent_kda; device execution lives in accuracy/routes."""

from pathlib import Path

from tests.operators.recurrent_kda.common.case_matrix import manifest

ROOT = Path(__file__).resolve().parents[5]


def _read_repo_file(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_host_contract_has_platform_and_negative_matrix():
    data = manifest()
    assert set(data["capability"]["soc"]) >= {"ascend910b", "ascend910_93", "ascend950"}
    negatives = [case for case in data["cases"] if "negative" in case["tags"]]
    assert negatives
    for case in negatives:
        assert case["expect"]["return_code"] != "ACLNN_SUCCESS"
        assert case["expect"].get("message_contains_any")
        assert "aclnn" in case["run_on"] or case["expect"]["return_code"] == "RuntimeError"


def test_route_case_uses_one_shape_definition():
    data = manifest()
    route_cases = [case for case in data["cases"] if "route" in case["tags"]]
    assert route_cases
    assert any({"ascendc", "aclnn", "direct_launch"} <= set(case["run_on"]) for case in route_cases)


def test_raw_gate_contract_is_single_operator():
    data = manifest()
    raw_gate_cases = [case for case in data["cases"] if case["attrs"].get("use_gate_in_kernel")]
    assert raw_gate_cases
    assert all(case["optional_inputs"].get("A_log") == "[H_v]" for case in raw_gate_cases)


def test_positive_cases_stay_inside_supported_kv_enums():
    data = manifest()
    supported_kv = {(128, 128), (128, 256)}
    positive_cases = [case for case in data["cases"] if "negative" not in case["tags"]]
    assert positive_cases
    assert {
        (int(case["shape"]["K"]), int(case["shape"]["V"]))
        for case in positive_cases
    } <= supported_kv


def test_state_tensors_allow_non_contiguous_views():
    source = _read_repo_file("fla/ops/ascendc/kda/recurrent_kda/op_host/recurrent_kda_def.cpp")
    state_input_block = source[
        source.index('this->Input("state")'):source.index('this->Input("actual_seq_lengths")')
    ]
    state_output_block = source[
        source.index('this->Output("state")'):source.index('this->Attr("layout")')
    ]

    assert ".IgnoreContiguous()" in state_input_block
    assert ".IgnoreContiguous()" in state_output_block


def test_state_view_contract_uses_work_tensors():
    source = _read_repo_file("fla/ops/ascendc/kda/recurrent_kda/op_host/op_api/aclnn_recurrent_kda.cpp")

    assert "DataContiguous(contiguousState" in source
    assert "stateNeedViewCopy" in source
    assert "ViewCopy(result[1], params.stateRef" in source


def test_device_metadata_and_capacity_state_contract():
    api = _read_repo_file("fla/ops/ascendc/kda/recurrent_kda/op_host/op_api/aclnn_recurrent_kda.h")
    op_def = _read_repo_file("fla/ops/ascendc/kda/recurrent_kda/op_host/recurrent_kda_def.cpp")
    tiling = _read_repo_file("fla/ops/ascendc/kda/recurrent_kda/op_host/recurrent_kda_tiling_processor.h")

    assert "const aclTensor *actualSeqLengths" in api
    assert "aclIntArray" not in api
    actual_input = op_def[op_def.index('this->Input("actual_seq_lengths")'):]
    assert ".ParamType(REQUIRED)" in actual_input.split('this->Input("ssm_state_indices")', 1)[0]
    assert "speculative [seq_num,max_step]" in tiling
    assert "state_capacity must equal seq_num" in tiling


def test_actual_seq_lengths_uses_gdn_value_semantics():
    kernel = _read_repo_file("fla/ops/ascendc/kda/recurrent_kda/op_kernel/recurrent_kda.h")

    assert "int64_t seq0 = actualSeqLengthsGm_.GetValue(0)" in kernel
    assert "int64_t seqLen64 = actualSeqLengthsGm_.GetValue(batch_i + 1)" in kernel
    assert "total += length" in kernel
    assert "return total == static_cast<int64_t>(T_)" in kernel
    assert "seqLen64 = seq1 - seq0" not in kernel


def test_tiling_processor_owns_its_context():
    source = _read_repo_file(
        "fla/ops/ascendc/kda/recurrent_kda/op_host/recurrent_kda_tiling_processor.h"
    )

    assert "RecurrentKdaTilingContext ctx_;" in source
    assert "const RecurrentKdaTilingContext &ctx_;" not in source


def test_mutable_state_is_wired_as_an_inplace_output():
    l0_source = _read_repo_file("fla/ops/ascendc/kda/recurrent_kda/op_host/op_api/recurrent_kda.cpp")
    schema = _read_repo_file("torch_custom/fla_npu/npu_custom.yaml")
    ctypes_init = _read_repo_file("torch_custom/fla_npu/fla_npu/ops/ascendc/__init__.py")

    assert "OP_OUTPUT(out, stateRef)" in l0_source
    assert "Tensor(a!) initial_state" in schema
    assert "Tensor actual_seq_lengths" in schema
    assert '"npu_recurrent_kda": ("initial_state",)' in ctypes_init
