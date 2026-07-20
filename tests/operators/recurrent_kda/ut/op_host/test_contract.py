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
    initial_state_block = source[
        source.index('this->Input("initial_state")'):source.index('this->Input("cu_seqlens")')
    ]
    final_state_block = source[
        source.index('this->Output("final_state")'):source.index('this->Attr("layout")')
    ]

    assert ".IgnoreContiguous()" in initial_state_block
    assert ".IgnoreContiguous()" in final_state_block


def test_state_view_contract_uses_work_tensors():
    source = _read_repo_file("fla/ops/ascendc/kda/recurrent_kda/op_host/op_api/aclnn_recurrent_kda.cpp")

    assert "DataContiguous(initialStateForKernel" in source
    assert "finalStateNeedViewCopy" in source
    assert "ViewCopy(result[1], params.finalState" in source
