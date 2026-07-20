"""Static op_host contract for recurrent_kda; device execution lives in accuracy/routes."""

from tests.operators.recurrent_kda.common.case_matrix import manifest


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
