"""Static op_host contract for chunk_bwd_dqkwg; device execution lives in accuracy/routes."""

from pathlib import Path

from tests.operators.chunk_bwd_dqkwg.common.case_matrix import manifest


def test_host_contract_has_platform_and_negative_matrix():
    data = manifest()
    assert set(data["capability"]["soc"]) >= {"ascend910b", "ascend910_93", "ascend950"}
    negatives = [case for case in data["cases"] if "negative" in case["tags"]]
    assert negatives
    for case in negatives:
        assert case["expect"]["return_code"] != "ACLNN_SUCCESS"
        assert case["expect"].get("message_contains")
        assert "aclnn" in case["run_on"] or case["expect"]["return_code"] == "RuntimeError"


def test_route_case_uses_one_shape_definition():
    data = manifest()
    route_cases = [case for case in data["cases"] if "route" in case["tags"]]
    assert route_cases
    assert any({"ascendc", "aclnn", "direct_launch"} <= set(case["run_on"]) for case in route_cases)


def test_tiling_key_selects_the_declared_dtype_templates():
    source = (
        Path(__file__).resolve().parents[5]
        / "fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dqkwg/op_host/op_tiling/chunk_bwd_dqkwg_tiling.cpp"
    ).read_text(encoding="utf-8")
    assert "GET_TPL_TILING_KEY(strategyKey, dTQ, dTG, V)" in source
    assert "CHUNK_BWD_DQKWG_TPL_FP32" in source
    assert "context->SetTilingKey(1);" not in source
