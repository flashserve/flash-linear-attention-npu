"""Static op_host contract for chunk_local_cumsum; device execution lives in accuracy/routes."""

from pathlib import Path

from tests.operators.chunk_local_cumsum.common.case_matrix import manifest


ROOT = Path(__file__).resolve().parents[5]
ACLNN_HEADER = (
    ROOT
    / "fla/ops/ascendc/gdn/gdn_preprocess/chunk_local_cumsum/op_host/op_api"
    / "aclnn_chunk_local_cumsum.h"
)


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


def test_varlen_metadata_uses_host_int_array_abi():
    header = ACLNN_HEADER.read_text(encoding="utf-8")
    assert header.count("const aclIntArray *cuSeqlensOptional") == 2
    assert header.count("const aclIntArray *chunkIndicesOutOptional") == 2
    assert "const aclTensor *cuSeqlensOptional" not in header
    assert "const aclTensor *chunkIndicesOutOptional" not in header
