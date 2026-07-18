"""Static op_host contract for chunk_fwd_o; device execution lives in accuracy/routes."""

from pathlib import Path

from tests.operators.chunk_fwd_o.common.case_matrix import manifest


ROOT = Path(__file__).resolve().parents[5]
OP_ROOT = ROOT / "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o"


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


def test_chunk_metadata_uses_chunk_indices_name():
    sources = [path for path in OP_ROOT.rglob("*") if path.suffix in {".cpp", ".h", ".hpp", ".md"}]
    text = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    assert 'Input("chunk_indices")' in text
    assert "chunk_offsets" not in text
    assert "chunkOffsets" not in text
